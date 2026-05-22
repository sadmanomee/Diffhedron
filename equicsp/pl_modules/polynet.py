import torch
import torch.nn as nn
import math
from torch_scatter import scatter
from torch_geometric.nn import radius_graph


class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies=10, n_space=1):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CSPLayer(nn.Module):
    """
    Standard Message Passing Layer adapted for Scalar-only features (E(3) Invariant).
    """

    def __init__(
        self, hidden_dim=128, act_fn=nn.SiLU(), dis_emb=None, ln=False, attn=False
    ):
        super(CSPLayer, self).__init__()

        self.dis_dim = dis_emb.dim if dis_emb else 1

        # Input: [h_i, h_j, distance_embedding]
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )

        # New Attention MLP
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + self.dis_dim, 1),  # Outputs 1 scalar score
            nn.Sigmoid(),
        )

        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

        self.attn = attn

    def forward(self, node_features, edge_index, dist_emb):
        # 1. Edge Model
        # features are purely invariant scalars
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]

        edge_input = torch.cat([hi, hj, dist_emb], dim=1)
        edge_features = self.edge_mlp(edge_input)

        if self.attn:
            attn_scores = self.att_mlp(edge_input)
            edge_features = edge_features * attn_scores

        # 2. Node Aggregation
        agg = scatter(
            edge_features,
            edge_index[0],
            dim=0,
            reduce="mean",
            dim_size=node_features.shape[0],
        )
        agg = torch.cat([node_features, agg], dim=1)

        # 3. Node Update
        node_output = self.node_mlp(agg)

        if self.ln:
            node_output = self.layer_norm(node_output)

        return node_features + node_output


class PolyNet(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        latent_dim=256,  # Added to match input embedding size
        num_layers=4,
        max_atoms=100,
        act_fn="silu",
        num_freqs=10,
        cutoff=6.0,
        max_neighbors=20,
        ln=True,
        edge_style="fc",
    ):
        super(PolyNet, self).__init__()

        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.edge_style = edge_style
        self.act_fn = nn.SiLU() if act_fn == "silu" else nn.ReLU()

        # 1. Embeddings
        self.node_embedding = nn.Embedding(max_atoms, hidden_dim)

        # Note: We do NOT use an internal SinusoidsEmbedding for time anymore,
        # because 'time_emb' is passed in already embedded (size: latent_dim).

        # Project concatenated [AtomEmb, TimeEmb] -> [Hidden]
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)

        # Distance Embedding (for RBF-like features)
        self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs, n_space=1)

        # 2. Message Passing Layers
        self.layers = nn.ModuleList(
            [
                CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln)
                for _ in range(num_layers)
            ]
        )

        # 3. Final Equivariant Head
        # Predicts SCALAR weights for radial vectors (x_i - x_j)
        self.final_edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + self.dis_emb.dim, hidden_dim),
            self.act_fn,
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def gen_edges(self, pos, batch):
        if self.edge_style == "radius":
            from torch_geometric.nn import radius_graph

            edge_index = radius_graph(
                pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_neighbors
            )
        elif self.edge_style == "fc":
            # Fully connected logic
            mask = batch.unsqueeze(0) == batch.unsqueeze(1)
            mask.fill_diagonal_(False)
            edge_index = mask.nonzero(as_tuple=False).t()
        else:
            raise ValueError(
                f"Unimplemented/Unknown edge style: {self.edge_style} - must be one of ['radius', 'fc']"
            )

        j, i = edge_index
        edge_vec = pos[j] - pos[i]  # Vector: x_j - x_i
        edge_dist = edge_vec.norm(dim=-1, p=2)  # Scalar: ||x_j - x_i||

        return edge_index, edge_vec, edge_dist

    def forward(self, time_emb, atom_types, pos, num_atoms, batch):
        """
        Args:
            time_emb:   [Batch, TimeDim] (Already embedded)
            atom_types: [TotalNodes]
            pos:        [TotalNodes, 3]
            num_atoms:  [Batch] (Number of atoms per graph)
            batch:      [TotalNodes] (Graph index for each node)
        """

        # 1. Graph Construction (Cartesian)
        edge_index, edge_vec, edge_dist = self.gen_edges(pos, batch)
        dist_emb = self.dis_emb(edge_dist)  # Invariant features

        # 2. Initial Node Features
        h = self.node_embedding(atom_types)  # [TotalNodes, HiddenDim]

        # 3. Handle Time Embedding (The fix)
        # time_emb is [Batch, TimeDim]. We need [TotalNodes, TimeDim].
        # We use repeat_interleave with num_atoms, exactly like original CSPNet.
        t_per_node = time_emb.repeat_interleave(num_atoms, dim=0)

        # Concatenate and Project
        h = torch.cat([h, t_per_node], dim=1)
        h = self.atom_latent_emb(h)

        # 4. Message Passing
        for layer in self.layers:
            h = layer(h, edge_index, dist_emb)

        # 5. Equivariant Output Construction
        # We need to predict a vector field.
        # We predict scalar weights w_ij based on invariant features (h_i, h_j, d_ij).
        # Then Output_i = Sum_j ( (x_j - x_i) * w_ij )

        hi, hj = h[edge_index[0]], h[edge_index[1]]
        edge_input = torch.cat([hi, hj, dist_emb], dim=1)

        # Predict scalar weights [NumEdges, 1]
        edge_weights = self.final_edge_mlp(edge_input)

        # Create weighted directional vectors [NumEdges, 3]
        # (x_j - x_i) is the vector, edge_weights is the magnitude
        weighted_vectors = edge_vec * edge_weights

        # Aggregate back to nodes [TotalNodes, 3]
        # This summation is permutation invariant and E(3) equivariant
        noise_out = scatter(
            weighted_vectors, edge_index[0], dim=0, reduce="mean", dim_size=h.shape[0]
        )

        return noise_out, None
