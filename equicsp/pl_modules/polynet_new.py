# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import math
# from torch_scatter import scatter
# from torch_geometric.utils import dense_to_sparse
# from einops import repeat

# MAX_ATOMIC_NUM = 100


# class SinusoidsEmbedding(nn.Module):
#     def __init__(self, n_frequencies=10, n_space=3):
#         super().__init__()
#         self.n_frequencies = n_frequencies
#         self.n_space = n_space
#         self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
#         self.dim = self.n_frequencies * 2 * self.n_space

#     def forward(self, x):
#         emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
#         emb = emb.reshape(-1, self.n_frequencies * self.n_space)
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb


# class CSPLayer(nn.Module):
#     """Message passing layer for PolyNet — adapted to use cartesian edge vectors,
#     but keeps a 6-d placeholder so older checkpoints remain loadable.
#     """

#     def __init__(self, hidden_dim=128, act_fn=nn.SiLU(), dis_emb=None, ln=False):
#         super(CSPLayer, self).__init__()

#         self.dis_dim = 3
#         self.dis_emb = dis_emb
#         if dis_emb is not None:
#             self.dis_dim = dis_emb.dim

#         # Keep 6 dims reserved for lattice/IP features to remain backward-compatible
#         lattice_feat_dim = 6

#         # Now edge input: hi, hj, (6 lattice/IP dims), edge embedding
#         in_dim = hidden_dim * 2 + lattice_feat_dim + self.dis_dim
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             act_fn,
#             nn.Linear(hidden_dim, hidden_dim),
#             act_fn,
#         )
#         self.node_mlp = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             act_fn,
#             nn.Linear(hidden_dim, hidden_dim),
#             act_fn,
#         )
#         self.ln = ln
#         if self.ln:
#             self.layer_norm = nn.LayerNorm(hidden_dim)

#     def edge_model(
#         self,
#         node_features,
#         cart_coords,
#         edge_index,
#         edge2graph,
#         edge_vec=None,
#         lattice_ips_flatten_edges=None,
#     ):
#         """
#         edge_vec: (E,3) Cartesian vector j - i for each edge (matching edge_index order)
#         lattice_ips_flatten_edges: optional (E,6) tensor of lattice/IP features.
#         If None, we use zeros to keep compatibility with older checkpoints.
#         """

#         hi = node_features[edge_index[0]]
#         hj = node_features[edge_index[1]]

#         if edge_vec is None:
#             # compute j-i in cart coords if edge_vec not provided
#             xi = cart_coords[edge_index[1]]  # j
#             xj = cart_coords[edge_index[0]]  # i
#             edge_vec = xj - xi  # j - i

#         # optionally apply distance embedding on edge_vec (sinusoidal)
#         if self.dis_emb is not None:
#             edge_feat_vec = self.dis_emb(edge_vec)
#         else:
#             edge_feat_vec = edge_vec

#         # If lattice/IP features provided, use them; else use zeros for backward compatibility
#         if lattice_ips_flatten_edges is None:
#             # E is number of edges
#             E = edge_index.shape[1]
#             device = node_features.device
#             lattice_ips_flatten_edges = torch.zeros(
#                 (E, 6), dtype=node_features.dtype, device=device
#             )

#         edges_input = torch.cat(
#             [hi, hj, lattice_ips_flatten_edges, edge_feat_vec], dim=1
#         )
#         edge_features = self.edge_mlp(edges_input)
#         return edge_features

#     def node_model(self, node_features, edge_features, edge_index):
#         agg = scatter(
#             edge_features,
#             edge_index[0],
#             dim=0,
#             reduce="mean",
#             dim_size=node_features.shape[0],
#         )
#         agg = torch.cat([node_features, agg], dim=1)
#         out = self.node_mlp(agg)
#         return out

#     def forward(
#         self,
#         node_features,
#         cart_coords,
#         edge_index,
#         edge2graph,
#         edge_vec=None,
#         lattice_ips_flatten_edges=None,
#     ):
#         node_input = node_features
#         if self.ln:
#             node_features = self.layer_norm(node_input)
#         edge_features = self.edge_model(
#             node_features,
#             cart_coords,
#             edge_index,
#             edge2graph,
#             edge_vec=edge_vec,
#             lattice_ips_flatten_edges=lattice_ips_flatten_edges,
#         )
#         node_output = self.node_model(node_features, edge_features, edge_index)
#         return node_input + node_output


# class PolyNet_new(nn.Module):
#     """
#     PolyNet adapted for cartesian-only diffusion.
#     - No lattice outputs.
#     - Uses cartesian coords to build neighbor graph if needed (knn_nopbc).
#     - forward signature: (t, atom_types, cart_coords, num_atoms, node2graph, mean_operate=False)
#     - returns: (coord_out, log_out)  OR (coord_out, log_out, type_out) if pred_type
#     """

#     def __init__(
#         self,
#         hidden_dim=128,
#         latent_dim=256,
#         num_layers=4,
#         max_atoms=100,
#         act_fn="silu",
#         dis_emb="sin",
#         num_freqs=10,
#         edge_style="fc",
#         cutoff=6.0,
#         max_neighbors=20,
#         ln=False,
#         ip=False,
#         smooth=False,
#         pred_type=False,
#         pred_scalar=False,
#         type_mlp=False,
#     ):
#         super(PolyNet_new, self).__init__()

#         self.ip = ip
#         self.smooth = smooth
#         self.pred_scalar = pred_scalar
#         self.edge_style = edge_style
#         self.cutoff = cutoff
#         self.max_neighbors = max_neighbors
#         self.ln = ln
#         self.pred_type = pred_type

#         if self.smooth:
#             self.node_embedding = nn.Linear(max_atoms, hidden_dim)
#         else:
#             self.node_embedding = nn.Embedding(max_atoms, hidden_dim)

#         self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)

#         if act_fn == "silu":
#             self.act_fn = nn.SiLU()

#         if dis_emb == "sin":
#             self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
#         else:
#             self.dis_emb = None

#         for i in range(0, num_layers):
#             self.add_module(
#                 "csp_layer_%d" % i,
#                 CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln),
#             )
#         self.num_layers = num_layers

#         self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
#         self.log_out = nn.Linear(hidden_dim, 3, bias=False)

#         if self.ln:
#             self.final_layer_norm = nn.LayerNorm(hidden_dim)

#         if self.pred_type:
#             if type_mlp:
#                 self.type_out = nn.Sequential(
#                     nn.Linear(hidden_dim, hidden_dim),
#                     self.act_fn,
#                     nn.Linear(hidden_dim, MAX_ATOMIC_NUM),
#                 )
#             else:
#                 self.type_out = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

#         if self.pred_scalar:
#             self.scalar_out = nn.Linear(hidden_dim, 1)

#     def gen_edges_cart(self, num_atoms, cart_coords):
#         """
#         Build kNN edges (no PBC) from batched cart_coords.
#         Returns: edge_index (2, E) and edge_vec (E,3) (j - i).
#         This is the same logic as the knn_nopbc helper you had, but wired to accept cart_coords directly.
#         """

#         device = cart_coords.device
#         batch_size = len(num_atoms)
#         edges_i = []
#         edges_j = []
#         start = 0
#         for b in range(batch_size):
#             n = int(num_atoms[b].item())
#             if n <= 1:
#                 start += n
#                 continue
#             block = cart_coords[start : start + n]  # (n,3)
#             dists = torch.cdist(block, block, p=2)  # (n,n)
#             diag_mask = torch.eye(n, device=device, dtype=torch.bool)
#             dists.masked_fill_(diag_mask, float("inf"))
#             k = min(self.max_neighbors, n - 1)
#             if k <= 0:
#                 start += n
#                 continue
#             vals, idxs = torch.topk(dists, k=k, largest=False, sorted=False)
#             i_idx = torch.arange(n, device=device).unsqueeze(1).repeat(1, k).reshape(-1)
#             j_idx = idxs.reshape(-1)
#             edges_i.append((i_idx + start).cpu())
#             edges_j.append((j_idx + start).cpu())
#             start += n

#         if len(edges_i) == 0:
#             return torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros(
#                 (0, 3), device=device
#             )

#         edges_i = torch.cat(edges_i).to(device)
#         edges_j = torch.cat(edges_j).to(device)
#         edge_index = torch.stack([edges_j, edges_i], dim=0)  # (2,E) j,i convention

#         edge_vec = cart_coords[edge_index[0]] - cart_coords[edge_index[1]]  # j - i

#         # Make symmetric
#         edge_index_sym = torch.cat(
#             [edge_index, torch.stack([edge_index[1], edge_index[0]], dim=0)], dim=1
#         )
#         edge_vec_sym = torch.cat([edge_vec, -edge_vec], dim=0)

#         return edge_index_sym, edge_vec_sym

#     def forward(
#         self,
#         t,
#         atom_types,
#         cart_coords,
#         num_atoms,
#         node2graph,
#         mean_operate=False,
#         edges_in=None,
#         edge_vec_in=None,
#     ):
#         """
#         New forward:
#         - t: time embedding per-graph (B, time_dim)
#         - atom_types: (total_nodes,)
#         - cart_coords: (total_nodes, 3)
#         - num_atoms: (B,)
#         - node2graph: (total_nodes,) cluster indices
#         - optionally edges_in and edge_vec_in can be provided (precomputed)
#         """

#         # Build or use edges
#         if edges_in is not None and edge_vec_in is not None:
#             edges = edges_in
#             edge_vec = edge_vec_in
#         else:
#             edges, edge_vec = self.gen_edges_cart(num_atoms, cart_coords)

#         edge2graph = node2graph[edges[0]]

#         if self.smooth:
#             node_features = self.node_embedding(atom_types)
#         else:
#             node_features = self.node_embedding(atom_types - 1)

#         t_per_atom = t.repeat_interleave(num_atoms, dim=0)
#         node_features = torch.cat([node_features, t_per_atom], dim=1)
#         node_features = self.atom_latent_emb(node_features)

#         for i in range(0, self.num_layers):
#             node_features = self._modules["csp_layer_%d" % i](
#                 node_features,
#                 cart_coords,
#                 edges,
#                 edge2graph,
#                 edge_vec,
#             )

#         if self.ln:
#             node_features = self.final_layer_norm(node_features)

#         coord_out = self.coord_out(node_features)
#         if mean_operate:
#             context_mean = scatter(coord_out, node2graph, dim=0, reduce="mean")
#             coord_out = coord_out + context_mean.repeat_interleave(num_atoms, dim=0)

#         log_out = self.log_out(node_features)
#         if mean_operate:
#             log_mean = scatter(log_out, node2graph, dim=0, reduce="mean")
#             log_out = log_out + log_mean.repeat_interleave(num_atoms, dim=0)

#         graph_features = scatter(node_features, node2graph, dim=0, reduce="mean")

#         if self.pred_scalar:
#             return self.scalar_out(graph_features)

#         if self.pred_type:
#             type_out = self.type_out(node_features)
#             return coord_out, log_out, type_out

#         return coord_out, log_out


import torch
import torch.nn as nn
import math
from torch_scatter import scatter
from torch_geometric.utils import dense_to_sparse
from einops import repeat

MAX_ATOMIC_NUM = 100


class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies=10, n_space=3):
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


class SinusoidalHybridEmbedding(nn.Module):

    def __init__(self, n_frequencies=10, n_space=3, scale=1000.0):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2.0 * math.pi * torch.arange(self.n_frequencies).float()
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        # x: (B, n_space) OR (B,) if n_space==1
        if x.dim() == 1:
            # treat as (B,1)
            x = x.unsqueeze(-1)
        assert (
            x.dim() == 2 and x.shape[1] == self.n_space
        ), f"x must be (B, {self.n_space}) but got {x.shape}"
        x = x * 1000
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RadialFourierEmbedding(nn.Module):
    def __init__(self, n_frequencies=16, max_radius=10.0, logspace=True):
        super().__init__()
        if logspace:
            self.freq = (
                2.0
                * math.pi
                * torch.logspace(-1, math.log10(n_frequencies), n_frequencies)
            )
        else:
            self.freq = 2.0 * math.pi * torch.arange(1, n_frequencies + 1).float()
        self.register_buffer("freq_buf", self.freq)  # on same device
        self.max_radius = max_radius
        self.out_dim = 2 * n_frequencies
        self.dim = self.out_dim

    def forward(self, r):  # r: (E,) or (E,1)
        if r.dim() == 2 and r.size(1) == 1:
            r = r.squeeze(1)
        r = r.clamp(min=0.0)
        x = r[:, None] / (self.max_radius + 1e-8)  # optional scale to [0,1]
        angles = x * self.freq_buf[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (E, 2*n_freq)


class CSPLayer(nn.Module):
    def __init__(
        self, hidden_dim=128, act_fn=nn.SiLU(), dis_emb=None, ln=False, dropout=0.0
    ):
        super(CSPLayer, self).__init__()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + self.dis_dim + 3, hidden_dim),
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
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    # def edge_model(
    #     self, node_features, cart_coords, edge_index, edge2graph, cart_diff=None
    # ):
    #     hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
    #     if cart_diff is None:
    #         xi, xj = cart_coords[edge_index[0]], cart_coords[edge_index[1]]
    #         cart_diff = xj - xi  # [E,3]
    #     if self.dis_emb is not None:
    #         dis_emb = self.dis_emb(cart_diff)  # [E, dis_dim]
    #     else:
    #         dis_emb = cart_diff  # fallback
    #     edges_input = torch.cat([hi, hj, dis_emb, cart_diff], dim=1)
    #     edge_features = self.edge_mlp(edges_input)
    #     return edge_features

    def edge_model(
        self, node_features, cart_coords, edge_index, edge2graph, cart_diff=None
    ):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if cart_diff is None:
            xi, xj = cart_coords[edge_index[0]], cart_coords[edge_index[1]]
            cart_diff = xj - xi  # [E,3]

        # Compute invariant scalar distance
        r = torch.norm(cart_diff, dim=1, keepdim=True)  # shape [E,1]

        if self.dis_emb is not None:
            dis_emb = self.dis_emb(r)  # [E, dis_dim]  <-- use radius, not vector
        else:
            dis_emb = r  # fallback: use scalar radius

        # Note: Keep cart_diff available if you still want to include vector info in edges_input.
        edges_input = torch.cat([hi, hj, dis_emb, cart_diff], dim=1)
        edge_features = self.edge_mlp(edges_input)
        edge_features = self.dropout(edge_features)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter(
            edge_features,
            edge_index[0],
            dim=0,
            reduce="mean",
            dim_size=node_features.shape[0],
        )
        agg = torch.cat([node_features, agg], dim=1)
        out = self.node_mlp(agg)
        out = self.dropout(out)
        return out

    def forward(
        self, node_features, cart_coords, edge_index, edge2graph, cart_diff=None
    ):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features, cart_coords, edge_index, edge2graph, cart_diff
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class PolyNet_new(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        num_layers=4,
        max_atoms=100,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=10,
        edge_style="fc",
        cutoff=6.0,
        max_neighbors=20,
        ln=False,
        ip=False,
        smooth=False,
        pred_type=False,
        pred_scalar=False,
        type_mlp=False,
        dropout=0.0,
        tanh=False,
        residual=False,
        attention=False,
        coords_aggr=False,
    ):
        super(PolyNet_new, self).__init__()
        self.ip = ip
        self.tanh = tanh
        self.residual = residual
        self.attention = attention
        self.coords_aggr = coords_aggr
        self.smooth = smooth
        self.pred_scalar = pred_scalar
        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.act_fn = nn.SiLU() if act_fn == "silu" else nn.ReLU()
        if dis_emb == "sin":
            # self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs, n_space=3)
            # self.dis_emb = SinusoidalHybridEmbedding(n_frequencies=num_freqs, n_space=3)
            self.dis_emb = RadialFourierEmbedding(
                n_frequencies=num_freqs, max_radius=7.0
            )
            self.dis_dim = self.dis_emb.out_dim
        else:
            self.dis_emb = None
            self.dis_dim = 1

        self.edge_vec_mlp = nn.Sequential(
            nn.Linear(
                hidden_dim * 2 + (self.dis_dim if self.dis_emb else 1), hidden_dim
            ),
            self.act_fn,
            nn.Linear(hidden_dim, 1),  # scalar weight per edge
        )
        for i in range(num_layers):
            self.add_module(
                f"csp_layer_{i}",
                CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, dropout=dropout),
            )
        self.num_layers = num_layers
        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)  # predicted noise / score
        self.log_out = nn.Linear(hidden_dim, 3, bias=False)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.ln = ln
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.pred_type = pred_type
        if self.pred_type:
            if type_mlp:
                self.type_out = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    self.act_fn,
                    nn.Linear(hidden_dim, MAX_ATOMIC_NUM),
                )
            else:
                self.type_out = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)
        if self.pred_scalar:
            self.scalar_out = nn.Linear(hidden_dim, 1)
        self.edge_style = edge_style

    def gen_edges(self, num_atoms, cart_coords, node2graph):
        if self.edge_style == "fc":
            lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (cart_coords[fc_edges[1]] - cart_coords[fc_edges[0]])

            # # # Optimized FC Graph (Vectorized, No Block Diag)
            # # # 1. Create mask where (i, j) are in the same graph
            # # mask = node2graph.unsqueeze(0) == node2graph.unsqueeze(1)

            # # # 2. FIX: Remove self-loops (consistent with KNN)
            # # mask.fill_diagonal_(False)

            # # # 3. Get indices
            # # edge_index = mask.nonzero(as_tuple=False).t()

            # # # 4. Calculate Vector (Target - Source)
            # # j, i = edge_index
            # cart_diff = cart_coords[j] - cart_coords[i]

            # return edge_index, cart_diff
        elif self.edge_style == "knn":
            # edge_index_list = []
            # diff_list = []
            # start = 0
            # for i, n in enumerate(num_atoms):
            #     n = int(n.item()) if isinstance(n, torch.Tensor) else int(n)
            #     coords = cart_coords[start : start + n]
            #     if n == 0:
            #         start += n
            #         continue
            #     dists = torch.cdist(coords, coords)
            #     mask = (dists <= self.cutoff) & (dists > 0.0)
            #     if mask.sum() == 0:
            #         start += n
            #         continue
            #     idx_i, idx_j = torch.nonzero(mask, as_tuple=True)
            #     global_i = idx_i + start
            #     global_j = idx_j + start
            #     local_edge_index = torch.stack([global_i, global_j], dim=0)
            #     edge_index_list.append(local_edge_index)
            #     diff_list.append(cart_coords[global_j] - cart_coords[global_i])
            #     start += n
            # if len(edge_index_list) == 0:
            #     return torch.zeros(
            #         (2, 0), dtype=torch.long, device=cart_coords.device
            #     ), torch.zeros((0, 3), device=cart_coords.device)
            # edge_index = torch.cat(edge_index_list, dim=1)
            # cart_diff = torch.cat(diff_list, dim=0)
            # return edge_index, -cart_diff

            edge_index_list = []
            diff_list = []
            start = 0

            for i, n in enumerate(num_atoms):
                n = int(n.item()) if isinstance(n, torch.Tensor) else int(n)

                if n == 0:
                    start += n
                    continue

                coords = cart_coords[start : start + n]
                dists = torch.cdist(coords, coords)

                # Mask excludes self-loops (dists > 0)
                mask = (dists <= self.cutoff) & (dists > 1e-5)

                if mask.sum() == 0:
                    start += n
                    continue

                # Note: PyTorch nonzero returns [row, col] -> [i, j]
                idx_i, idx_j = torch.nonzero(mask, as_tuple=True)

                global_i = idx_i + start
                global_j = idx_j + start

                local_edge_index = torch.stack([global_i, global_j], dim=0)
                edge_index_list.append(local_edge_index)

                # FIX: Consistent Direction (Target - Source)
                diff_list.append(cart_coords[global_j] - cart_coords[global_i])

                start += n

            if len(edge_index_list) == 0:
                return torch.zeros(
                    (2, 0), dtype=torch.long, device=cart_coords.device
                ), torch.zeros((0, 3), device=cart_coords.device)

            edge_index = torch.cat(edge_index_list, dim=1)
            cart_diff = torch.cat(diff_list, dim=0)

            # FIX: Removed the negative sign
            return edge_index, cart_diff

    # def gen_edges(self, num_atoms, cart_coords, node2graph):
    #     if self.edge_style == "fc":
    #         lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
    #         fc_graph = torch.block_diag(*lis)
    #         edge_index, _ = dense_to_sparse(fc_graph)
    #         cart_diff = (
    #             cart_coords[edge_index[1]] - cart_coords[edge_index[0]]
    #         )  # x_j - x_i
    #         return edge_index, cart_diff

    #     elif self.edge_style == "knn":
    #         edge_index_list, diff_list = [], []
    #         start = 0
    #         for n in num_atoms.tolist():
    #             coords = cart_coords[start : start + n]
    #             if n > 0:
    #                 dists = torch.cdist(coords, coords)
    #                 mask = (dists <= self.cutoff) & (dists > 0.0)
    #                 if mask.any():
    #                     i, j = torch.nonzero(mask, as_tuple=True)
    #                     gi, gj = i + start, j + start
    #                     edge_index_list.append(torch.stack([gi, gj], dim=0))
    #                     diff_list.append(cart_coords[gj] - cart_coords[gi])  # x_j - x_i
    #             start += n
    #         if not edge_index_list:
    #             return torch.zeros(
    #                 (2, 0), dtype=torch.long, device=cart_coords.device
    #             ), torch.zeros((0, 3), device=cart_coords.device)
    #         edge_index = torch.cat(edge_index_list, dim=1)
    #         cart_diff = torch.cat(diff_list, dim=0)
    #         return edge_index, cart_diff

    def forward(
        self, t, atom_types, cart_coords, num_atoms, node2graph, mean_operate=False
    ):
        edges, cart_diff = self.gen_edges(num_atoms, cart_coords, node2graph)
        edge2graph = (
            node2graph[edges[0]] if edges.size(1) > 0 else node2graph.new_zeros(0)
        )
        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - 1)
        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)
        for i in range(0, self.num_layers):
            node_features = self._modules[f"csp_layer_{i}"](
                node_features, cart_coords, edges, edge2graph, cart_diff=cart_diff
            )
        if self.ln:
            node_features = self.final_layer_norm(node_features)
        coord_out = self.coord_out(node_features)
        if mean_operate:
            context_mean = scatter(coord_out, node2graph, dim=0, reduce="mean")
            coord_out = coord_out + context_mean.repeat_interleave(num_atoms, dim=0)
        log_out = self.log_out(node_features)
        if mean_operate:
            log_mean = scatter(log_out, node2graph, dim=0, reduce="mean")
            log_out = log_out + log_mean.repeat_interleave(num_atoms, dim=0)
        graph_features = scatter(node_features, node2graph, dim=0, reduce="mean")
        if self.pred_scalar:
            return self.scalar_out(graph_features)
        if self.pred_type:
            type_out = self.type_out(node_features)
            return coord_out, log_out, type_out
        return coord_out, log_out

    # def forward(
    #     self, t, atom_types, cart_coords, num_atoms, node2graph, mean_operate=False
    # ):
    #     edges, cart_diff = self.gen_edges(
    #         num_atoms, cart_coords, node2graph
    #     )  # x_j - x_i
    #     edge2graph = (
    #         node2graph[edges[0]] if edges.size(1) > 0 else node2graph.new_zeros(0)
    #     )

    #     # Node embeddings + time (graph-level) -> per-node
    #     node_features = (
    #         self.node_embedding(atom_types - 1)
    #         if not self.smooth
    #         else self.node_embedding(atom_types)
    #     )
    #     t_per_atom = t.repeat_interleave(num_atoms, dim=0)
    #     node_features = self.atom_latent_emb(
    #         torch.cat([node_features, t_per_atom], dim=1)
    #     )

    #     # CSP stack
    #     for i in range(self.num_layers):
    #         node_features = self._modules[f"csp_layer_{i}"](
    #             node_features, cart_coords, edges, edge2graph, cart_diff=cart_diff
    #         )
    #     if self.ln:
    #         node_features = self.final_layer_norm(node_features)

    #     # -------- equivariant vector head --------
    #     if edges.size(1) == 0:
    #         vel = torch.zeros_like(cart_coords)
    #     else:
    #         hi = node_features[edges[0]]
    #         hj = node_features[edges[1]]
    #         r = torch.norm(cart_diff, dim=1, keepdim=True)  # (E,1), invariant

    #         if self.dis_emb is not None:
    #             r_feat = self.dis_emb(r.squeeze(1))  # (E, dis_dim)
    #         else:
    #             r_feat = r  # (E,1)

    #         phi_in = torch.cat([hi, hj, r_feat], dim=1)  # invariants only
    #         w_ij = self.edge_vec_mlp(phi_in)  # (E,1) scalar weights
    #         vec_msg = w_ij * cart_diff  # (E,3)
    #         # aggregate to nodes i
    #         vel = scatter(
    #             vec_msg, edges[0], dim=0, reduce="mean", dim_size=node_features.size(0)
    #         )  # (N,3)

    #     # optional: ensure zero-mean per graph (harmless; your outer module also centers)
    #     if mean_operate:
    #         vel_mean = scatter(vel, node2graph, dim=0, reduce="mean")
    #         vel = vel - vel_mean.repeat_interleave(num_atoms, dim=0)

    #     return vel
