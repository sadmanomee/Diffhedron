# # egnn_decoder.py
# """
# EGNNDenoiser compatible with PolyDiffusion_new.forward(...)

# Decoder contract (call exactly as in your diffusion code):
#     pred_c = decoder(
#         time_input,          # either (B, time_in_dim) precomputed embedding OR (B,) / (B,1) scalar times
#         atom_types,          # (N_total,) long tensor with Z numbers
#         input_cart_coords,   # (N_total, 3) centered coordinates (no batch dim)
#         num_atoms,           # iterable/tensor with per-graph node counts, length B
#         node2graph,          # (N_total,) long mapping node -> graph index in [0..B-1]
#     )
# Returns:
#     pred_c: tensor (N_total, 3) - predicted noise vectors per node
# """

# import math
# import torch
# import torch.nn as nn
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import degree
# from torch_scatter import scatter

# MAX_ATOMIC_NUM = 100


# class SinusoidalPosEmb(nn.Module):
#     """Embed scalar times -> sinusoidal vector (user-provided style)"""

#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         x = x.reshape(-1) * 1000.0
#         device = x.device
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb  # (B, dim)


# class RadialFourierEmbedding(nn.Module):
#     def __init__(self, n_frequencies=16, max_radius=10.0, logspace=True):
#         super().__init__()
#         if logspace:
#             freq = (
#                 2.0
#                 * math.pi
#                 * torch.logspace(-1, math.log10(n_frequencies), n_frequencies)
#             )
#         else:
#             freq = 2.0 * math.pi * torch.arange(1, n_frequencies + 1).float()
#         self.register_buffer("freq_buf", freq)
#         self.max_radius = max_radius
#         self.out_dim = 2 * n_frequencies

#     def forward(self, r):  # r: (E,) or (E,1)
#         if r.dim() == 2 and r.size(1) == 1:
#             r = r.squeeze(1)
#         r = r.clamp(min=0.0)
#         x = r[:, None] / (self.max_radius + 1e-8)
#         angles = x * self.freq_buf[None, :]
#         return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (E, 2*n_freq)


# class EGNNLayer(MessagePassing):
#     """Single EGNN layer that updates positions x and features h."""

#     def __init__(self, hidden_dim, edge_attr_dim=0, aggr="add"):
#         super().__init__(aggr=aggr)
#         in_dim = 2 * hidden_dim + 1 + (edge_attr_dim if edge_attr_dim else 0)
#         self.phi_e = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.SiLU(),
#         )
#         self.phi_x = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, 1),
#         )
#         self.phi_h = nn.Sequential(
#             nn.Linear(hidden_dim + hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )

#     def forward(self, x, h, edge_index, edge_attr=None, c=None):
#         if c is None:
#             c = degree(edge_index[0], x.size(0)).unsqueeze(-1).clamp(min=1.0)
#         out = self.propagate(edge_index=edge_index, x=x, h=h, edge_attr=edge_attr)
#         mx_agg = out[:, :3]
#         mh_agg = out[:, 3:]
#         h_new = self.phi_h(torch.cat([h, mh_agg], dim=-1))
#         x_new = x + (mx_agg / c)
#         return x_new, h_new

#     def message(self, x_i, x_j, h_i, h_j, edge_attr):
#         r2 = torch.sum((x_i - x_j) ** 2, dim=-1, keepdim=True)
#         if edge_attr is None or edge_attr.size(1) == 0:
#             inp = torch.cat([h_i, h_j, r2], dim=-1)
#         else:
#             inp = torch.cat([h_i, h_j, r2, edge_attr], dim=-1)
#         mh = self.phi_e(inp)
#         w = self.phi_x(mh)
#         mx = (x_i - x_j) * w
#         return torch.cat([mx, mh], dim=-1)


# class EGNNDenoiser(nn.Module):
#     """
#     EGNN denoiser that matches your diffusion decoder callsite.
#     Accepts precomputed time embeddings or scalar times. Instantiation parameters:
#       hidden_dim, n_layers, time_emb_dim (internal), atom_embed_dim,
#       rbf_freqs, max_radius, use_edge_attr, time_in_dim (if you will pass precomputed time embeddings),
#       cutoff
#     """

#     def __init__(
#         self,
#         hidden_dim=128,
#         latent_dim=256,
#         n_layers=4,
#         time_emb_dim=64,
#         atom_embed_dim=64,
#         rbf_freqs=16,
#         max_radius=7.0,
#         use_edge_attr=True,
#         time_in_dim=None,
#         cutoff=6.0,
#     ):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.time_emb_dim = time_emb_dim
#         self.atom_embed_dim = atom_embed_dim
#         self.use_edge_attr = use_edge_attr
#         self.cutoff = cutoff

#         # atom embedding: Z -> vector
#         self.atom_emb = nn.Embedding(MAX_ATOMIC_NUM + 1, atom_embed_dim)

#         # internal scalar->embedding (if user passes scalar times)
#         self.time_posemb = SinusoidalPosEmb(time_emb_dim)
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_emb_dim, time_emb_dim),
#             nn.SiLU(),
#             nn.Linear(time_emb_dim, time_emb_dim),
#         )

#         # if incoming precomputed time embeddings exist, optionally project them
#         if time_in_dim is None:
#             self.time_proj = None
#             self.time_in_dim = time_emb_dim
#         else:
#             self.time_in_dim = time_in_dim
#             if self.time_in_dim == self.time_emb_dim:
#                 self.time_proj = None
#             else:
#                 self.time_proj = nn.Linear(self.time_in_dim, self.time_emb_dim)

#         # project node input (atom_emb + time_emb) -> hidden
#         self.node_proj = nn.Sequential(
#             nn.Linear(atom_embed_dim + time_emb_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )

#         # edge rbf
#         if use_edge_attr:
#             self.rbf = RadialFourierEmbedding(
#                 n_frequencies=rbf_freqs, max_radius=max_radius
#             )
#             edge_attr_dim = self.rbf.out_dim
#         else:
#             self.rbf = None
#             edge_attr_dim = 0

#         # stack EGNN layers
#         self.layers = nn.ModuleList(
#             [
#                 EGNNLayer(hidden_dim, edge_attr_dim=edge_attr_dim)
#                 for _ in range(n_layers)
#             ]
#         )

#         # final head -> 3D vector
#         self.to_coord = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, 3, bias=False),
#         )

#     def gen_edges(self, num_atoms, cart_coords):
#         """
#         Build cutoff neighbor edges per graph.
#         Returns edge_index (2,E), edge_attr (E,rbf_dim or 0), edge_graph (E,)
#         """
#         edge_src = []
#         edge_dst = []
#         edge_attr_list = []
#         edge_graph = []

#         device = cart_coords.device
#         start = 0
#         for g_idx, n in enumerate(num_atoms):
#             n_local = int(n.item()) if isinstance(n, torch.Tensor) else int(n)
#             if n_local <= 1:
#                 start += n_local
#                 continue
#             coords = cart_coords[start : start + n_local]
#             dists = torch.cdist(coords, coords)
#             mask = (dists <= self.cutoff) & (dists > 0.0)
#             if mask.sum() == 0:
#                 start += n_local
#                 continue
#             idx_i, idx_j = torch.nonzero(mask, as_tuple=True)
#             gi = idx_i + start
#             gj = idx_j + start
#             edge_src.append(gi)
#             edge_dst.append(gj)
#             if self.rbf is not None:
#                 r_vals = dists[idx_i, idx_j]
#                 edge_attr_list.append(self.rbf(r_vals))
#             edge_graph.append(
#                 torch.full((gi.size(0),), g_idx, dtype=torch.long, device=device)
#             )
#             start += n_local

#         if len(edge_src) == 0:
#             eidx = cart_coords.new_zeros((2, 0), dtype=torch.long)
#             eattr = cart_coords.new_zeros((0, self.rbf.out_dim if self.rbf else 0))
#             egraph = cart_coords.new_zeros((0,), dtype=torch.long)
#             return eidx, eattr, egraph

#         src = torch.cat(edge_src, dim=0)
#         dst = torch.cat(edge_dst, dim=0)
#         edge_index = torch.stack([src, dst], dim=0)
#         if self.rbf is not None and len(edge_attr_list) > 0:
#             edge_attr = torch.cat(edge_attr_list, dim=0)
#         else:
#             edge_attr = cart_coords.new_zeros((edge_index.size(1), 0))
#         edge_graph = (
#             torch.cat(edge_graph, dim=0)
#             if len(edge_graph) > 0
#             else cart_coords.new_zeros((edge_index.size(1),), dtype=torch.long)
#         )
#         return edge_index, edge_attr, edge_graph

#     def forward(
#         self, t, atom_types, cart_coords, num_atoms, node2graph, mean_operate=False
#     ):
#         """
#         t: (B, time_in_dim) precomputed embedding OR (B,) / (B,1) scalar times
#         atom_types: (N_total,) long
#         cart_coords: (N_total, 3)
#         num_atoms: iterable of length B
#         node2graph: (N_total,) long
#         """
#         device = cart_coords.device

#         # edges
#         edges, edge_attr, edge_graph = self.gen_edges(num_atoms, cart_coords)

#         # atom -> embedding
#         if atom_types.dim() == 1:
#             atom_feat = self.atom_emb(atom_types)  # (N, atom_embed_dim)
#         else:
#             atom_feat = atom_types

#         # handle time input
#         if t.dim() == 1 or (t.dim() == 2 and t.size(1) == 1):
#             # scalar times -> sinusoidal pos emb -> mlp
#             t_in = t.float().reshape(-1, 1)
#             t_pe = self.time_posemb(t_in)
#             t_emb = self.time_mlp(t_pe)  # (B, time_emb_dim)
#         else:
#             # t is an embedding (B, T)
#             if t.size(1) == self.time_emb_dim:
#                 t_emb = t
#             else:
#                 if self.time_proj is None:
#                     raise RuntimeError(
#                         f"Received time embedding dim {t.size(1)} but model.time_emb_dim={self.time_emb_dim} and no projection configured. Instantiate with time_in_dim={t.size(1)}"
#                     )
#                 t_emb = self.time_proj(t)

#         # expand per-node
#         t_per_node = t_emb[node2graph]  # (N_total, time_emb_dim)

#         # node features -> hidden
#         node_input = torch.cat(
#             [atom_feat, t_per_node], dim=-1
#         )  # (N, atom_emb + time_emb)
#         # check dims
#         if isinstance(self.node_proj, nn.Sequential) and isinstance(
#             self.node_proj[0], nn.Linear
#         ):
#             expected = self.node_proj[0].in_features
#             if node_input.shape[1] != expected:
#                 raise RuntimeError(
#                     f"node_input dim {node_input.shape[1]} != node_proj expected {expected}. Check atom_embed_dim and time_emb_dim/time_in_dim."
#                 )
#         h = self.node_proj(node_input)  # (N, hidden_dim)

#         x = cart_coords.clone()

#         # EGNN layers
#         for layer in self.layers:
#             x, h = layer(x=x, h=h, edge_index=edges, edge_attr=edge_attr)

#         coord_out = self.to_coord(h)  # (N,3)

#         if mean_operate:
#             context_mean = scatter(coord_out, node2graph, dim=0, reduce="mean")
#             coord_out = coord_out + context_mean.repeat_interleave(
#                 torch.tensor(num_atoms, device=device), dim=0
#             )

#         return coord_out


# egnn_decoder_original_style.py
# Put this file in the same package where `models.py` (with E_GCL_mask) is importable.

import math
import torch
import torch.nn as nn
from torch_scatter import scatter


class MLP(nn.Module):
    """a simple 4-layer MLP"""

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class GCL_basic(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(GCL_basic, self).__init__()

    def edge_model(self, source, target, edge_attr):
        pass

    def node_model(self, h, edge_index, edge_attr):
        pass

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_feat)
        return x, edge_feat


class GCL(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_nf=0,
        act_fn=nn.ReLU(),
        bias=True,
        attention=False,
        t_eq=False,
        recurrent=True,
    ):
        super(GCL, self).__init__()
        self.attention = attention
        self.t_eq = t_eq
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn,
        )
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, 1, bias=bias),
                nn.Sigmoid(),
            )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, output_nf, bias=bias),
        )

        # if recurrent:
        # self.gru = nn.GRUCell(hidden_nf, hidden_nf)

    def edge_model(self, source, target, edge_attr):
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)
        out = self.edge_mlp(edge_in)
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out

    def node_model(self, h, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=1)
        out = self.node_mlp(out)
        if self.recurrent:
            out = out + h
            # out = self.gru(out, h)
        return out


class GCL_rf(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self, nf=64, edge_attr_nf=0, reg=0, act_fn=nn.LeakyReLU(0.2), clamp=False
    ):
        super(GCL_rf, self).__init__()

        self.clamp = clamp
        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(edge_attr_nf + 1, nf), act_fn, layer)
        self.reg = reg

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff**2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        if self.clamp:
            m_ij = torch.clamp(m_ij, min=-100, max=100)
        return m_ij

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        x_out = x + agg - x * self.reg
        return x_out


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.ReLU(),
        recurrent=True,
        coords_weight=1.0,
        attention=False,
        clamp=False,
        norm_diff=False,
        tanh=False,
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        # if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(
            trans, min=-100, max=100
        )  # This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg * self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / (norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr


class E_GCL_vel(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.ReLU(),
        recurrent=True,
        coords_weight=1.0,
        attention=False,
        norm_diff=False,
        tanh=False,
    ):
        E_GCL.__init__(
            self,
            input_nf,
            output_nf,
            hidden_nf,
            edges_in_d=edges_in_d,
            nodes_att_dim=nodes_att_dim,
            act_fn=act_fn,
            recurrent=recurrent,
            coords_weight=coords_weight,
            attention=attention,
            norm_diff=norm_diff,
            tanh=tanh,
        )
        self.norm_diff = norm_diff
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, 1)
        )

    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)

        coord += self.coord_mlp_vel(h) * vel
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr


class GCL_rf_vel(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self, nf=64, edge_attr_nf=0, act_fn=nn.LeakyReLU(0.2), coords_weight=1.0
    ):
        super(GCL_rf_vel, self).__init__()
        self.coords_weight = coords_weight
        self.coord_mlp_vel = nn.Sequential(nn.Linear(1, nf), act_fn, nn.Linear(nf, 1))

        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        # layer.weight.uniform_(-0.1, 0.1)
        self.phi = nn.Sequential(
            nn.Linear(1 + edge_attr_nf, nf), act_fn, layer, nn.Tanh()
        )  # we had to add the tanh to keep this method stable

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        row, col = edge_index
        edge_m = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_m)
        x += vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff**2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        return m_ij

    def node_model(self, x, edge_index, edge_m):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_m, row, num_segments=x.size(0))
        x_out = x + agg * self.coords_weight
        return x_out


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


MAX_ATOMIC_NUM = 100


class SinusoidalPosEmb(nn.Module):
    """Your sinusoidal pos emb for scalar times."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B,) or (B,1)
        x = x.reshape(-1) * 1000.0
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb  # (B, dim)


class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_attr_dim=0,
        act_fn=nn.ReLU(),
        recurrent=True,
        coords_weight=1.0,
        attention=False,
    ):
        E_GCL.__init__(
            self,
            input_nf,
            output_nf,
            hidden_nf,
            edges_in_d=edges_in_d,
            nodes_att_dim=nodes_attr_dim,
            act_fn=act_fn,
            recurrent=recurrent,
            coords_weight=coords_weight,
            attention=attention,
        )

        del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord += agg * self.coords_weight
        return coord

    def forward(
        self,
        h,
        edge_index,
        coord,
        node_mask,
        edge_mask,
        edge_attr=None,
        node_attr=None,
        n_nodes=None,
    ):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask

        # coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNNDenoiser(nn.Module):
    """
    Wrapper decoder that uses E_GCL_mask layers (original QM9 style) to produce equivariant
    coordinate predictions.

    Usage (exact contract used by PolyDiffusion_new.forward):
        pred_c = decoder(time_input, atom_types, input_cart_coords, num_atoms, node2graph)

    Where:
      - time_input: either (B,) / (B,1) scalar times OR (B, T_in) precomputed time embeddings.
      - atom_types: (N_total,) long tensor of Z numbers (1..MAX_ATOMIC_NUM)
      - input_cart_coords: (N_total, 3) centered coords (no padding)
      - num_atoms: iterable/tensor of per-graph node counts (length B)
      - node2graph: (N_total,) long mapping node->graph index in [0..B-1]

    Returns:
      coord_out: (N_total, 3) predicted displacement (noise) per node.
    """

    def __init__(
        self,
        hidden_nf=128,
        latent_dim=256,
        n_layers=4,
        in_node_emb=32,  # atom embedding dim (before projecting to hidden_nf)
        time_emb_dim=64,  # internal time embedding dimension
        time_in_dim=None,  # incoming time embedding dim (if you pass precomputed embeddings)
        cutoff=6.0,
        use_edge_mask=True,
        device="cpu",
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.in_node_emb = in_node_emb
        self.time_emb_dim = time_emb_dim
        self.time_in_dim = time_in_dim if time_in_dim is not None else time_emb_dim
        self.cutoff = cutoff
        self.device = device
        self.use_edge_mask = use_edge_mask

        # embed atom types -> small dense vector
        self.atom_emb = nn.Embedding(MAX_ATOMIC_NUM + 1, in_node_emb)

        # time embedding: supports scalar times or precomputed embeddings
        self.time_posemb = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        if time_in_dim is not None and time_in_dim != time_emb_dim:
            self.time_proj = nn.Linear(time_in_dim, time_emb_dim)
        else:
            self.time_proj = None

        # initial projection: (atom_emb + time_emb) -> hidden_nf (this parallels EGNN.embedding)
        self.embedding = nn.Linear(in_node_emb + time_emb_dim, hidden_nf)

        # build E_GCL_mask layers (like EGNN does)
        # E_GCL_mask(in_nf, out_nf, hidden_nf, edges_in_d=..., nodes_attr_dim=..., act_fn=..., recurrent=True, coords_weight=..., attention=...)
        self.gcl_layers = nn.ModuleList()
        for i in range(self.n_layers):
            # we follow EGNN's use: E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=0, nodes_attr_dim=..., ...)
            # node_attr dims not used here; set nodes_attr_dim=0
            gcl = E_GCL_mask(
                self.hidden_nf,
                self.hidden_nf,
                self.hidden_nf,
                edges_in_d=0,
                nodes_attr_dim=0,
                act_fn=nn.SiLU(),
                recurrent=True,
                coords_weight=1.0,
                attention=False,
            )
            self.gcl_layers.append(gcl)

        # optionally a small MLP for final scalar gating (not required). We will predict displacement via final coords - input_coords.
        # Provide a small output head in case you want an extra learned scalar multiplier:
        self.final_scalar = None
        # self.final_scalar = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf), nn.SiLU(), nn.Linear(self.hidden_nf, 1))

        self.to(self.device)

    # ---------------------------
    # Utility: build flat edge_index (row, col) and edge_mask
    # ---------------------------
    def _build_edge_index_and_mask(self, num_atoms, coords):
        """
        Build edges per graph (cutoff neighbors) and return:
          - edge_index: tuple(row, col) where row/col are 1D LongTensors (E,)
          - edge_mask: tensor (E,1) of ones (or zeros for pruned), float
        coords: (N_total, 3)
        num_atoms: iterable length B
        """
        edge_row = []
        edge_col = []
        edge_mask_list = []
        start = 0
        device = coords.device
        for g_idx, n in enumerate(num_atoms):
            n_local = int(n.item()) if isinstance(n, torch.Tensor) else int(n)
            if n_local <= 1:
                start += n_local
                continue
            c = coords[start : start + n_local]  # (n_local, 3)
            dists = torch.cdist(c, c)  # (n_local, n_local)
            # build mask for off-diagonal pairs within cutoff
            mask = (dists <= self.cutoff) & (dists > 0.0)
            if mask.sum() == 0:
                start += n_local
                continue
            i_idx, j_idx = torch.nonzero(mask, as_tuple=True)
            # convert to global indices
            gi = i_idx + start
            gj = j_idx + start
            edge_row.append(gi)
            edge_col.append(gj)
            # keep edge mask (ones)
            edge_mask_list.append(torch.ones((gi.size(0), 1), device=device))
            start += n_local

        if len(edge_row) == 0:
            # no edges
            empty = coords.new_zeros((1,), dtype=torch.long) * 0
            return (
                coords.new_zeros((0,), dtype=torch.long),
                coords.new_zeros((0,), dtype=torch.long),
            ), coords.new_zeros((0, 1), device=coords.device)

        row = torch.cat(edge_row, dim=0).long().to(device)
        col = torch.cat(edge_col, dim=0).long().to(device)
        edge_mask = torch.cat(edge_mask_list, dim=0).to(device)
        return (row, col), edge_mask

    # ---------------------------
    # forward
    # ---------------------------
    def forward(self, t, atom_types, input_cart_coords, num_atoms, node2graph):
        """
        t: (B,) or (B,1) scalar times OR (B, T_in) precomputed embeddings
        atom_types: (N_total,) long (Z numbers)
        input_cart_coords: (N_total, 3)
        num_atoms: iterable of length B
        node2graph: (N_total,) long mapping node -> graph idx
        """
        device = input_cart_coords.device
        B = len(num_atoms)

        # ---------- edges & masks ----------
        edge_index, edge_mask = self._build_edge_index_and_mask(
            num_atoms, input_cart_coords
        )
        # edge_index: tuple(row, col) each (E,)
        # edge_mask: (E,1)

        # ---------- node mask (N_total, 1) ----------
        # If some nodes are placeholders with atom_types==0, use that; otherwise all ones.
        node_mask = torch.ones(
            (atom_types.size(0), 1), dtype=input_cart_coords.dtype, device=device
        )
        try:
            # If atom_types has zeros for padding, mask them
            node_mask = (atom_types > 0).float().unsqueeze(-1)
        except Exception:
            pass

        # ---------- build node input features h0 ----------
        # atom embedding
        atom_feat = self.atom_emb(
            atom_types.long().to(device)
        )  # (N_total, in_node_emb)

        # time embedding handling
        if t is None:
            # use zero time embedding
            t_emb = torch.zeros((B, self.time_emb_dim), device=device)
        else:
            if t.dim() == 1 or (t.dim() == 2 and t.size(1) == 1):
                # scalar times -> sinusoidal pos emb -> mlp
                t_in = t.float().reshape(-1, 1).to(device)
                t_pe = self.time_posemb(t_in)  # (B, time_emb_dim)
                t_emb = self.time_mlp(t_pe)  # (B, time_emb_dim)
            else:
                # precomputed embedding
                if t.size(1) == self.time_emb_dim:
                    t_emb = t.to(device)
                else:
                    if self.time_proj is None:
                        # create on-the-fly projection (defensive)
                        self.time_proj = nn.Linear(t.size(1), self.time_emb_dim).to(
                            device
                        )
                    t_emb = self.time_proj(t.to(device))

        # expand time embedding to nodes
        t_per_node = t_emb[node2graph.long().to(device)]  # (N_total, time_emb_dim)

        # combine atom + time, then project to hidden
        node_input = torch.cat(
            [atom_feat, t_per_node], dim=-1
        )  # (N_total, in_node_emb+time_emb_dim)
        h = self.embedding(node_input)  # (N_total, hidden_nf)

        # ---------- coordinates (flattened) ----------
        coords = input_cart_coords.clone().to(device)  # (N_total, 3)
        coords_in = coords.clone()

        # ---------- run GCL layers (propagate) ----------
        for i in range(self.n_layers):
            gcl = self.gcl_layers[i]
            # E_GCL_mask API (from your models.py):
            # forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None)
            # returns: h, coord, edge_attr
            h, coords, _ = gcl(
                h,
                edge_index,
                coords,
                node_mask,
                edge_mask,
                edge_attr=None,
                node_attr=None,
                n_nodes=None,
            )

        # ---------- produce coordinate output: displacement (equivariant) ----------
        coord_out = coords - coords_in  # (N_total, 3)

        # optional fine-scaling by learned scalar
        if self.final_scalar is not None:
            s = self.final_scalar(h)  # (N_total, 1)
            coord_out = coord_out * s

        return coord_out
