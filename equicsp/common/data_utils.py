import numpy as np
import pandas as pd
import networkx as nx
import torch
import copy
from itertools import combinations
import json
import re
from pathlib import Path

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from networkx.algorithms.components import is_connected

from sklearn.metrics import accuracy_score, recall_score, precision_score
from scipy.spatial import cKDTree

from torch_scatter import scatter
from torch_scatter import segment_coo, segment_csr

from p_tqdm import p_umap

from pathos.pools import ProcessPool as Pool
from tqdm import tqdm
from functools import partial

import faulthandler

faulthandler.enable()


# Tensor of unit cells. Assumes 27 cells in -1, 0, 1 offsets in the x and y dimensions
OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]

EPSILON = 1e-5
MAX_ATOMIC_NUM = 100

### NEW: constants for multi-polyhedron packing
MAX_SLOTS = 5  # max unique polyhedra per crystal
ATOMS_PER_SLOT = 13  # max atoms per polyhedron (1 center + 12 neighbors)
TOTAL_ATOMS = MAX_SLOTS * ATOMS_PER_SLOT  # 65
PAD_TOKEN = 0  # atom_type for padding atoms
### END NEW

chemical_symbols = [
    "X",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False
)

BASE_DIR = Path(__file__).resolve().parents[2]
base_data_dir = BASE_DIR / "data" / "poly_data"
polyhedra_connectivity_file = base_data_dir / "polyhedron_connectivity.json"

# Load connectivity file if it exists
MOTIF_SYMBOL_TO_EDGES = {}
if polyhedra_connectivity_file.exists():
    with open(polyhedra_connectivity_file, "r") as f:
        MOTIF_SYMBOL_TO_EDGES = json.load(f)


### NEW: composition vector encoding
def composition_to_vector(comp_str):
    """
    Parse 'Sr1Ti1O3' → (MAX_ATOMIC_NUM,) float vector.
    Index i = count of element with atomic number i+1.
    """
    import re as _re

    vec = np.zeros(MAX_ATOMIC_NUM, dtype=np.float32)
    pattern = r"([A-Z][a-z]?)(\d*)"
    for elem, count in _re.findall(pattern, comp_str):
        if elem and elem in chemical_symbols:
            z = chemical_symbols.index(elem)
            if 0 < z <= MAX_ATOMIC_NUM:
                vec[z - 1] = int(count) if count else 1
    return vec


### END NEW


def compute_rms_pairwise_scale(X, eps=1e-9):
    n = X.shape[0]
    diffs = X[:, None, :] - X[None, :, :]
    d2 = np.sum(diffs**2, axis=-1)
    iu = np.triu_indices(n, k=1)
    rms = np.sqrt(np.mean(d2[iu]))
    return 1.0 / (rms + eps)


def compute_median_nn_scale(X, eps=1e-9):
    tree = cKDTree(X)
    dists, idx = tree.query(X, k=2)
    nn = dists[:, 1]
    med = np.median(nn)
    return 1.0 / (med + eps)


def build_crystal(crystal_str, niggli=True, primitive=False):
    crystal = Structure.from_str(crystal_str, fmt="cif")
    if primitive:
        crystal = crystal.get_primitive_structure()
    if niggli:
        crystal = crystal.get_reduced_structure()
    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return canonical_crystal


def build_crystal_polyhedron(crystal_str, niggli=True, primitive=False):
    crystal = Structure.from_str(crystal_str, fmt="cif")
    if primitive:
        crystal = crystal.get_primitive_structure()
    if niggli:
        crystal = crystal.get_reduced_structure()
    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return canonical_crystal


def refine_spacegroup(crystal, tol=0.01):
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_conventional_standard_structure()
    space_group = spga.get_space_group_number()
    crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return crystal, space_group


def build_crystal_graph(crystal, graph_method="crystalnn"):
    if graph_method == "crystalnn":
        try:
            crystal_graph = StructureGraph.with_local_env_strategy(crystal, CrystalNN)
        except:
            crystalNN_tmp = local_env.CrystalNN(
                distance_cutoffs=None,
                x_diff_weight=-1,
                porous_adjustment=False,
                search_cutoff=10,
            )
            crystal_graph = StructureGraph.with_local_env_strategy(
                crystal, crystalNN_tmp
            )
    elif graph_method == "none":
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(
        crystal.lattice.matrix, lattice_params_to_matrix(*lengths, *angles)
    )

    edge_indices, to_jimages = [], []
    if graph_method != "none":
        for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return (
        frac_coords,
        atom_types,
        lengths,
        angles,
        edge_indices,
        to_jimages,
        num_atoms,
    )


def build_crystal_graph_polyhedron(crystal, graph_method="crystalnn"):
    if graph_method == "crystalnn":
        try:
            crystal_graph = StructureGraph.with_local_env_strategy(crystal, CrystalNN)
        except:
            crystalNN_tmp = local_env.CrystalNN(
                distance_cutoffs=None,
                x_diff_weight=-1,
                porous_adjustment=False,
                search_cutoff=10,
            )
            crystal_graph = StructureGraph.with_local_env_strategy(
                crystal, crystalNN_tmp
            )
    elif graph_method == "none":
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    cart_coords = crystal.cart_coords
    central_atoms = (1,) + (0,) * (len(crystal.atomic_numbers) - 1)
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(
        crystal.lattice.matrix, lattice_params_to_matrix(*lengths, *angles)
    )

    edge_indices, to_jimages = [], []
    if graph_method != "none":
        for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    central_atoms = np.array(central_atoms)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return (
        frac_coords,
        cart_coords,
        central_atoms,
        atom_types,
        lengths,
        angles,
        edge_indices,
        to_jimages,
        num_atoms,
    )


def build_crystal_graph_polyhedron_test(entry, graph_method="poly"):
    center = entry["center"]
    neighbors = entry["neighbors"]
    num_atoms = 1 + len(neighbors)

    if Element(center["element"]).Z > MAX_ATOMIC_NUM:
        return None
    else:
        atom_types = [Element(center["element"]).Z]
    for n in neighbors:
        atomic_num = Element(n["element"]).Z
        if atomic_num > MAX_ATOMIC_NUM:
            return None
        else:
            atom_types.append(atomic_num)
    atom_types = np.array(atom_types)

    coords = [center["coords"]]
    coords += [n["coords"] for n in neighbors]
    cart_coords = np.array(coords, dtype=float)

    centroid = cart_coords.mean(axis=0, keepdims=True)
    cart_coords = cart_coords - centroid

    frac_coords = np.random.rand(num_atoms, 3)
    central_atoms = np.zeros(num_atoms, dtype=int)
    central_atoms[0] = Element(center["element"]).Z
    lengths = np.random.rand(3) * 5 + 2
    angles = np.random.rand(3) * 60 + 60

    edge_indices = []
    to_jimages = []

    for i in range(1, num_atoms):
        edge_indices.append([0, i])
        edge_indices.append([i, 0])
        to_jimages.append((0, 0, 0))
        to_jimages.append((0, 0, 0))

    motif_symbol = entry.get("motif_symbol")
    permutation = entry.get("permutation")

    if motif_symbol in MOTIF_SYMBOL_TO_EDGES and permutation:
        conn_info = MOTIF_SYMBOL_TO_EDGES[motif_symbol]
        edges = conn_info["edges"]
        expected_cn = conn_info["coordination_number"]

        if len(permutation) != expected_cn:
            return None

        for i, j in edges:
            li, lj = permutation[i], permutation[j]
            if li < len(neighbors) and lj < len(neighbors):
                edge_indices.append([li + 1, lj + 1])
                edge_indices.append([lj + 1, li + 1])
                to_jimages.append((0, 0, 0))
                to_jimages.append((0, 0, 0))
    else:
        return None

    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)

    return (
        frac_coords,
        cart_coords,
        central_atoms,
        atom_types,
        lengths,
        angles,
        edge_indices,
        to_jimages,
        num_atoms,
    )


### NEW: Pack all polyhedra of one crystal into a fixed-size 65-atom sample
def pack_crystal_polyhedra(crystal_entry):
    """
    Pack all unique polyhedra of one crystal into fixed-size tensors.

    Args:
        crystal_entry: dict with 'composition' and 'polyhedra' list
            Each polyhedron has: center (element, coords), neighbors [{element, coords}]

    Returns:
        dict with packed tensors, or None if invalid
    """
    composition = crystal_entry.get("composition", "")
    polyhedra = crystal_entry.get("polyhedra", [])

    if not composition or not polyhedra:
        return None

    # Filter: max 5 polyhedra, max 12 neighbors each
    valid_polyhedra = []
    for poly in polyhedra:
        center = poly.get("center", {})
        neighbors = poly.get("neighbors", [])
        center_elem = center.get("element", "")

        if not center_elem or not neighbors:
            continue
        if len(neighbors) > ATOMS_PER_SLOT - 1:
            continue

        # Check all elements are valid
        try:
            center_z = Element(center_elem).Z
            if center_z > MAX_ATOMIC_NUM:
                continue
            valid = True
            for n in neighbors:
                nz = Element(n["element"]).Z
                if nz > MAX_ATOMIC_NUM:
                    valid = False
                    break
            if not valid:
                continue
        except Exception:
            continue

        valid_polyhedra.append(poly)

    if len(valid_polyhedra) == 0 or len(valid_polyhedra) > MAX_SLOTS:
        return None

    # Sort polyhedra deterministically: by center Z, then by coordination number
    def sort_key(p):
        z = Element(p["center"]["element"]).Z
        cn = len(p["neighbors"])
        return (z, cn)

    valid_polyhedra.sort(key=sort_key)

    n_poly = len(valid_polyhedra)

    # Initialize fixed-size arrays
    atom_types = np.zeros(TOTAL_ATOMS, dtype=np.int64)
    gt_types = np.zeros(TOTAL_ATOMS, dtype=np.int64)
    ### FIX: padding coords set to large sentinel, NOT [0,0,0]
    ### [0,0,0] is the center atom position, so padding there creates ambiguity.
    ### Large values ensure padding atoms are far from real atoms in coordinate space,
    ### making their radial features (sinusoidal embedding of distance) near-zero,
    ### so they contribute nothing to message passing within the slot.
    PAD_COORD_SENTINEL = 99.0
    cart_coords = np.full((TOTAL_ATOMS, 3), PAD_COORD_SENTINEL, dtype=np.float32)
    ### END FIX
    real_mask = np.zeros(TOTAL_ATOMS, dtype=np.int64)
    center_mask = np.zeros(TOTAL_ATOMS, dtype=np.int64)
    slot_ids = np.zeros(TOTAL_ATOMS, dtype=np.int64)
    slot_mask = np.zeros(MAX_SLOTS, dtype=np.int64)

    for slot_idx, poly in enumerate(valid_polyhedra):
        offset = slot_idx * ATOMS_PER_SLOT
        center = poly["center"]
        neighbors = poly["neighbors"]

        # Center atom
        center_z = Element(center["element"]).Z
        center_coords = np.array(center["coords"], dtype=np.float32)

        atom_types[offset] = center_z
        gt_types[offset] = center_z
        cart_coords[offset] = np.zeros(3)  # center at origin
        real_mask[offset] = 1
        center_mask[offset] = 1

        # Neighbor atoms (relative to center)
        for j, nbr in enumerate(neighbors):
            pos = offset + 1 + j
            n_z = Element(nbr["element"]).Z
            n_coords = np.array(nbr["coords"], dtype=np.float32)

            atom_types[pos] = n_z
            gt_types[pos] = n_z
            cart_coords[pos] = n_coords - center_coords
            real_mask[pos] = 1

        # Slot IDs for all 13 positions
        slot_ids[offset : offset + ATOMS_PER_SLOT] = slot_idx
        slot_mask[slot_idx] = 1

    # For empty slots, still assign slot_ids
    for slot_idx in range(n_poly, MAX_SLOTS):
        offset = slot_idx * ATOMS_PER_SLOT
        slot_ids[offset : offset + ATOMS_PER_SLOT] = slot_idx

    # Composition vector
    comp_vec = composition_to_vector(composition)

    return {
        "id": crystal_entry.get("id", ""),
        "composition": composition,
        "atom_types": atom_types,
        "gt_types": gt_types,
        "cart_coords": cart_coords,
        "real_mask": real_mask,
        "center_mask": center_mask,
        "slot_ids": slot_ids,
        "slot_mask": slot_mask,
        "comp_vec": comp_vec,
        "num_atoms": TOTAL_ATOMS,
        "num_real_atoms": int(real_mask.sum()),
        "n_slots": n_poly,
        "spacegroup": 1,
        "formation_energy_per_atom": np.float64(0.0),
        # Dummy lattice for scaler compatibility
        "scaled_lattice": np.zeros(6, dtype=np.float32),
    }


def preprocess_multi_polyhedra(
    input_file,
    num_workers,
    niggli,
    primitive,
    graph_method,
    prop_list,
    use_space_group=False,
    tol=0.01,
):
    """
    ### NEW: Preprocess polyhedra JSON into multi-polyhedron packed samples.

    Each crystal becomes ONE sample with up to 5 polyhedra packed into 65 atoms.
    """
    # Determine which JSON to load
    input_file_json = ""
    split_name = input_file.split("/")[-1].split("_")[0]
    if split_name == "train":
        input_file_json = base_data_dir / "train_small.json"
    elif split_name == "val":
        input_file_json = base_data_dir / "val_small.json"
    else:
        input_file_json = base_data_dir / "test_small.json"

    print(f"Loading multi-polyhedra from: {input_file_json}")

    with open(input_file_json, "r") as f:
        data = json.load(f)

    print(f"Total crystal entries: {len(data)}")

    # Group polyhedra by crystal and deduplicate by label prefix
    crystal_entries = []
    for entry in data:
        crystal_id = entry["id"]
        comp = entry["composition"]

        # Deduplicate polyhedra: keep one per unique label prefix (e.g., "TiO6")
        seen_labels = set()
        unique_polys = []
        for poly in entry.get("polyhedra", []):
            label = poly.get("label", "")
            label_prefix = label.split("_")[0] if label else ""
            if label_prefix and label_prefix not in seen_labels:
                seen_labels.add(label_prefix)
                unique_polys.append(poly)

        if 1 <= len(unique_polys) <= MAX_SLOTS:
            crystal_entries.append(
                {
                    "id": crystal_id,
                    "composition": comp,
                    "polyhedra": unique_polys,
                }
            )

    print(f"Crystals with 1-{MAX_SLOTS} unique polyhedra: {len(crystal_entries)}")

    # Pack each crystal
    results = []
    skipped = 0
    for ce in crystal_entries:
        packed = pack_crystal_polyhedra(ce)
        if packed is not None:
            results.append(packed)
        else:
            skipped += 1

    print(f"Packed samples: {len(results)}, skipped: {skipped}")

    # Distribution of polyhedra counts
    from collections import Counter

    slot_dist = Counter(r["n_slots"] for r in results)
    print("Polyhedra per crystal distribution:")
    for k in sorted(slot_dist.keys()):
        print(
            f"  {k} polyhedra: {slot_dist[k]} crystals ({100*slot_dist[k]/len(results):.1f}%)"
        )

    return results


### END NEW


# === Keep all existing functions below unchanged ===


def abs_cap(val, max_abs_val=1):
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)
    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    val = abs_cap(val)
    gamma_star = np.arccos(val)
    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def lattice_params_to_matrix_torch(lengths, angles):
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)
    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)
    vector_a = torch.stack(
        [
            lengths[:, 0] * sins[:, 1],
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 0] * coses[:, 1],
        ],
        dim=1,
    )
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
            lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack(
        [
            torch.zeros(lengths.size(0), device=lengths.device),
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 2],
        ],
        dim=1,
    )
    return torch.stack([vector_a, vector_b, vector_c], dim=1)


class StandardScalerTorch(object):
    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        self.stds = torch.std(X, dim=0, unbiased=False) + EPSILON

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(), stds=self.stds.clone().detach()
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(means: {self.means.tolist()}, stds: {self.stds.tolist()})"


def get_scaler_from_data_list(data_list, key):
    targets = torch.tensor([d[key] for d in data_list])
    scaler = StandardScalerTorch()
    scaler.fit(targets)
    return scaler


def add_scaled_lattice_prop(data_list, lattice_scale_method):
    for d in data_list:
        if "graph_arrays" in d:
            (
                frac_coords,
                atom_types,
                lengths,
                angles,
                edge_indices,
                to_jimages,
                num_atoms,
            ) = d["graph_arrays"]
            d["scaled_lattice"] = np.concatenate([lengths, angles])
        elif "scaled_lattice" not in d:
            d["scaled_lattice"] = np.zeros(6, dtype=np.float32)


def add_scaled_lattice_prop_cart_coord_polyhedra(data_list, lattice_scale_method):
    for d in data_list:
        if "graph_arrays" in d:
            (
                frac_coords,
                cart_coords,
                central_atoms,
                atom_types,
                lengths,
                angles,
                edge_indices,
                to_jimages,
                num_atoms,
            ) = d["graph_arrays"]
            d["scaled_lattice"] = np.concatenate([lengths, angles])
        elif "scaled_lattice" not in d:
            d["scaled_lattice"] = np.zeros(6, dtype=np.float32)


def process_one(
    row, niggli, primitive, graph_method, prop_list, use_space_group=False, tol=0.01
):
    crystal_str = row["cif"]
    crystal = build_crystal(crystal_str, niggli=niggli, primitive=primitive)
    result_dict = {}
    if use_space_group:
        from pyxtal import pyxtal

        crystal, sym_info = get_symmetry_info(crystal, tol=tol)
        result_dict.update(sym_info)
    else:
        result_dict["spacegroup"] = 1
    graph_arrays = build_crystal_graph(crystal, graph_method)
    properties = {k: row[k] for k in prop_list if k in row.keys()}
    result_dict.update(
        {"id": row["material_id"], "cif": crystal_str, "graph_arrays": graph_arrays}
    )
    result_dict.update(properties)
    return result_dict


def process_one_polyhedron_test(
    row, niggli, primitive, graph_method, prop_list, use_space_group=False, tol=0.01
):
    crystal_str = ""
    crystal = None
    result_dict = {}
    result_dict["spacegroup"] = 1
    graph_arrays = build_crystal_graph_polyhedron_test(row, graph_method)
    if graph_arrays is None:
        return None
    properties = {"formation_energy_per_atom": np.float64(0.0)}
    result_dict.update(
        {"id": row["id"], "cif": crystal_str, "graph_arrays": graph_arrays}
    )
    result_dict.update(properties)
    return result_dict


def preprocess(
    input_file,
    num_workers,
    niggli,
    primitive,
    graph_method,
    prop_list,
    use_space_group=False,
    tol=0.01,
):
    df = pd.read_csv(input_file)
    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],
        [niggli] * len(df),
        [primitive] * len(df),
        [graph_method] * len(df),
        [prop_list] * len(df),
        [use_space_group] * len(df),
        [tol] * len(df),
        num_cpus=num_workers,
    )
    mpid_to_results = {result["id"]: result for result in unordered_results}
    ordered_results = [
        mpid_to_results[df.iloc[idx]["material_id"]] for idx in range(len(df))
    ]
    return ordered_results


def preprocess_polyhedra(
    input_file,
    num_workers,
    niggli,
    primitive,
    graph_method,
    prop_list,
    use_space_group=False,
    tol=0.01,
):
    input_file_json = ""
    if input_file.split("/")[-1].split("_")[0] == "train":
        input_file_json = base_data_dir / "train_one_polyhedron.json"
    elif input_file.split("/")[-1].split("_")[0] == "val":
        input_file_json = base_data_dir / "val_one_polyhedron.json"
    else:
        input_file_json = base_data_dir / "test_one_polyhedron.json"

    with open(input_file_json, "r") as f:
        data = json.load(f)

    results = []
    for entry in data:
        id = entry["id"]
        comp = entry["composition"]
        for poly in entry["polyhedra"]:
            results.append(
                {
                    "id": id,
                    "composition": comp,
                    "group_name": poly["label"].split("_")[0],
                    "motif_type": poly["motif_type"],
                    "motif_symbol": poly.get("motif_symbol"),
                    "center": poly["center"],
                    "neighbors": poly["neighbors"],
                    "permutation": poly.get("permutation"),
                }
            )

    print(f"Total entries in original data: {len(data)}")
    print(f"Total polyhedron samples kept: {len(results)}")

    unordered_results_test = p_umap(
        process_one_polyhedron_test,
        [results[idx] for idx in range(len(results))],
        [niggli] * len(results),
        [primitive] * len(results),
        [graph_method] * len(results),
        [prop_list] * len(results),
        [use_space_group] * len(results),
        [tol] * len(results),
        num_cpus=num_workers,
    )
    unordered_results_test = [r for r in unordered_results_test if r is not None]
    ordered_results_test = unordered_results_test
    print(f"len(ordered_results_test): {len(ordered_results_test)}")

    if len(ordered_results_test) > 0:
        print("Result types of preprocess_polyhedra test:")
        for tp in ordered_results_test[0].keys():
            print(f"{tp}: {type(ordered_results_test[0][tp])}")

    return ordered_results_test
