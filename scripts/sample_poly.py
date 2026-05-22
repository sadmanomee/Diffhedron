# import time
# import argparse
# import torch

# from pathlib import Path
# from torch_geometric.data import Data, Batch, DataLoader
# from torch.utils.data import Dataset
# from eval_utils import load_model, lattices_to_params_shape, get_crystals_list

# from pymatgen.core.structure import Structure
# from pymatgen.core.lattice import Lattice
# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# from pymatgen.io.cif import CifWriter
# from pyxtal.symmetry import Group
# import chemparse
# import numpy as np
# from p_tqdm import p_map

# import os

# # chemical_symbols = [
# #     # 0
# #     "X",
# #     # 1
# #     "H",
# #     "He",
# #     # 2
# #     "Li",
# #     "Be",
# #     "B",
# #     "C",
# #     "N",
# #     "O",
# #     "F",
# #     "Ne",
# #     # 3
# #     "Na",
# #     "Mg",
# #     "Al",
# #     "Si",
# #     "P",
# #     "S",
# #     "Cl",
# #     "Ar",
# #     # 4
# #     "K",
# #     "Ca",
# #     "Sc",
# #     "Ti",
# #     "V",
# #     "Cr",
# #     "Mn",
# #     "Fe",
# #     "Co",
# #     "Ni",
# #     "Cu",
# #     "Zn",
# #     "Ga",
# #     "Ge",
# #     "As",
# #     "Se",
# #     "Br",
# #     "Kr",
# #     # 5
# #     "Rb",
# #     "Sr",
# #     "Y",
# #     "Zr",
# #     "Nb",
# #     "Mo",
# #     "Tc",
# #     "Ru",
# #     "Rh",
# #     "Pd",
# #     "Ag",
# #     "Cd",
# #     "In",
# #     "Sn",
# #     "Sb",
# #     "Te",
# #     "I",
# #     "Xe",
# #     # 6
# #     "Cs",
# #     "Ba",
# #     "La",
# #     "Ce",
# #     "Pr",
# #     "Nd",
# #     "Pm",
# #     "Sm",
# #     "Eu",
# #     "Gd",
# #     "Tb",
# #     "Dy",
# #     "Ho",
# #     "Er",
# #     "Tm",
# #     "Yb",
# #     "Lu",
# #     "Hf",
# #     "Ta",
# #     "W",
# #     "Re",
# #     "Os",
# #     "Ir",
# #     "Pt",
# #     "Au",
# #     "Hg",
# #     "Tl",
# #     "Pb",
# #     "Bi",
# #     "Po",
# #     "At",
# #     "Rn",
# #     # 7
# #     "Fr",
# #     "Ra",
# #     "Ac",
# #     "Th",
# #     "Pa",
# #     "U",
# #     "Np",
# #     "Pu",
# #     "Am",
# #     "Cm",
# #     "Bk",
# #     "Cf",
# #     "Es",
# #     "Fm",
# #     "Md",
# #     "No",
# #     "Lr",
# #     "Rf",
# #     "Db",
# #     "Sg",
# #     "Bh",
# #     "Hs",
# #     "Mt",
# #     "Ds",
# #     "Rg",
# #     "Cn",
# #     "Nh",
# #     "Fl",
# #     "Mc",
# #     "Lv",
# #     "Ts",
# #     "Og",
# # ]

# chemical_symbols = [
#     "X",
#     "H",
#     "He",
#     "Li",
#     "Be",
#     "B",
#     "C",
#     "N",
#     "O",
#     "F",
#     "Ne",
#     "Na",
#     "Mg",
#     "Al",
#     "Si",
#     "P",
#     "S",
#     "Cl",
#     "Ar",
#     "K",
#     "Ca",
#     "Sc",
#     "Ti",
#     "V",
#     "Cr",
#     "Mn",
#     "Fe",
#     "Co",
#     "Ni",
#     "Cu",
#     "Zn",
#     "Ga",
#     "Ge",
#     "As",
#     "Se",
#     "Br",
#     "Kr",
#     "Rb",
#     "Sr",
#     "Y",
#     "Zr",
#     "Nb",
#     "Mo",
#     "Tc",
#     "Ru",
#     "Rh",
#     "Pd",
#     "Ag",
#     "Cd",
#     "In",
#     "Sn",
#     "Sb",
#     "Te",
#     "I",
#     "Xe",
#     "Cs",
#     "Ba",
#     "La",
#     "Ce",
#     "Pr",
#     "Nd",
#     "Pm",
#     "Sm",
#     "Eu",
#     "Gd",
#     "Tb",
#     "Dy",
#     "Ho",
#     "Er",
#     "Tm",
#     "Yb",
#     "Lu",
#     "Hf",
#     "Ta",
#     "W",
#     "Re",
#     "Os",
#     "Ir",
#     "Pt",
#     "Au",
#     "Hg",
#     "Tl",
#     "Pb",
#     "Bi",
#     "Po",
#     "At",
#     "Rn",
#     "Fr",
#     "Ra",
#     "Ac",
#     "Th",
#     "Pa",
#     "U",
#     "Np",
#     "Pu",
#     "Am",
#     "Cm",
#     "Bk",
#     "Cf",
#     "Es",
#     "Fm",
# ]  # 101 entries: index 0="X", indices 1-100 = valid atomic numbers


# def diffusion(loader, model, step_lr):

#     cart_coords = []
#     num_atoms = []
#     atom_types = []
#     for idx, batch in enumerate(loader):

#         if torch.cuda.is_available():
#             batch.cuda()
#         outputs, traj = model.sample(batch, step_lr=step_lr)
#         cart_coords.append(outputs["cart_coords"].detach().cpu())
#         num_atoms.append(outputs["num_atoms"].detach().cpu())
#         atom_types.append(outputs["atom_types"].detach().cpu())

#     cart_coords = torch.cat(cart_coords, dim=0)
#     num_atoms = torch.cat(num_atoms, dim=0)
#     atom_types = torch.cat(atom_types, dim=0)

#     return (cart_coords, atom_types, num_atoms)


# class SampleDataset(Dataset):

#     def __init__(self, formula, num_evals):
#         super().__init__()
#         self.formula = formula
#         self.num_evals = num_evals
#         self.get_structure()

#     def get_structure(self):
#         self.composition = chemparse.parse_formula(self.formula)
#         chem_list = []
#         for elem in self.composition:
#             num_int = int(self.composition[elem])
#             idx = chemical_symbols.index(elem)
#             if idx < 1 or idx > 100:
#                 raise ValueError(
#                     f"Element {elem} (Z={idx}) outside supported range 1-100"
#                 )
#             chem_list.extend([idx] * num_int)
#         self.chem_list = chem_list

#     def __len__(self) -> int:
#         return self.num_evals

#     def __getitem__(self, index):
#         return Data(
#             atom_types=torch.LongTensor(self.chem_list),
#             num_atoms=len(self.chem_list),
#             num_nodes=len(self.chem_list),
#         )


# def get_pymatgen(crystal_array):
#     frac_coords = crystal_array["frac_coords"]
#     atom_types = crystal_array["atom_types"]
#     lengths = crystal_array["lengths"]
#     angles = crystal_array["angles"]
#     try:
#         structure = Structure(
#             lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
#             species=atom_types,
#             coords=frac_coords,
#             coords_are_cartesian=False,
#         )
#         return structure
#     except:
#         return None


# def main(args):
#     # load_data if do reconstruction.
#     model_path = Path(args.model_path)
#     print(f"Model path: {model_path}")
#     model, _, cfg = load_model(model_path, load_data=False)

#     if torch.cuda.is_available():
#         model.to("cuda")

#     ### modification
#     norm_factor_path = model_path / "coord_norm_factor.pt"
#     if norm_factor_path.exists():
#         model.coord_norm_factor = torch.load(norm_factor_path)
#         print(f"Loaded coord_norm_factor: {model.coord_norm_factor}")
#     else:
#         print("WARNING: coord_norm_factor.pt not found, using default 1.0")
#     ### modification

#     if torch.cuda.is_available():
#         model.to("cuda")

#     ### modification
#     model.eval()
#     ### modification

#     # tar_dir = os.path.join(args.save_path, args.formula)
#     # os.makedirs(tar_dir, exist_ok=True)

#     print("Evaluate the diffusion model.")

#     test_set = SampleDataset(args.formula, args.num_evals)
#     test_loader = DataLoader(test_set, batch_size=min(args.batch_size, args.num_evals))

#     start_time = time.time()
#     (cart_coords, atom_types, num_atoms) = diffusion(test_loader, model, args.step_lr)

#     # crystal_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)
#     print(cart_coords.shape, atom_types.shape, num_atoms.shape)
#     print(cart_coords)

#     # strcuture_list = p_map(get_pymatgen, crystal_list)

#     # for i,structure in enumerate(strcuture_list):
#     #     tar_file = os.path.join(tar_dir, f"{args.formula}_{i+1}.cif")
#     #     if structure is not None:
#     #         writer = CifWriter(structure)
#     #         writer.write_file(tar_file)
#     #     else:
#     #         print(f"{i+1} Error Structure.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", required=True)
#     # parser.add_argument("--save_path", required=True)
#     parser.add_argument("--formula", required=True)
#     parser.add_argument("--num_evals", default=1, type=int)
#     parser.add_argument("--batch_size", default=500, type=int)
#     parser.add_argument("--step_lr", default=5e-6, type=float)

#     args = parser.parse_args()

#     main(args)


import time
import argparse
import torch

from pathlib import Path
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset
from eval_utils import load_model, lattices_to_params_shape, get_crystals_list

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pyxtal.symmetry import Group
import chemparse
import numpy as np
from p_tqdm import p_map

import os

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
]  # 101 entries: index 0="X", indices 1-100 = valid atomic numbers


### changed — now returns list of per-sample coord tensors
def diffusion(loader, model, step_lr):
    all_cart_coords = []
    all_num_atoms = []
    all_atom_types = []
    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample(batch, step_lr=step_lr)
        all_cart_coords.append(outputs["cart_coords"].detach().cpu())
        all_num_atoms.append(outputs["num_atoms"].detach().cpu())
        all_atom_types.append(outputs["atom_types"].detach().cpu())

    cart_coords = torch.cat(all_cart_coords, dim=0)
    num_atoms = torch.cat(all_num_atoms, dim=0)
    atom_types = torch.cat(all_atom_types, dim=0)
    return (cart_coords, atom_types, num_atoms)


class SampleDataset(Dataset):
    def __init__(self, formula, num_evals):
        super().__init__()
        self.formula = formula
        self.num_evals = num_evals
        self.get_structure()

    def get_structure(self):
        self.composition = chemparse.parse_formula(self.formula)
        chem_list = []
        for elem in self.composition:
            num_int = int(self.composition[elem])
            idx = chemical_symbols.index(elem)
            if idx < 1 or idx > 100:
                raise ValueError(
                    f"Element {elem} (Z={idx}) outside supported range 1-100"
                )
            chem_list.extend([chemical_symbols.index(elem)] * num_int)
        self.chem_list = chem_list

    def __len__(self) -> int:
        return self.num_evals

    def __getitem__(self, index):
        return Data(
            atom_types=torch.LongTensor(self.chem_list),
            num_atoms=len(self.chem_list),
            num_nodes=len(self.chem_list),
        )


def main(args):
    model_path = Path(args.model_path)
    print(f"Model path: {model_path}")
    model, _, cfg = load_model(model_path, load_data=False)

    ### load coord_norm_factor
    norm_factor_path = model_path / "coord_norm_factor.pt"
    if norm_factor_path.exists():
        model.coord_norm_factor = torch.load(norm_factor_path)
        print(f"Loaded coord_norm_factor: {model.coord_norm_factor}")
    else:
        print("WARNING: coord_norm_factor.pt not found, using default 1.0")

    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    print("Evaluate the diffusion model.")

    test_set = SampleDataset(args.formula, args.num_evals)
    test_loader = DataLoader(test_set, batch_size=min(args.batch_size, args.num_evals))

    start_time = time.time()
    (cart_coords, atom_types, num_atoms) = diffusion(test_loader, model, args.step_lr)

    ### changed — print each sample separately with markers so parser can split them
    offset = 0
    for i in range(len(num_atoms)):
        n = num_atoms[i].item()
        sample_coords = cart_coords[offset : offset + n]
        print(f"===SAMPLE {i}===")
        print(sample_coords)
        offset += n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--formula", required=True)
    parser.add_argument("--num_evals", default=1, type=int)
    parser.add_argument("--batch_size", default=500, type=int)
    parser.add_argument("--step_lr", default=5e-6, type=float)

    args = parser.parse_args()
    main(args)
