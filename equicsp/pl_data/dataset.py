import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
import os
from torch_geometric.data import Data
import pickle
import numpy as np
from torch_geometric.utils import dense_to_sparse

from equicsp.common.utils import PROJECT_ROOT
from equicsp.common.data_utils import (
    preprocess,
    preprocess_polyhedra,
    add_scaled_lattice_prop,
    add_scaled_lattice_prop_cart_coord_polyhedra,
    ### NEW imports for multi-poly
    preprocess_multi_polyhedra,
    MAX_SLOTS,
    ATOMS_PER_SLOT,
    TOTAL_ATOMS,
    ### END NEW
)


class PreTrainCrystDataset(Dataset):
    def __init__(self, name: ValueNode, filepath: ValueNode, listpath: ValueNode):
        super().__init__()
        self.name = name
        self.filepath = filepath
        self.listpath = listpath
        self.load_data()

    def load_data(self):
        with open(self.listpath, "r") as f:
            lines = f.readlines()
        self.datanames = [_.strip() for _ in lines]
        with open(self.filepath, "rb") as f:
            self.datas = pickle.load(f)

    def __len__(self) -> int:
        return len(self.datanames)

    def __getitem__(self, index):
        idx = self.datanames[index]
        structure = self.datas[idx]
        lattice = structure.lattice
        data = Data(
            frac_coords=torch.Tensor(structure.frac_coords),
            atom_types=torch.LongTensor([_.Z for _ in structure.species]),
            lengths=torch.Tensor(lattice.abc).view(1, -1),
            angles=torch.Tensor(lattice.angles).view(1, -1),
            num_atoms=len(structure),
            num_nodes=len(structure),
        )
        return data


class CrystDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        prop: ValueNode,
        niggli: ValueNode,
        primitive: ValueNode,
        graph_method: ValueNode,
        preprocess_workers: ValueNode,
        lattice_scale_method: ValueNode,
        save_path: ValueNode,
        tolerance: ValueNode,
        use_space_group: ValueNode,
        use_pos_index: ValueNode,
        reprocess: ValueNode,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance
        self.reprocess = reprocess

        self.preprocess(save_path, preprocess_workers, prop)
        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def preprocess(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path) and not self.reprocess:
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess(
                self.path,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_list=[prop],
                use_space_group=self.use_space_group,
                tol=self.tolerance,
            )
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        prop = self.scaler.transform(data_dict[self.prop])
        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,
            to_jimages,
            num_atoms,
        ) = data_dict["graph_arrays"]

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
            y=prop.view(1, -1),
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])
            data.ops = torch.Tensor(data_dict["wyckoff_ops"])
            data.anchor_index = torch.LongTensor(data_dict["anchors"])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class PolyhedraDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        prop: ValueNode,
        niggli: ValueNode,
        primitive: ValueNode,
        graph_method: ValueNode,
        preprocess_workers: ValueNode,
        lattice_scale_method: ValueNode,
        save_path: ValueNode,
        tolerance: ValueNode,
        use_space_group: ValueNode,
        use_pos_index: ValueNode,
        reprocess: ValueNode,
        coord_norm_factor: ValueNode = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance
        self.reprocess = reprocess
        self.coord_norm_factor = coord_norm_factor

        self.preprocess_polyhedra(save_path, preprocess_workers, prop)
        add_scaled_lattice_prop_cart_coord_polyhedra(
            self.cached_data, lattice_scale_method
        )
        self.lattice_scaler = None
        self.scaler = None

    def preprocess_polyhedra(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path) and not self.reprocess:
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess_polyhedra(
                self.path,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_list=[prop],
                use_space_group=self.use_space_group,
                tol=self.tolerance,
            )
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        prop = self.scaler.transform(data_dict[self.prop])
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
        ) = data_dict["graph_arrays"]

        cart_coords = cart_coords / self.coord_norm_factor

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            cart_coords=torch.Tensor(cart_coords),
            central_atoms=torch.LongTensor(central_atoms),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
            y=prop.view(1, -1),
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])
            data.ops = torch.Tensor(data_dict["wyckoff_ops"])
            data.anchor_index = torch.LongTensor(data_dict["anchors"])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"PolyhedraDataset({self.name=}, {self.path=})"


### NEW: Multi-polyhedron dataset — one sample per crystal, all polyhedra packed
class MultiPolyDataset(Dataset):
    """
    Each sample = one crystal with ALL unique polyhedra packed into fixed-size tensors.
    5 slots × 13 atoms = 65 atoms per sample.
    
    Fields per Data object:
        atom_types:     (65,)   atomic numbers, 0=PAD
        gt_types:       (65,)   ground truth types (same, used for type loss)
        cart_coords:    (65,3)  Cartesian coords, zeros for padding
        real_mask:      (65,)   1=real atom, 0=padding
        center_mask:    (65,)   1=center atom of a polyhedron
        slot_ids:       (65,)   which slot (0-4) each atom belongs to
        slot_mask:      (5,)    which slots are active
        comp_vec:       (100,)  crystal composition vector
        num_atoms:      scalar  always 65
        num_real_atoms: scalar  actual number of real atoms
        n_slots:        scalar  number of active polyhedra
    """

    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        prop: ValueNode,
        niggli: ValueNode,
        primitive: ValueNode,
        graph_method: ValueNode,
        preprocess_workers: ValueNode,
        lattice_scale_method: ValueNode,
        save_path: ValueNode,
        tolerance: ValueNode,
        use_space_group: ValueNode,
        use_pos_index: ValueNode,
        reprocess: ValueNode,
        coord_norm_factor: ValueNode = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance
        self.reprocess = reprocess
        self.coord_norm_factor = coord_norm_factor

        self._preprocess(save_path, preprocess_workers, prop)
        self.lattice_scaler = None
        self.scaler = None

    def _preprocess(self, save_path, preprocess_workers, prop):
        # Use a distinct save_path suffix to avoid collision with single-poly cache
        multi_save_path = save_path.replace(".pt", "_multi.pt") if save_path.endswith(".pt") else save_path + "_multi"
        
        if os.path.exists(multi_save_path) and not self.reprocess:
            print(f"Loading cached multi-poly data from {multi_save_path}")
            self.cached_data = torch.load(multi_save_path)
        else:
            print(f"Preprocessing multi-poly data from {self.path}")
            cached_data = preprocess_multi_polyhedra(
                self.path,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_list=[prop],
                use_space_group=self.use_space_group,
                tol=self.tolerance,
            )
            torch.save(cached_data, multi_save_path)
            self.cached_data = cached_data

        print(f"[MultiPolyDataset] {len(self.cached_data)} crystal samples loaded")

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        d = self.cached_data[index]

        # Normalize coordinates by coord_norm_factor (applied at load time)
        cart_coords = d["cart_coords"] / self.coord_norm_factor

        data = Data(
            atom_types=torch.LongTensor(d["atom_types"]),
            gt_types=torch.LongTensor(d["gt_types"]),
            cart_coords=torch.FloatTensor(cart_coords),
            real_mask=torch.LongTensor(d["real_mask"]),
            center_mask=torch.LongTensor(d["center_mask"]),
            slot_ids=torch.LongTensor(d["slot_ids"]),
            slot_mask=torch.LongTensor(d["slot_mask"]),
            comp_vec=torch.FloatTensor(d["comp_vec"]),
            num_atoms=TOTAL_ATOMS,
            num_real_atoms=d["num_real_atoms"],
            n_slots=d["n_slots"],
            num_nodes=TOTAL_ATOMS,  # PyG batching key
            # Dummy fields for scaler compatibility
            y=torch.zeros(1, 1),
        )
        return data

    def __repr__(self) -> str:
        return f"MultiPolyDataset({self.name=}, {self.path=}, samples={len(self.cached_data)})"
### END NEW


class DisCrystDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        prop: ValueNode,
        niggli: ValueNode,
        primitive: ValueNode,
        graph_method: ValueNode,
        preprocess_workers: ValueNode,
        lattice_scale_method: ValueNode,
        save_path: ValueNode,
        tolerance: ValueNode,
        use_space_group: ValueNode,
        use_pos_index: ValueNode,
        rank=0,
        num_dis=1,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.rank = rank
        self.num_dis = num_dis
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance

        self.preprocess(save_path, preprocess_workers, prop)
        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)

    def preprocess(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess(
                self.path,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_list=[prop],
                use_space_group=self.use_space_group,
                tol=self.tolerance,
            )
            if self.rank == 0:
                torch.save(cached_data, save_path)
                self.cached_data = cached_data

        num_rows = len(self.cached_data)
        subset_size = num_rows // self.num_dis
        start_idx = self.rank * subset_size
        if self.rank == self.num_dis - 1:
            end_idx = num_rows
        else:
            end_idx = start_idx + subset_size
        self.cached_data = self.cached_data[start_idx:end_idx]

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,
            to_jimages,
            num_atoms,
        ) = data_dict["graph_arrays"]

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])
            data.ops = torch.Tensor(data_dict["wyckoff_ops"])
            data.anchor_index = torch.LongTensor(data_dict["anchors"])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"DisCrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(
        self,
        crystal_array_list,
        niggli,
        primitive,
        graph_method,
        preprocess_workers,
        lattice_scale_method,
        **kwargs,
    ):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
        )

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,
            to_jimages,
            num_atoms,
        ) = data_dict["graph_arrays"]

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from equicsp.common.data_utils import get_scaler_from_data_list

    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data, key="scaled_lattice"
    )
    scaler = get_scaler_from_data_list(dataset.cached_data, key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
