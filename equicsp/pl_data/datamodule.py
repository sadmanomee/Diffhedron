import random
from typing import Optional, Sequence
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomRotate

from equicsp.common.utils import PROJECT_ROOT
from equicsp.common.data_utils import get_scaler_from_data_list


def worker_init_fn(id: int):
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class PreTrainCrystDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


class CrystDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        scaler_path=None,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        pass

    def get_scaler(self, scaler_path):
        if scaler_path is None:
            train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.lattice_scaler = get_scaler_from_data_list(
                train_dataset.cached_data, key="scaled_lattice"
            )
            self.scaler = get_scaler_from_data_list(
                train_dataset.cached_data, key=train_dataset.prop
            )
        else:
            self.lattice_scaler = torch.load(Path(scaler_path) / "lattice_scaler.pt")
            self.scaler = torch.load(Path(scaler_path) / "prop_scaler.pt")

    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]
            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
            for val_dataset in self.val_datasets:
                val_dataset.lattice_scaler = self.lattice_scaler
                val_dataset.scaler = self.scaler

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]
            for test_dataset in self.test_datasets:
                test_dataset.lattice_scaler = self.lattice_scaler
                test_dataset.scaler = self.scaler

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


class PolyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        scaler_path=None,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        pass

    def get_scaler(self, scaler_path):
        if scaler_path is None:
            train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.lattice_scaler = get_scaler_from_data_list(
                train_dataset.cached_data, key="scaled_lattice"
            )
            self.scaler = get_scaler_from_data_list(
                train_dataset.cached_data, key=train_dataset.prop
            )
            all_coords = []
            print(f"len(train_dataset.cached_data) = {len(train_dataset.cached_data)}")
            print(train_dataset.cached_data[0].keys())
            for d in train_dataset.cached_data:
                cart_c = d["graph_arrays"][1]  # cart_coords
                all_coords.append(cart_c.reshape(-1))
            all_coords = np.concatenate(all_coords)
            self.coord_norm_factor = float(np.std(all_coords))
            if self.coord_norm_factor < 1e-8:
                print(
                    "Warning: coord_norm_factor is very small, setting to 1.0 to avoid instability."
                )
                self.coord_norm_factor = 1.0
            print(f"[PolyDataModule] coord_norm_factor = {self.coord_norm_factor:.6f}")
        else:
            print(f"Loading scalers from {scaler_path}")
            self.lattice_scaler = torch.load(Path(scaler_path) / "lattice_scaler.pt")
            self.scaler = torch.load(Path(scaler_path) / "prop_scaler.pt")
            self.coord_norm_factor = float(
                torch.load(Path(scaler_path) / "coord_norm_factor.pt")
            )

    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.train_dataset.transform = RandomRotate(degrees=180, axis=-1)
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]
            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
            self.train_dataset.coord_norm_factor = self.coord_norm_factor
            for val_dataset in self.val_datasets:
                val_dataset.lattice_scaler = self.lattice_scaler
                val_dataset.scaler = self.scaler
                val_dataset.coord_norm_factor = self.coord_norm_factor

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]
            for test_dataset in self.test_datasets:
                test_dataset.lattice_scaler = self.lattice_scaler
                test_dataset.scaler = self.scaler
                test_dataset.coord_norm_factor = self.coord_norm_factor

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


### NEW: Multi-polyhedron DataModule
class MultiPolyDataModule(pl.LightningDataModule):
    """
    DataModule for multi-polyhedron crystal generation.
    
    Computes coord_norm_factor from ALL real atoms across ALL training crystals.
    Each sample = one crystal with all its unique polyhedra packed into 65 atoms.
    """

    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        scaler_path=None,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        pass

    def get_scaler(self, scaler_path):
        if scaler_path is None:
            # Instantiate train dataset to compute scalers
            train_dataset = hydra.utils.instantiate(self.datasets.train)
            
            # Dummy lattice/prop scalers for compatibility
            from equicsp.common.data_utils import StandardScalerTorch
            self.lattice_scaler = StandardScalerTorch()
            self.lattice_scaler.means = torch.zeros(6)
            self.lattice_scaler.stds = torch.ones(6)
            self.scaler = StandardScalerTorch()
            self.scaler.means = torch.zeros(1)
            self.scaler.stds = torch.ones(1)

            # Compute coord_norm_factor from REAL atoms only
            all_coords = []
            for d in train_dataset.cached_data:
                cart_c = d["cart_coords"]
                real_m = d["real_mask"]
                # Only include real atom coordinates
                real_coords = cart_c[real_m == 1]
                if len(real_coords) > 0:
                    all_coords.append(real_coords.reshape(-1))
            all_coords = np.concatenate(all_coords)
            self.coord_norm_factor = float(np.std(all_coords))
            if self.coord_norm_factor < 1e-8:
                print("Warning: coord_norm_factor is very small, setting to 1.0")
                self.coord_norm_factor = 1.0
            print(f"[MultiPolyDataModule] coord_norm_factor = {self.coord_norm_factor:.6f}")
            print(f"[MultiPolyDataModule] computed from {len(all_coords)} coordinate values")
        else:
            print(f"Loading scalers from {scaler_path}")
            self.lattice_scaler = torch.load(Path(scaler_path) / "lattice_scaler.pt")
            self.scaler = torch.load(Path(scaler_path) / "prop_scaler.pt")
            self.coord_norm_factor = float(
                torch.load(Path(scaler_path) / "coord_norm_factor.pt")
            )

    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]
            # Pass coord_norm_factor to all datasets
            self.train_dataset.coord_norm_factor = self.coord_norm_factor
            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
            for val_dataset in self.val_datasets:
                val_dataset.coord_norm_factor = self.coord_norm_factor
                val_dataset.lattice_scaler = self.lattice_scaler
                val_dataset.scaler = self.scaler

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]
            for test_dataset in self.test_datasets:
                test_dataset.coord_norm_factor = self.coord_norm_factor
                test_dataset.lattice_scaler = self.lattice_scaler
                test_dataset.scaler = self.scaler

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )
### END NEW


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup("fit")
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()
