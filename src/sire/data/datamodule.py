from typing import Callable, List

import pytorch_lightning as pl

from monai.data import CacheDataset
from monai.data.dataloader import DataLoader
from monai.transforms import Compose, EnsureChannelFirstd, ScaleIntensityRanged

from src.sire.data.dataloader import ListDataLoader
from src.sire.data.preprocessors import (
    BuildForGEMGCN,
    BuildForLandmarkDetection,
    BuildSIREScales,
    ExtractLandmarksFromCenterline,
    LandmarksMaskCrop,
    LandmarksResized,
    LandmarksToPhysical,
)
from src.sire.data.readers import (
    LoadCenterline,
    LoadContour,
    LoadImage,
    LoadImageFromHDF5,
    LoadMask,
    LoadTrackerCenterline,
)
from src.sire.data.transforms import SamplePairedSIRESegmentation, SampleSIRETracker
from src.sire.utils.yaml import load_config


class DataModuleBase(pl.LightningDataModule):
    def __init__(
        self,
        split_path: str,
        num_workers: int = 0,
        pre_transforms: Callable = None,
        transforms: Callable = None,
        **kwargs,
    ):
        super().__init__()
        self.split_path = split_path

        self.num_workers = num_workers
        self.pre_transforms = pre_transforms
        self.transforms = transforms

    def _load_data_split(self):
        split = load_config(self.split_path)

        train_dict = [{"id": sample} for sample in split["train"]]
        val_dict = [{"id": sample} for sample in split["valid"]]
        test_dict = [{"id": sample} for sample in split["test"]]

        return train_dict, val_dict, test_dict

    def setup(self, stage: str):
        train_dict, val_dict, test_dict = self._load_data_split()

        if stage == "test":
            self.test_ds = CacheDataset(
                data=test_dict,
                cache_rate=1.0,
                transform=self.pre_transforms,
                num_workers=self.num_workers,
            )

        else:
            self.transforms = Compose(
                [
                    *self.pre_transforms.transforms,
                    *self.transforms.transforms,
                ]
            )

            self.train_ds = CacheDataset(
                data=train_dict,
                cache_rate=1.0,
                transform=self.transforms,
                num_workers=self.num_workers,
            )
            self.val_ds = CacheDataset(
                data=val_dict,
                cache_rate=1.0,
                transform=self.transforms,
                num_workers=self.num_workers,
            )

    def train_dataloader(self):
        train_loader = ListDataLoader(
            self.train_ds,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return train_loader

    def val_dataloader(self):
        val_loader = ListDataLoader(
            self.val_ds,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return val_loader


class AAASegmentationDataModule(DataModuleBase):
    def __init__(
        self,
        image_path: str,
        contour_path: str,
        split_path: str,
        npoints: int,
        subdivisions: int,
        sire_scales: List[float],
        contour_types: List[str],
        branch_types: List[str],
        center_noise: int = 1,
        bs: int = 1,
        num_workers: int = 0,
        **kwargs,
    ):
        pre_transforms = Compose(
            [
                # Loading data
                LoadContour(contour_path, contour_types=contour_types, branch_types=branch_types),
                LoadImageFromHDF5(image_path),
                # Preprocess
                ScaleIntensityRanged(keys=["image"], a_min=-400, a_max=800, b_min=0, b_max=1, clip=False),
                BuildSIREScales(scales=sire_scales, npoints=npoints, subdivisions=subdivisions),
                BuildForGEMGCN(keys=["scales"]),
            ]
        )

        transforms = Compose(
            [
                SamplePairedSIRESegmentation(
                    npoints=npoints,
                    num_samples=bs,
                    contour_types=contour_types,
                    center_noise=center_noise,
                    stratify_radius=False,
                )
            ]
        )

        super().__init__(split_path, num_workers, pre_transforms, transforms)


class AAATrackingDataModule(DataModuleBase):
    def __init__(
        self,
        image_path: str,
        centerline_path: str,
        split_path: str,
        npoints: int,
        subdivisions: int,
        sire_scales: List[float],
        center_noise: int = 1,
        bs: int = 1,
        num_workers: int = 0,
        **kwargs,
    ):
        pre_transforms = Compose(
            [
                # Loading data
                LoadTrackerCenterline(centerline_path),
                LoadImageFromHDF5(image_path),
                # Preprocess
                ScaleIntensityRanged(keys=["image"], a_min=-400, a_max=800, b_min=0, b_max=1, clip=False),
                BuildSIREScales(scales=sire_scales, npoints=npoints, subdivisions=subdivisions),
                BuildForGEMGCN(keys=["scales"]),
            ]
        )

        transforms = Compose(
            [SampleSIRETracker(npoints=npoints, num_samples=bs, center_noise=center_noise, stratify_radius=False)]
        )

        super().__init__(split_path, num_workers, pre_transforms, transforms)
