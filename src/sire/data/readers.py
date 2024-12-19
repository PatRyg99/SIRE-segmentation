import glob
import os
import re

from typing import Any, Dict, List

import h5py
import numpy as np
import SimpleITK as sitk
import torch

from monai.transforms import Transform
from scipy.interpolate import interp1d

from src.sire.utils.affine import get_affine


class LoadImageFromHDF5(Transform):
    def __init__(self, hdf_file: str):
        self.hf = h5py.File(hdf_file, "r")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["image"] = torch.tensor(self.hf[data["id"]]["img"][()])
        data["image_meta_dict"] = {
            "affine": torch.from_numpy(self.hf[data["id"]]["affine"][()]),
            "spacing": torch.from_numpy(self.hf[data["id"]]["spacing"][()]),
        }

        return data


class LoadImage(Transform):
    def __init__(self, data_root: str):
        self.data_root = data_root

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        itk_image = sitk.ReadImage(glob.glob(os.path.join(self.data_root, data["id"], "*.mhd"))[0])
        affine, spacing = get_affine(itk_image)
        image = sitk.GetArrayFromImage(itk_image)

        data["image"] = torch.from_numpy(image)
        data["image_meta_dict"] = {"affine": affine, "spacing": spacing}

        return data


class LoadMask(Transform):
    def __init__(self, data_root: str):
        self.data_root = data_root

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        itk_image = sitk.ReadImage(glob.glob(os.path.join(self.data_root, f"*{data['id']}*.nii.gz"))[0])
        image = sitk.GetArrayFromImage(itk_image)
        data["mask"] = torch.from_numpy(image)

        return data


class LoadCenterline(Transform):
    def __init__(
        self,
        data_root: str,
        sub_dir: str = "centerlines",
        centerline_segments: List[str] = ["abdominal_aorta", "iliac_left", "iliac_right", "renal_left", "renal_right"],
    ):
        self.data_root = data_root
        self.sub_dir = sub_dir
        self.centerline_segments = centerline_segments

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = data["id"]
        centerline_dict = {
            centerline_segment: {"points": [], "author": []} for centerline_segment in self.centerline_segments
        }

        for centerline_segment in self.centerline_segments:
            filenames = glob.glob(
                os.path.join(self.data_root, sample_id, self.sub_dir, f"centerline_{centerline_segment}*.npy")
            )

            for filename in filenames:
                regex = re.search(f"centerline_{centerline_segment}_(.*).npy", filename)

                if regex is not None:
                    author = regex.group(1)
                else:
                    author = "Unknown"

                points = np.load(os.path.join(self.data_root, sample_id, self.sub_dir, filename))

                centerline_dict[centerline_segment]["author"].append(author)
                centerline_dict[centerline_segment]["points"].append(points)

        data["centerline"] = centerline_dict

        return data


class LoadTrackerCenterline(Transform):
    def __init__(self, data_root: str, r_fraction: float = 0.25):
        self.data_root = data_root
        self.r_fraction = r_fraction

    def _interpolate(self, points: np.array):
        dists = np.concatenate([np.array([0]), np.cumsum(np.linalg.norm(np.diff(points[:, :3], axis=0), axis=1))])

        min_bound = 0 + 0.25 * np.max(points[:, -1])
        max_bound = np.max(dists) - 0.25 * np.max(points[:, -1])

        return interp1d(dists, points.T), min_bound, max_bound

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        control_points = np.load(glob.glob(os.path.join(self.data_root, f"{data['id']}*.npy"))[0])
        centerline_function, min_bound, max_bound = self._interpolate(control_points)

        data["centerline"] = {
            "control_points": control_points,
            "max_bound": max_bound,
            "min_bound": min_bound,
            "interpolation": centerline_function,
        }

        return data


class LoadContour(Transform):
    def __init__(self, data_root: str, contour_types: List[str], branch_types: List[str], pad_contours: bool = True):
        self.data_root = data_root
        self.contour_types = contour_types
        self.branch_types = branch_types
        self.pad_contours = pad_contours

    def _read_cso(self, cso_filepath: str, branch_type: str = ""):
        def _read_block(block: str):
            # Path points
            path_points_block = re.search("# NumPathPointLists(?P<block>.*)", block, re.DOTALL).group("block")
            path_points = np.concatenate(
                [
                    np.array(list(map(str.strip, line.split(" ")[2:-1]))).astype(float).reshape(-1, 3)
                    for line in path_points_block.split("\n")
                    if line.startswith("U32_")
                    if len(line) > 10
                ]
            )
            center = path_points.mean(axis=0)

            normal_line = re.search("# PlaneNormal\nVEC3 (?P<normal>.*)\n", block).group("normal")
            normal = np.array(normal_line.split(",")).astype(float)

            return torch.from_numpy(path_points), torch.from_numpy(center), torch.from_numpy(normal)

        with open(cso_filepath, "r") as file:
            text = file.read()

        out = {"points": [], "normal": [], "center": [], "branch": []}

        for node_block in [string for string in text.split("# Id")][1:]:
            points, center, normal = _read_block(node_block)

            if len(points) > 2:
                out["points"].append(points)
                out["normal"].append(normal)
                out["center"].append(center)
                out["branch"].append(branch_type)

        return out

    def _match_coplanar_contours(self, contours: Dict[str, Any], match_to: str = "lumen", tol: float = 1e-6):
        matched_contours = {match_to: contours[match_to]}

        for name, contour_dict in contours.items():
            if name != match_to:
                matched_idcs = []

                center_targets = torch.stack(contour_dict["center"])
                normal_targets = torch.stack(contour_dict["normal"])

                for i, (center, normal) in enumerate(zip(contours[match_to]["center"], contours[match_to]["normal"])):
                    normal_mask = torch.linalg.norm(torch.cross(normal_targets, normal[None]), axis=1) < tol
                    center_mask = torch.abs(torch.einsum("ij,ij->i", normal[None], center[None] - center_targets)) < tol

                    matched_indices = torch.argwhere(normal_mask & center_mask)

                    if len(matched_indices) != 1:
                        raise ValueError("Multiple or none coplanar contours found!")
                    else:
                        matched_idcs.append(matched_indices.item())

                matched_contours[name] = {key: [value[i] for i in matched_idcs] for key, value in contour_dict.items()}

        return matched_contours

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        contours = {}

        # Load contours
        for contour_type in self.contour_types:
            contour = {"points": [], "normal": [], "center": [], "branch": []}

            for branch_type in self.branch_types:
                cso_files = glob.glob(
                    os.path.join(
                        self.data_root, data["id"], "contours", f"contour_T_*{branch_type}*{contour_type}*.cso"
                    )
                )

                for cso_file in cso_files:
                    contour_dict = self._read_cso(cso_file, branch_type)

                    for key in list(contour):
                        contour[key] += contour_dict[key]

            contours[contour_type] = contour

        # Match coplanar contours
        contours = self._match_coplanar_contours(contours)

        # Pad contours to stack
        if self.pad_contours:
            all_contours = [contour["points"] for contour in contours.values()]
            max_points_len = max([len(points) for contour in all_contours for points in contour])

            for contour_type in self.contour_types:
                for i, points in enumerate(contours[contour_type]["points"]):
                    points_padded = torch.ones((max_points_len, 3))
                    points_padded[: len(points)] = points
                    points_padded[len(points) :] = points[-1].repeat(len(points_padded) - len(points), 1)
                    contours[contour_type]["points"][i] = points_padded

                contours[contour_type] = {
                    k: torch.stack(v).float() if k != "branch" else v for k, v in contours[contour_type].items()
                }

        data["contour"] = contours

        return data
