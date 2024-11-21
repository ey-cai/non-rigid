import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import lightning as L
import numpy as np
import omegaconf
import torch
import torch.utils.data as data
from pytorch3d.transforms import Translate

from non_rigid.utils.augmentation_utils import maybe_apply_augmentations
from non_rigid.utils.pointcloud_utils import downsample_pcd, get_multi_anchor_scene
from non_rigid.utils.transform_utils import random_se3


@dataclass
class RigidDatasetCfg:
    name: str = "rigid"
    data_dir: str = "data/rigid"
    type: str = "train"
    # Misc. kwargs to pass to the dataset e.g. dataset specific parameters
    misc_kwargs: Dict = field(default_factory=dict)

    ###################################################
    # General Dataset Parameters
    ###################################################
    # Number of demos to load
    num_demos: Optional[int] = None
    # Length of the train dataset
    train_dataset_size: int = 256
    # Length of the validation dataset
    val_dataset_size: int = 16
    # Use default values for some parameters on the validation set
    val_use_defaults: bool = False

    ###################################################
    # Point Cloud Transformation Parameters
    ###################################################
    # [action, anchor, none], centers the point clouds w.r.t. the action, anchor, or no centering
    center_type: str = "anchor"
    # Number of points to downsample to
    sample_size_action: int = 1024
    sample_size_anchor: int = 1024
    # Method of downsampling the point cloud
    downsample_type: str = "fps"
    # Scale factor to apply to the point clouds
    pcd_scale_factor: float = 1.0

    # Demonstration transformation parameters
    action_transform_type: str = "quat_uniform"
    action_translation_variance: float = 0.0
    action_rotation_variance: float = 180

    # Demonstration transformation parameters
    anchor_transform_type: str = "quat_uniform"
    anchor_translation_variance: float = 0.0
    anchor_rotation_variance: float = 180

    ###################################################
    # Distractor parameters
    ###################################################
    # Number of distractor anchor point clouds to load
    distractor_anchor_pcds: int = 0
    # Transformation type to apply when generating distractor pcds
    distractor_transform_type: str = "random_flat_upright"
    # Translation and rotation variance for the distractor transformations
    distractor_translation_variance: float = 0.5
    distractor_rotation_variance: float = 180

    ###################################################
    # Data Augmentation Parameters
    ###################################################
    # Probability of applying plane occlusion
    action_plane_occlusion: float = 0.0
    anchor_plane_occlusion: float = 0.0
    # Standoff distance of the occluding plane from selected plane origin
    action_plane_standoff: float = 0.0
    anchor_plane_standoff: float = 0.0
    # Probability of applying ball occlusion
    action_ball_occlusion: float = 0.0
    anchor_ball_occlusion: float = 0.0
    # Radius of the occluding ball
    action_ball_radius: float = 0.0
    anchor_ball_radius: float = 0.0


class RigidPointDataset(data.Dataset):
    def __init__(
        self,
        root: Path,
        type: str = "train",
        dataset_cfg: RigidDatasetCfg = RigidDatasetCfg(),
    ):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg

        self.dataset_dir = self.root / self.type
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        self.demo_files = list(self.dataset_dir.glob("*_teleport_obj_points.npz"))
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir}")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos])

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        # TODO: Remove this hardcoded class selection
        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        # Center the point clouds
        if self.dataset_cfg.center_type == "action_center":
            center = points_action.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_center":
            center = points_anchor.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_random":
            center = points_anchor[np.random.choice(len(points_anchor))]
        elif self.dataset_cfg.center_type == "none":
            center = np.zeros(3)
        else:
            raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
        points_action -= center
        points_anchor -= center

        points_action = torch.as_tensor(points_action).float()
        points_anchor = torch.as_tensor(points_anchor).float()

        # Downsample the point clouds
        points_action, _ = downsample_pcd(
            points_action.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size_action,
            type=self.dataset_cfg.downsample_type,
        )
        points_anchor, _ = downsample_pcd(
            points_anchor.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size_anchor,
            type=self.dataset_cfg.downsample_type,
        )
        points_action = points_action.squeeze(0)
        points_anchor = points_anchor.squeeze(0)

        # Apply scale factor
        points_action *= self.dataset_cfg.pcd_scale_factor
        points_anchor *= self.dataset_cfg.pcd_scale_factor

        # Transform the point clouds
        # ransform the point clouds
        T0 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.action_rotation_variance,
            trans_var=self.dataset_cfg.action_translation_variance,
            rot_sample_method=self.dataset_cfg.action_transform_type,
        )
        T1 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.anchor_rotation_variance,
            trans_var=self.dataset_cfg.anchor_translation_variance,
            rot_sample_method=self.dataset_cfg.anchor_transform_type,
        )

        goal_points_action = T1.transform_points(points_action)
        goal_points_anchor = T1.transform_points(points_anchor)

        # Get starting action point cloud
        # Transform the action point cloud
        goal_points_action_mean = goal_points_action.mean(dim=0)
        points_action = goal_points_action - goal_points_action_mean
        points_action = T0.transform_points(points_action)

        T_action2goal = T0.inverse().compose(
            Translate(goal_points_action_mean.unsqueeze(0))
        )

        return {
            "pc": goal_points_action,  # Action points in goal position
            "pc_anchor": goal_points_anchor,  # Anchor points in goal position
            "pc_action": points_action,  # Action points for context
            "T0": T0.get_matrix().squeeze(0).T,
            "T1": T1.get_matrix().squeeze(0).T,
            "T_action2goal": T_action2goal.get_matrix().squeeze(0).T,
        }


class RigidFlowDataset(data.Dataset):
    def __init__(
        self,
        root: Path,
        type: str = "train",
        dataset_cfg: RigidDatasetCfg = RigidDatasetCfg(),
    ):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg

        self.dataset_dir = self.root / self.type
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        self.demo_files = list(self.dataset_dir.glob("*_teleport_obj_points.npz"))
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir}")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos])

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        # TODO: Remove this hardcoded class selection
        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        # Center the point clouds
        if self.dataset_cfg.center_type == "action_center":
            center = points_action.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_center":
            center = points_anchor.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_random":
            center = points_anchor[np.random.choice(len(points_anchor))]
        elif self.dataset_cfg.center_type == "none":
            center = np.zeros(3)
        else:
            raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
        goal_points_action = points_action - center
        goal_points_anchor = points_anchor - center

        goal_points_action = torch.as_tensor(goal_points_action).float()
        goal_points_anchor = torch.as_tensor(goal_points_anchor).float()

        # Downsample the point clouds
        goal_points_action, _ = downsample_pcd(
            goal_points_action.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size_action,
            type=self.dataset_cfg.downsample_type,
        )
        goal_points_anchor, _ = downsample_pcd(
            goal_points_anchor.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size_anchor,
            type=self.dataset_cfg.downsample_type,
        )
        goal_points_action = goal_points_action.squeeze(0)
        goal_points_anchor = goal_points_anchor.squeeze(0)

        # Apply scale factor
        goal_points_action *= self.dataset_cfg.pcd_scale_factor
        goal_points_anchor *= self.dataset_cfg.pcd_scale_factor

        # Transform the point clouds
        T0 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.action_rotation_variance,
            trans_var=self.dataset_cfg.action_translation_variance,
            rot_sample_method=self.dataset_cfg.action_transform_type,
        )
        T1 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.anchor_rotation_variance,
            trans_var=self.dataset_cfg.anchor_translation_variance,
            rot_sample_method=self.dataset_cfg.anchor_transform_type,
        )

        goal_points_action = T1.transform_points(goal_points_action)
        goal_points_anchor = T1.transform_points(goal_points_anchor)

        # Get starting action point cloud
        # Transform the action point cloud
        points_action = goal_points_action.clone() - goal_points_action.mean(dim=0)
        points_action = T0.transform_points(points_action)

        # Center the action point cloud
        points_action = points_action - points_action.mean(dim=0)

        # Calculate goal flow
        flow = goal_points_action - points_action

        return {
            "pc": points_action,  # Action points in starting position
            "pc_anchor": goal_points_anchor,  # Anchor points in goal position
            "pc_action": goal_points_action,  # Action points in goal position
            "flow": flow,
            "T0": T0.get_matrix().squeeze(0).T,
            "T1": T1.get_matrix().squeeze(0).T,
        }


class NDFPointDataset(data.Dataset):
    def __init__(
        self,
        root: Path,
        type: str = "train",
        dataset_cfg: RigidDatasetCfg = RigidDatasetCfg(),
    ):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg

        dir_type = self.type if self.type == "train" else "test"
        self.dataset_dir = self.root / f"{dir_type}_data/renders"
        print(f"Loading NDF dataset from {self.dataset_dir}")
        self.demo_files = list(self.dataset_dir.glob("*_teleport_obj_points.npz"))
        self.num_demos = len(self.demo_files)
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir}")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos])

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        # TODO: Remove this hardcoded class selection
        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        points_action = torch.as_tensor(points_action).float()
        points_anchor = torch.as_tensor(points_anchor).float()

        # Apply scale factor
        points_action *= self.dataset_cfg.pcd_scale_factor
        points_anchor *= self.dataset_cfg.pcd_scale_factor

        # Add distractor anchor point clouds
        if self.dataset_cfg.distractor_anchor_pcds > 0:
            (
                _,
                points_action,
                points_anchor_base,
                distractor_anchor_pcd_list,
                T_distractor_list,
                debug,
            ) = get_multi_anchor_scene(
                points_gripper=None,
                points_action=points_action.unsqueeze(0),
                points_anchor_base=points_anchor.unsqueeze(0),
                rot_var=self.dataset_cfg.distractor_rotation_variance,
                trans_var=self.dataset_cfg.distractor_translation_variance,
                rot_sample_method=self.dataset_cfg.distractor_transform_type,
                num_anchors_to_add=self.dataset_cfg.distractor_anchor_pcds,
            )
            points_anchor = torch.cat(
                [points_anchor_base] + distractor_anchor_pcd_list, dim=1
            ).squeeze(0)
            points_action = points_action.squeeze(0)

        # Center the point clouds
        if self.dataset_cfg.center_type == "action_center":
            center = points_action.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_center":
            center = points_anchor.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_random":
            center = points_anchor[np.random.choice(len(points_anchor))]
        elif self.dataset_cfg.center_type == "none":
            center = np.zeros(3)
        else:
            raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
        points_action -= center
        points_anchor -= center

        if self.type == "train" or (
            self.type == "val" and not self.dataset_cfg.val_use_defaults
        ):
            # Apply augmentations to the point clouds in their final positions
            points_action = maybe_apply_augmentations(
                points_action,
                min_num_points=self.dataset_cfg.sample_size_action,
                ball_occlusion_param={
                    "ball_occlusion": self.dataset_cfg.action_ball_occlusion,
                    "ball_radius": self.dataset_cfg.action_ball_radius
                    * self.dataset_cfg.pcd_scale_factor,
                },
                plane_occlusion_param={
                    "plane_occlusion": self.dataset_cfg.action_plane_occlusion,
                    "plane_standoff": self.dataset_cfg.action_plane_standoff
                    * self.dataset_cfg.pcd_scale_factor,
                },
            )
            points_anchor = maybe_apply_augmentations(
                points_anchor,
                min_num_points=self.dataset_cfg.sample_size_anchor,
                ball_occlusion_param={
                    "ball_occlusion": self.dataset_cfg.anchor_ball_occlusion,
                    "ball_radius": self.dataset_cfg.anchor_ball_radius
                    * self.dataset_cfg.pcd_scale_factor,
                },
                plane_occlusion_param={
                    "plane_occlusion": self.dataset_cfg.anchor_plane_occlusion,
                    "plane_standoff": self.dataset_cfg.anchor_plane_standoff
                    * self.dataset_cfg.pcd_scale_factor,
                },
            )

        if self.type == "val" and self.dataset_cfg.val_use_defaults:
            # Downsample the point clouds
            points_action, _ = downsample_pcd(
                points_action.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_action,
                type="fps",
            )
            points_anchor, _ = downsample_pcd(
                points_anchor.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_anchor,
                type="fps",
            )
        else:
            # Downsample the point clouds
            points_action, _ = downsample_pcd(
                points_action.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_action,
                type=self.dataset_cfg.downsample_type,
            )
            points_anchor, _ = downsample_pcd(
                points_anchor.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_anchor,
                type=self.dataset_cfg.downsample_type,
            )
        points_action = points_action.squeeze(0)
        points_anchor = points_anchor.squeeze(0)

        # Get transforms for the action and anchor point clouds
        T0 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.action_rotation_variance,
            trans_var=self.dataset_cfg.action_translation_variance,
            rot_sample_method=self.dataset_cfg.action_transform_type,
        )
        T1 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.anchor_rotation_variance,
            trans_var=self.dataset_cfg.anchor_translation_variance,
            rot_sample_method=self.dataset_cfg.anchor_transform_type,
        )

        # Get the goal action and anchor point clouds
        goal_points_action = T1.transform_points(points_action)
        goal_points_anchor = T1.transform_points(points_anchor)

        # Get the 'context' action point cloud
        goal_points_action_mean = goal_points_action.mean(dim=0)
        points_action = goal_points_action - goal_points_action_mean
        points_action = T0.transform_points(points_action)

        # Get transforms for metrics calculation and visualizations
        T_action2goal = T0.inverse().compose(
            Translate(goal_points_action_mean.unsqueeze(0))
        )
        T_aug_action2goal_list = []
        for T_distractor in T_distractor_list:
            T_aug_action2goal = T_action2goal.compose(
                T1.inverse()
                .compose(Translate(center.unsqueeze(0)))
                .compose(T_distractor)
                .compose(Translate(-center.unsqueeze(0)))
                .compose(T1)
            )
            T_aug_action2goal_list.append(T_aug_action2goal)

        data = {
            "pc": goal_points_action,  # Action points in goal position
            "pc_anchor": goal_points_anchor,  # Anchor points in goal position
            "pc_action": points_action,  # Action points for context
            "T0": T0.get_matrix().squeeze(0).T,
            "T1": T1.get_matrix().squeeze(0).T,
            "T_action2goal": T_action2goal.get_matrix().squeeze(0).T,
        }

        # If we have distractor anchor point clouds, add their transforms
        if self.dataset_cfg.distractor_anchor_pcds > 0:
            data["T_distractor_list"] = torch.stack(
                [T.get_matrix().squeeze(0).T for T in T_distractor_list]
            )
            data["T_action2distractor_list"] = torch.stack(
                [T.get_matrix().squeeze(0).T for T in T_aug_action2goal_list]
            )

        return data


class RPDiffPointDataset(data.Dataset):
    def __init__(
        self,
        root: Path,
        type: str = "train",
        dataset_cfg: RigidDatasetCfg = RigidDatasetCfg(),
    ):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg

        dir_type = self.type if self.type == "train" else "test"
        self.dataset_dir = self.root / f"{dir_type}/renders"
        print(f"Loading NDF dataset from {self.dataset_dir}")
        self.demo_files = list(self.dataset_dir.glob("*_teleport_obj_points.npz"))
        self.num_demos = len(self.demo_files)
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir}")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        demo = np.load(self.demo_files[index % self.num_demos], allow_pickle=True)

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        # TODO: Remove this hardcoded class selection
        points_action_raw = points_raw[classes_raw == 0]
        points_anchor_raw = points_raw[classes_raw == 1]

        points_action_raw = torch.as_tensor(points_action_raw).float()
        points_anchor_raw = torch.as_tensor(points_anchor_raw).float()

        # Apply scale factor
        points_action_scaled = points_action_raw * self.dataset_cfg.pcd_scale_factor
        points_anchor_scaled = points_anchor_raw * self.dataset_cfg.pcd_scale_factor

        # Add distractor anchor point clouds
        if self.dataset_cfg.distractor_anchor_pcds > 0:
            (
                _,
                points_action_scaled,
                points_anchor_scaled_base,
                distractor_anchor_pcd_list,
                T_distractor_list,
                debug,
            ) = get_multi_anchor_scene(
                points_gripper=None,
                points_action=points_action_scaled.unsqueeze(0),
                points_anchor_base=points_anchor_scaled.unsqueeze(0),
                rot_var=self.dataset_cfg.distractor_rotation_variance,
                trans_var=self.dataset_cfg.distractor_translation_variance,
                rot_sample_method=self.dataset_cfg.distractor_transform_type,
                num_anchors_to_add=self.dataset_cfg.distractor_anchor_pcds,
            )
            points_anchor_scaled = torch.cat(
                [points_anchor_scaled_base] + distractor_anchor_pcd_list, dim=1
            ).squeeze(0)
            points_action_scaled = points_action_scaled.squeeze(0)

        # Center the point clouds
        if self.dataset_cfg.center_type == "action_center":
            center = points_action_scaled.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_center":
            center = points_anchor_scaled.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_random":
            center = points_anchor_scaled[np.random.choice(len(points_anchor_scaled))]
        elif self.dataset_cfg.center_type == "none":
            center = np.zeros(3)
        else:
            raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
        points_action = points_action_scaled - center
        points_anchor = points_anchor_scaled - center

        if self.type == "train" or (
            self.type == "val" and not self.dataset_cfg.val_use_defaults
        ):
            # Apply augmentations to the point clouds in their final positions
            points_action = maybe_apply_augmentations(
                points_action,
                min_num_points=self.dataset_cfg.sample_size_action,
                ball_occlusion_param={
                    "ball_occlusion": self.dataset_cfg.action_ball_occlusion,
                    "ball_radius": self.dataset_cfg.action_ball_radius
                    * self.dataset_cfg.pcd_scale_factor,
                },
                plane_occlusion_param={
                    "plane_occlusion": self.dataset_cfg.action_plane_occlusion,
                    "plane_standoff": self.dataset_cfg.action_plane_standoff
                    * self.dataset_cfg.pcd_scale_factor,
                },
            )
            points_anchor = maybe_apply_augmentations(
                points_anchor,
                min_num_points=self.dataset_cfg.sample_size_anchor,
                ball_occlusion_param={
                    "ball_occlusion": self.dataset_cfg.anchor_ball_occlusion,
                    "ball_radius": self.dataset_cfg.anchor_ball_radius
                    * self.dataset_cfg.pcd_scale_factor,
                },
                plane_occlusion_param={
                    "plane_occlusion": self.dataset_cfg.anchor_plane_occlusion,
                    "plane_standoff": self.dataset_cfg.anchor_plane_standoff
                    * self.dataset_cfg.pcd_scale_factor,
                },
            )

        if self.type == "val" and self.dataset_cfg.val_use_defaults:
            # Downsample the point clouds
            points_action, _ = downsample_pcd(
                points_action.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_action,
                type="fps",
            )
            points_anchor, _ = downsample_pcd(
                points_anchor.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_anchor,
                type="fps",
            )
        else:
            # Downsample the point clouds
            points_action, _ = downsample_pcd(
                points_action.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_action,
                type=self.dataset_cfg.downsample_type,
            )
            points_anchor, _ = downsample_pcd(
                points_anchor.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_anchor,
                type=self.dataset_cfg.downsample_type,
            )
        points_action = points_action.squeeze(0)
        points_anchor = points_anchor.squeeze(0)

        # Get transforms for the action and anchor point clouds
        T0 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.action_rotation_variance,
            trans_var=self.dataset_cfg.action_translation_variance,
            rot_sample_method=self.dataset_cfg.action_transform_type,
        )
        T1 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.anchor_rotation_variance,
            trans_var=self.dataset_cfg.anchor_translation_variance,
            rot_sample_method=self.dataset_cfg.anchor_transform_type,
        )

        # Get the goal action and anchor point clouds
        goal_points_action = T1.transform_points(points_action)
        goal_points_anchor = T1.transform_points(points_anchor)

        # Get the 'context' action point cloud
        goal_points_action_mean = goal_points_action.mean(dim=0)
        points_action = goal_points_action - goal_points_action_mean
        points_action = T0.transform_points(points_action)

        # Get transforms for metrics calculation and visualizations
        T_action2goal = T0.inverse().compose(
            Translate(goal_points_action_mean.unsqueeze(0))
        )

        data = {
            "pc": goal_points_action,  # Action points in goal position
            "pc_anchor": goal_points_anchor,  # Anchor points in goal position
            "pc_action": points_action,  # Action points for context
            "T0": T0.get_matrix().squeeze(0).T,
            "T1": T1.get_matrix().squeeze(0).T,
            "T_action2goal": T_action2goal.get_matrix().squeeze(0).T,
        }

        # If we have distractor anchor point clouds, add their transforms
        if self.dataset_cfg.distractor_anchor_pcds > 0:
            T_aug_action2goal_list = []
            for T_distractor in T_distractor_list:
                T_aug_action2goal = T_action2goal.compose(
                    T1.inverse()
                    .compose(Translate(center.unsqueeze(0)))
                    .compose(T_distractor)
                    .compose(Translate(-center.unsqueeze(0)))
                    .compose(T1)
                )
                T_aug_action2goal_list.append(T_aug_action2goal)

            data["T_distractor_list"] = torch.stack(
                [T.get_matrix().squeeze(0).T for T in T_distractor_list]
            )
            data["T_action2distractor_list"] = torch.stack(
                [T.get_matrix().squeeze(0).T for T in T_aug_action2goal_list]
            )

        # Support for RPDiff precision WTA error
        if "multi_obj_mesh_file" in demo and "multi_obj_final_obj_pose" in demo:
            assert (
                "rpdiff_descriptions_path" in self.dataset_cfg.misc_kwargs
            ), "rpdiff_descriptions_path must be provided in the dataset config"
            data["goal_action_center"] = goal_points_action_mean
            data["scene_center"] = center

            parent_fname = demo["multi_obj_mesh_file"].item()["parent"][0]
            parent_obj_name = (
                parent_fname.split("/")[-1].replace(".obj", "").replace("_dec", "")
            )
            parent_full_fname = os.path.join(
                self.dataset_cfg.misc_kwargs["rpdiff_descriptions_path"],
                parent_fname.split(parent_obj_name)[0].split("descriptions/")[-1],
            )
            saved_poses_path = os.path.join(
                parent_full_fname,
                "open_slot_poses",
                parent_obj_name + "_open_slot_poses.txt",
            )

            data["rpdiff_obj_mesh_file"] = demo["multi_obj_mesh_file"].item()
            data["rpdiff_saved_poses_path"] = saved_poses_path
            data["rpdiff_obj_final_obj_pose"] = demo["multi_obj_final_obj_pose"].item()
            data["rpdiff_pcd_scale_factor"] = self.dataset_cfg.pcd_scale_factor

        return data


DATASET_FN = {
    "rigid_point": RigidPointDataset,
    "rigid_flow": RigidFlowDataset,
    "ndf_point": NDFPointDataset,
    "rpdiff_point": RPDiffPointDataset,
}


class RigidDataModule(L.LightningModule):
    def __init__(
        self,
        root: Path,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        dataset_cfg: omegaconf.DictConfig,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dataset_cfg = dataset_cfg

    def prepare_data(self):
        pass

    def setup(self, stage: str) -> None:
        self.train_dataset = DATASET_FN[self.dataset_cfg.type](
            self.root,
            type="train",
            dataset_cfg=RigidDatasetCfg(
                **omegaconf.OmegaConf.to_container(self.dataset_cfg)  # type: ignore
            ),
        )
        self.val_dataset = DATASET_FN[self.dataset_cfg.type](
            self.root,
            type="val",
            dataset_cfg=RigidDatasetCfg(
                **omegaconf.OmegaConf.to_container(self.dataset_cfg)  # type: ignore
            ),
        )

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
