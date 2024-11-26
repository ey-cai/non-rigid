from dataclasses import dataclass, field
from functools import lru_cache
import lightning as L
from non_rigid.utils.vis_utils import plot_multi_np
import numpy as np
import omegaconf
import os
from pathlib import Path
from pytorch3d.transforms import Transform3d, Translate, Rotate
import torch
import torch.utils.data as data
from typing import Dict

from non_rigid.utils.augmentation_utils import maybe_apply_augmentations
from non_rigid.utils.transform_utils import random_se3
from non_rigid.utils.pointcloud_utils import downsample_pcd, get_multi_anchor_scene

'''
@dataclass
class RigidDatasetCfg:
    name: str = "rigid"
    data_dir: str = "data/rigid"
    type: str = "train"
    scene: bool = False
    # Misc. kwargs to pass to the dataset e.g. dataset specific parameters
    misc_kwargs: Dict = field(default_factory=dict)

    ###################################################
    # General Dataset Parameters
    ###################################################
    # Number of demos to load
    num_demos: int = None
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
'''

class RigidPointDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, type):
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
    def __init__(self, root, dataset_cfg, type):
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


class NDFDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, type):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg

        # setting sample sizes
        # NOTE: scene is not implemented for now, thus not seg
        self.scene = self.dataset_cfg.scene
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor
        self.world_frame = self.dataset_cfg.world_frame
        
        # setting dataset meta info
        # NDF only has train&test, thus treat val=test
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
        elif self.type == "val" or self.type == "test":
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
            (self.type == "val" or self.type == "test")
            and not self.dataset_cfg.val_use_defaults
        ):
            # Apply augmentations to the point clouds in their final positions
            points_action, _ = maybe_apply_augmentations(
                init_points=points_action,
                final_points=None,
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
                random_drop_param={
                    "random_drop_probability": self.dataset_cfg.action_random_drop_probability,
                    "random_drop_percent": self.dataset_cfg.action_random_drop_percent,
                },
            )
            points_anchor, _ = maybe_apply_augmentations(
                init_points=points_anchor,
                final_points=None, 
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
                random_drop_param={
                    "random_drop_probability": self.dataset_cfg.anchor_random_drop_probability,
                    "random_drop_percent": self.dataset_cfg.anchor_random_drop_percent,
                },
            )
            
            # apply random scale uniformly to actiona and anchor pcd
            if torch.rand(1) < self.dataset_cfg.random_pcd_scaling_probability:
                random_scale_percent = np.random.uniform(self.dataset_cfg.random_pcd_scaling_min, self.dataset_cfg.random_pcd_scaling_max)
                points_action = points_action * random_scale_percent
                points_anchor = points_anchor * random_scale_percent

        if (self.type == "val" or self.type == "test") and self.dataset_cfg.val_use_defaults:
            # Downsample the point clouds
            points_action, _ = downsample_pcd(
                init_points=points_action.unsqueeze(0),
                final_points=None,
                num_points=self.dataset_cfg.sample_size_action,
                type="fps",
            )
            points_anchor, _ = downsample_pcd(
                init_points=points_anchor.unsqueeze(0),
                final_points=None,
                num_points=self.dataset_cfg.sample_size_anchor,
                type="fps",
            )
        else:
            # Downsample the point clouds
            points_action, _ = downsample_pcd(
                init_points=points_action.unsqueeze(0),
                final_points=None,
                num_points=self.dataset_cfg.sample_size_action,
                type=self.dataset_cfg.downsample_type,
            )
            points_anchor, _ = downsample_pcd(
                init_points=points_anchor.unsqueeze(0),
                final_points=None,
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

        # Get Goal2World and Action2World transforms
        T_goal2world = T1.inverse().compose(
            Translate(center.unsqueeze(0))
        )

        T_action2world = T_goal2world.compose(T_action2goal)


        flow = goal_points_action - points_action

        data = {
            "pc": goal_points_action,  # Action points in goal position
            "pc_anchor": goal_points_anchor,  # Anchor points in goal position
            "pc_action": points_action,  # Action points for context
            "flow": flow,
            "T0": T0.get_matrix().squeeze(0).T,
            "T1": T1.get_matrix().squeeze(0).T,
            "T_action2goal": T_action2goal.get_matrix().squeeze(0).T,
            "T_goal2world" : T_goal2world.get_matrix().squeeze(0),
            "T_action2world" : T_action2world.get_matrix().squeeze(0),
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

        return data


class RPDiffDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, type):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg

        self.dataset_dir = self.root / "task_name_mug_on_rack_multi"
        self.split_dir = self.dataset_dir / "split_info"
        self.split_file = f"{self.type}_split.txt" if self.type != "val" else "train_val_split.txt"

        print(f"Loading RPDiff dataset from {self.dataset_dir}")

        with open(os.path.join(self.split_dir, self.split_file), "r") as file:
            self.demo_files = [f"{self.dataset_dir}/{line.strip()}" for line in file]

        self.num_demos = len(self.demo_files)
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir} with {self.split_file}")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size if self.dataset_cfg.train_dataset_size is not None else len(self.demo_files)
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size if self.dataset_cfg.val_dataset_size is not None else len(self.demo_files)
        elif self.type == "test":
            return self.dataset_cfg.test_dataset_size if self.dataset_cfg.test_dataset_size is not None else len(self.demo_files)
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos], allow_pickle=True)

        # Access start and final PCDs for parent and child
        parent_start_pcd = demo['multi_obj_start_pcd'].item()['parent']
        child_start_pcd = demo['multi_obj_start_pcd'].item()['child']
        parent_final_pcd = demo['multi_obj_final_pcd'].item()['parent']
        child_final_pcd = demo['multi_obj_final_pcd'].item()['child']
        relative_trans = demo['relative_trans']

        action_pc = torch.as_tensor(child_start_pcd).float()
        anchor_pc = torch.as_tensor(parent_start_pcd).float()
        final_action_pc = torch.as_tensor(child_final_pcd).float()
        final_anchor_pc = torch.as_tensor(parent_final_pcd).float()

        R = relative_trans[:3, :3]
        t = relative_trans[:3, 3]
        T = Rotate(R).translate(t)
        
        # Apply scale factor
        action_pc = action_pc * self.dataset_cfg.pcd_scale_factor
        anchor_pc = anchor_pc * self.dataset_cfg.pcd_scale_factor
        final_action_pc = final_action_pc * self.dataset_cfg.pcd_scale_factor
        final_anchor_pc = final_anchor_pc * self.dataset_cfg.pcd_scale_factor


        if self.type == "train" or (
            (self.type == "val" or self.type == "test")
            and not self.dataset_cfg.val_use_defaults
        ):
            # Apply augmentations to the point clouds in their final positions
            action_pc, final_action_pc = maybe_apply_augmentations(
                init_points=action_pc,
                final_points=final_action_pc,
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
                random_drop_param={
                    "random_drop_probability": self.dataset_cfg.action_random_drop_probability,
                    "random_drop_percent": self.dataset_cfg.action_random_drop_percent,
                },
            )
            anchor_pc, final_anchor_pc = maybe_apply_augmentations(
                init_points=anchor_pc,
                final_points=final_anchor_pc, 
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
                random_drop_param={
                    "random_drop_probability": self.dataset_cfg.anchor_random_drop_probability,
                    "random_drop_percent": self.dataset_cfg.anchor_random_drop_percent,
                },
            )
            
            # apply random scale uniformly to actiona and anchor pcd
            if torch.rand(1) < self.dataset_cfg.random_pcd_scaling_probability:
                random_scale_percent = np.random.uniform(self.dataset_cfg.random_pcd_scaling_min, self.dataset_cfg.random_pcd_scaling_max)
                action_pc = action_pc * random_scale_percent
                anchor_pc = anchor_pc * random_scale_percent
                final_action_pc = final_action_pc * random_scale_percent
                final_anchor_pc = final_anchor_pc * random_scale_percent

        if (self.type == "val" or self.type == "test") and self.dataset_cfg.val_use_defaults:
            # Downsample the point clouds
            action_pc, final_action_pc = downsample_pcd(
                init_points=action_pc.unsqueeze(0),
                final_points=final_action_pc.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_action,
                type="fps",
            )
            anchor_pc, final_anchor_pc = downsample_pcd(
                init_points=anchor_pc.unsqueeze(0),
                final_points=final_anchor_pc.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_anchor,
                type="fps",
            )
        else:
            # Downsample the point clouds
            action_pc, final_action_pc = downsample_pcd(
                init_points=action_pc.unsqueeze(0),
                final_points=final_action_pc.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_action,
                type=self.dataset_cfg.downsample_type,
            )
            anchor_pc, final_anchor_pc = downsample_pcd(
                init_points=anchor_pc.unsqueeze(0),
                final_points=final_anchor_pc.unsqueeze(0),
                num_points=self.dataset_cfg.sample_size_anchor,
                type=self.dataset_cfg.downsample_type,
            )
        action_pc = action_pc.squeeze(0)
        anchor_pc = anchor_pc.squeeze(0)
        final_action_pc = final_action_pc.squeeze(0)
        final_anchor_pc = final_anchor_pc.squeeze(0)


        # Center the point clouds
        if self.dataset_cfg.center_type == "action_center":
            center = action_pc.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_center":
            center = anchor_pc.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_random":
            center = anchor_pc[np.random.choice(len(anchor_pc))]
        elif self.dataset_cfg.center_type == "none":
            center = np.zeros(3)
        else:
            raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")

        # Get the 'context' action point cloud
        if self.dataset_cfg.action_context_center_type == "center":
            action_center = action_pc.mean(axis=0)
        elif self.dataset_cfg.action_context_center_type == "random":
            action_center = action_pc[np.random.choice(len(action_pc))]
        elif self.dataset_cfg.action_context_center_type == "none":
            action_center = torch.zeros(3, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown action context center type: {self.dataset_cfg.action_context_center_type}")        

        goal_action_pc = final_action_pc - center
        anchor_pc = anchor_pc - center
        action_pc = action_pc - action_center

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


        goal_action_pc = T1.transform_points(goal_action_pc)
        anchor_pc = T1.transform_points(anchor_pc)
        action_pc = T0.transform_points(action_pc)

        T_goal2world = T1.inverse().compose(
            Translate(center.unsqueeze(0))
        )
        T_action2world = T0.inverse().compose(
            Translate(action_center.unsqueeze(0))
        )

        T_action2goal = T_goal2world.inverse().compose(T_action2world)
        
        gt_flow = goal_action_pc - action_pc
        data = {
            "pc": goal_action_pc,  # Action points in goal position
            "pc_action": action_pc,  # Action points for context
            "pc_anchor": anchor_pc,  # Anchor points in goal position
            "flow": gt_flow,
            "T0": T0.get_matrix().squeeze(0).T,
            "T1": T1.get_matrix().squeeze(0).T,
            "T_action2goal": T_action2goal.get_matrix().squeeze(0).T,
            "T_goal2world" : T_goal2world.get_matrix().squeeze(0),
            "T_action2world" : T_action2world.get_matrix().squeeze(0),
        }

        return data

DATASET_FN = {
    #"rigid_point": RigidPointDataset,
    #"rigid_flow": RigidFlowDataset,
    "ndf": NDFDataset,
    "rpdiff": RPDiffDataset,
}


class RigidDataModule(L.LightningModule):
    def __init__(
        self,
        #root: Path,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        dataset_cfg: omegaconf.DictConfig = None,
    ):
        super().__init__()
        #self.root = root
        data_dir = os.path.expanduser(dataset_cfg.data_dir)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dataset_cfg = dataset_cfg
        self.root = Path(data_dir)
    
    def prepare_data(self):
        pass

    def setup(self, stage: str) -> None:
        self.stage = stage

        # dataset sanity checks
        if self.dataset_cfg.scene and not self.dataset_cfg.world_frame:
            raise ValueError("Scene inputs require a world frame.")

        # if not in train mode, don't use rotation augmentations
        if self.stage != "fit":
            print("-------Turning off rotation augmentation for validation/inference.-------")
            self.dataset_cfg.action_transform_type = "identity"
            self.dataset_cfg.anchor_transform_type = "identity"
            self.dataset_cfg.action_translation_variance = 0
            self.dataset_cfg.action_rotation_variance = 0
            self.dataset_cfg.anchor_translation_variance = 0
            self.dataset_cfg.anchor_rotation_variance = 0

        # if world frame, don't mean-center the point clouds
        if self.dataset_cfg.world_frame:
            print("-------Turning off mean-centering for world frame predictions.-------")
            self.dataset_cfg.center_type = "none"
            self.dataset_cfg.action_context_center_type = "none"


        self.train_dataset = DATASET_FN[self.dataset_cfg.name](
            self.root, type="train", dataset_cfg=self.dataset_cfg)
        
        self.val_dataset = DATASET_FN[self.dataset_cfg.name](
            self.root, type="val", dataset_cfg=self.dataset_cfg)

        self.test_dataset = DATASET_FN[self.dataset_cfg.name](
            self.root, type="test", dataset_cfg=self.dataset_cfg)

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

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
