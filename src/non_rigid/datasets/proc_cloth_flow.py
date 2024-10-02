import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import torch.utils.data as data
import torchvision as tv
from torchvision import transforms as T

import rpad.visualize_3d.plots as vpl

import numpy as np
import torch_geometric.data as tgd
import torch_geometric.loader as tgl
import torch_geometric.transforms as tgt

from pathlib import Path
import os
from pytorch3d.transforms import Transform3d, Translate

from non_rigid.utils.transform_utils import random_se3
from non_rigid.utils.pointcloud_utils import downsample_pcd
from non_rigid.utils.augmentation_utils import ball_occlusion, plane_occlusion, maybe_apply_augmentations


class ProcClothFlowDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, type):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_dir = self.root / self.type
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        self.dataset_cfg = dataset_cfg

        # # setting dataset size
        # if "train" in self.type:
        #     self.size = self.dataset_cfg.train_size
        # else:
        #     self.size = self.dataset_cfg.val_size

        # determining dataset size - if not specified, use all demos in directory once
        size = self.dataset_cfg.train_size if "train" in self.type else self.dataset_cfg.val_size
        if size is not None:
            self.size = size
        else:
            self.size = self.num_demos

        # setting sample sizes
        self.scene = self.dataset_cfg.scene
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor
        self.world_frame = self.dataset_cfg.world_frame        
    
    def __len__(self):
        # return self.num_demos
        return self.size
    
    def __getitem__(self, index):
        file_index = index % self.num_demos

        # load data
        demo = np.load(self.dataset_dir / f"demo_{file_index}.npz", allow_pickle=True)
        action_pc = torch.as_tensor(demo["action_pc"]).float()
        action_seg = torch.as_tensor(demo["action_seg"]).int()
        anchor_pc = torch.as_tensor(demo["anchor_pc"]).float()
        anchor_seg = torch.as_tensor(demo["anchor_seg"]).int()
        flow = torch.as_tensor(demo["flow"]).float()

        # source-dependent fields
        if self.dataset_cfg.source == "dedo":
            speed_factor = torch.as_tensor(demo["speed_factor"]).float()
            rot = torch.as_tensor(demo["rot"]).float()
            trans = torch.as_tensor(demo["trans"]).float()
        elif self.dataset_cfg.source == "real":
            speed_factor = torch.ones(1)
            rot = torch.zeros(3)
            trans = torch.zeros(3)
        else:
            raise ValueError(f"Unknown source: {self.dataset_cfg.source}")

        # initializing item
        item = {
            "speed_factor": speed_factor,
            "rot": rot,
            "trans": trans,
        }
        # legacy, because some old demos don't have this field
        if "deform_params" in demo:
            item["deform_params"] = demo["deform_params"].item()

        # downsampling action point cloud
        if self.sample_size_action > 0 and action_pc.shape[0] > self.sample_size_action:
            action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), self.sample_size_action, type=self.dataset_cfg.downsample_type)
            action_pc = action_pc.squeeze(0)
            action_seg = action_seg[action_pc_indices.squeeze(0)]
            flow = flow[action_pc_indices.squeeze(0)]

        # downsampling anchor point cloud
        anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), self.sample_size_anchor, type=self.dataset_cfg.downsample_type)
        anchor_pc = anchor_pc.squeeze(0)
        anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]


        if self.scene:
            scene_pc = torch.cat([action_pc, anchor_pc], dim=0)
            scene_seg = torch.cat([action_seg, anchor_seg], dim=0)
            anchor_flow = torch.zeros_like(anchor_pc)
            scene_flow = torch.cat([flow, anchor_flow], dim=0)


            points_scene = scene_pc
            goal_points_scene = scene_pc + scene_flow

            # mean center about points_scene
            center = points_scene.mean(axis=0)
            points_scene = points_scene - center
            goal_points_scene = goal_points_scene - center


            # TODO: random transform based on anchor transform type
            T1 = random_se3(
                N=1,
                rot_var=self.dataset_cfg.rotation_variance,
                trans_var=self.dataset_cfg.translation_variance,
                rot_sample_method=self.dataset_cfg.anchor_transform_type,
            )
            points_scene = T1.transform_points(points_scene)
            goal_points_scene = T1.transform_points(goal_points_scene)
            T_goal2world = T1.inverse().compose(
                Translate(center.unsqueeze(0))
            )

            gt_flow = goal_points_scene - points_scene

            item["pc"] = goal_points_scene # Scene points in goal position
            item["pc_action"] = points_scene # Scene points in starting position
            item["seg"] = scene_seg
            item["flow"] = gt_flow
            item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0)

            # item["pc"] = scene_pc + scene_flow # Scene points in goal position
            # item["pc_action"] = scene_pc # Scene points in starting position
            # item["seg"] = scene_seg
            # item["flow"] = scene_flow
        else:
            points_action = action_pc + flow
            points_anchor = anchor_pc

            # Center the point clouds
            if self.dataset_cfg.center_type == "action_center":
                center = points_action.mean(axis=0)
            elif self.dataset_cfg.center_type == "anchor_center":
                center = points_anchor.mean(axis=0)
            elif self.dataset_cfg.center_type == "anchor_random":
                center = points_anchor[np.random.choice(len(points_anchor))]
            elif self.dataset_cfg.center_type == "none":
                center = torch.zeros(3, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
            goal_points_action = points_action - center
            goal_points_anchor = points_anchor - center

            # Transform the point clouds
            T0 = random_se3(
                N=1,
                rot_var=self.dataset_cfg.rotation_variance,
                trans_var=self.dataset_cfg.translation_variance,
                rot_sample_method=self.dataset_cfg.action_transform_type,
            )
            T1 = random_se3(
                N=1,
                rot_var=self.dataset_cfg.rotation_variance,
                trans_var=self.dataset_cfg.translation_variance,
                rot_sample_method=self.dataset_cfg.anchor_transform_type,
            )

            goal_points_action = T1.transform_points(goal_points_action)
            goal_points_anchor = T1.transform_points(goal_points_anchor)
            T_goal2world = T1.inverse().compose(
                Translate(center.unsqueeze(0))
            )

            if self.dataset_cfg.action_context_center_type == "center":
                action_center = action_pc.mean(axis=0)
            elif self.dataset_cfg.action_context_center_type == "random":
                action_center = action_pc[np.random.choice(len(action_pc))]
            elif self.dataset_cfg.action_context_center_type == "none":
                action_center = torch.zeros(3, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown action context center type: {self.dataset_cfg.action_context_center_type}")

            points_action = action_pc - action_center
            points_action = T0.transform_points(points_action)
            T_action2world = T0.inverse().compose(
                Translate(action_center.unsqueeze(0))
            )
            # Get the flow
            gt_flow = goal_points_action - points_action

            item["pc"] = goal_points_action # Action points in goal position
            item["pc_action"] = points_action # Action points for context
            item["pc_anchor"] = goal_points_anchor # Anchor points in goal position
            item["seg"] = action_seg
            item["seg_anchor"] = anchor_seg
            item["flow"] = gt_flow
            item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0)
            item["T_action2world"] = T_action2world.get_matrix().squeeze(0)
        return item



class ProcClothPointDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, type):
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_dir = self.root / self.type
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        self.dataset_cfg = dataset_cfg

        # # setting dataset size
        # if "train" in self.type:
        #     self.size = self.dataset_cfg.train_size
        # else:
        #     self.size = self.dataset_cfg.val_size

        # determining dataset size - if not specified, use all demos in directory once
        size = self.dataset_cfg.train_size if "train" in self.type else self.dataset_cfg.val_size
        if size is not None:
            self.size = size
        else:
            self.size = self.num_demos

        # setting sample sizes
        self.scene = self.dataset_cfg.scene
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor
        self.world_frame = self.dataset_cfg.world_frame

    def __len__(self):
        # return self.num_demos
        return self.size
    
    def __getitem__(self, index):
        file_index = index % self.num_demos
        # load data
        demo = np.load(self.dataset_dir / f"demo_{file_index}.npz", allow_pickle=True)
        action_pc = torch.as_tensor(demo["action_pc"]).float()
        action_seg = torch.as_tensor(demo["action_seg"]).int()
        anchor_pc = torch.as_tensor(demo["anchor_pc"]).float()
        anchor_seg = torch.as_tensor(demo["anchor_seg"]).int()
        flow = torch.as_tensor(demo["flow"]).float()

        # source-dependent fields
        if self.dataset_cfg.source == "dedo":
            speed_factor = torch.as_tensor(demo["speed_factor"]).float()
            rot = torch.as_tensor(demo["rot"]).float()
            trans = torch.as_tensor(demo["trans"]).float()
        elif self.dataset_cfg.source == "real":
            speed_factor = torch.ones(1)
            rot = torch.zeros(3)
            trans = torch.zeros(3)
        else:
            raise ValueError(f"Unknown source: {self.dataset_cfg.source}")

        # initializing item
        item = {
            "speed_factor": speed_factor,
            "rot": rot,
            "trans": trans,
        }
        # legacy, because some old demos don't have this field
        if "deform_params" in demo:
            item["deform_params"] = demo["deform_params"].item()

        # downsample action
        if self.sample_size_action > 0 and action_pc.shape[0] > self.sample_size_action:
            action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), self.sample_size_action, type=self.dataset_cfg.downsample_type)
            action_pc = action_pc.squeeze(0)
            action_seg = action_seg[action_pc_indices.squeeze(0)]
            flow = flow[action_pc_indices.squeeze(0)]


        # randomly occlude the anchor
        # anchor_pc_temp, temp_mask = plane_occlusion(anchor_pc, return_mask=True)
        # if anchor_pc_temp.shape[0] > self.sample_size_anchor:
        #     anchor_pc = anchor_pc_temp
        #     anchor_seg = anchor_seg[temp_mask]

        # downsample anchor
        anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), self.sample_size_anchor, type=self.dataset_cfg.downsample_type)
        anchor_pc = anchor_pc.squeeze(0)
        anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]

        # kinda confusing variables names for now, just for consistency
        points_action = action_pc + flow
        points_anchor = anchor_pc

        # Center the point clouds
        if self.dataset_cfg.center_type == "action_center":
            center = points_action.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_center":
            center = points_anchor.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_random":
            center = points_anchor[np.random.choice(len(points_anchor))]
        elif self.dataset_cfg.center_type == "none":
            center = torch.zeros(3, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
        goal_points_action = points_action - center
        goal_points_anchor = points_anchor - center


        # Randomly transform the point clouds
        T0 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.rotation_variance,
            trans_var=self.dataset_cfg.translation_variance,
            rot_sample_method=self.dataset_cfg.action_transform_type,
        )
        T1 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.rotation_variance,
            trans_var=self.dataset_cfg.translation_variance,
            rot_sample_method=self.dataset_cfg.anchor_transform_type,
        )

        goal_points_action = T1.transform_points(goal_points_action)
        goal_points_anchor = T1.transform_points(goal_points_anchor)
        T_goal2world = T1.inverse().compose(
            Translate(center.unsqueeze(0))
        )
        
        if self.dataset_cfg.action_context_center_type == "center":
            action_center = action_pc.mean(axis=0)
        elif self.dataset_cfg.action_context_center_type == "random":
            action_center = action_pc[np.random.choice(len(action_pc))]
        elif self.dataset_cfg.action_context_center_type == "none":
            action_center = torch.zeros(3, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown action context center type: {self.dataset_cfg.action_context_center_type}")

        points_action = action_pc - action_center
        points_action = T0.transform_points(points_action)
        T_action2world = T0.inverse().compose(
            Translate(action_center.unsqueeze(0))
        )

        item["pc"] = goal_points_action # Action points in goal position
        item["pc_action"] = points_action # Action points for context
        item["pc_anchor"] = goal_points_anchor # Anchor points in goal position
        item["seg"] = action_seg 
        item["seg_anchor"] = anchor_seg
        item["flow"] = flow
        item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0)
        item["T_action2world"] = T_action2world.get_matrix().squeeze(0)
        return item



class DeformablePlacementDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.dataset_dir = self.root / self.split
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        print(self.num_demos)
        self.dataset_cfg = dataset_cfg

        # determining dataset size - if not specified, use all demos in directory once
        size = self.dataset_cfg.train_size if "train" in self.split else self.dataset_cfg.val_size
        if size is not None:
            self.size = size
        else:
            self.size = self.num_demos

        # setting sample sizes
        self.scene = self.dataset_cfg.scene
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor
        self.world_frame = self.dataset_cfg.world_frame

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        # loop over the dataset multiple times - allows for arbitrary dataset and batch size
        file_index = index % self.num_demos
        # load data
        demo = np.load(self.dataset_dir / f"demo_{file_index}.npz", allow_pickle=True)
        action_pc = torch.as_tensor(demo["action_pc"]).float()
        action_seg = torch.as_tensor(demo["action_seg"]).int()
        anchor_pc = torch.as_tensor(demo["anchor_pc"]).float()
        anchor_seg = torch.as_tensor(demo["anchor_seg"]).int()
        flow = torch.as_tensor(demo["flow"]).float()

        # initializing item with source-dependent fields
        if self.dataset_cfg.source == "dedo":
            # speed_factor = torch.as_tensor(demo["speed_factor"]).float()
            # rot = torch.as_tensor(demo["rot"]).float()
            # trans = torch.as_tensor(demo["trans"]).float()
            # deform_params = demo["deform_params"].item()

            item = {
                "rot": torch.as_tensor(demo["rot"]).float(),
                "trans": torch.as_tensor(demo["trans"]).float(),
                "deform_params": demo["deform_params"].item(),
            }
        elif self.dataset_cfg.source == "real":
            # speed_factor = torch.ones(1)
            # rot = torch.zeros(3)
            # trans = torch.zeros(3)

            item = {
                "rot": torch.zeros(3),
                "trans": torch.zeros(3),
            }
        else:
            raise ValueError(f"Unknown source: {self.dataset_cfg.source}")
        
        # initializing item
        # item = {
        #     "rot": rot,
        #     "trans": trans,
        #     "deform_params": deform_params,
        #     # "speed_factor": speed_factor,
        # }

        # downsample action
        if self.sample_size_action > 0 and action_pc.shape[0] > self.sample_size_action:
            action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), self.sample_size_action, type=self.dataset_cfg.downsample_type)
            action_pc = action_pc.squeeze(0)
            action_seg = action_seg[action_pc_indices.squeeze(0)]
            flow = flow[action_pc_indices.squeeze(0)]

        # downsample anchor
        if self.sample_size_anchor > 0 and anchor_pc.shape[0] > self.sample_size_anchor:
            anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), self.sample_size_anchor, type=self.dataset_cfg.downsample_type)
            anchor_pc = anchor_pc.squeeze(0)
            anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]

        # randomly occlude the anchor
        if self.dataset_cfg.anchor_occlusion:
            anchor_pc_temp, temp_mask = plane_occlusion(anchor_pc, return_mask=True)
            if anchor_pc_temp.shape[0] > self.sample_size_anchor:
                anchor_pc = anchor_pc_temp
                anchor_seg = anchor_seg[temp_mask]

        # scene-level dataset
        if self.scene:
            scene_pc = torch.cat([action_pc, anchor_pc], dim=0)
            scene_seg = torch.cat([action_seg, anchor_seg], dim=0)
            anchor_flow = torch.zeros_like(anchor_pc)
            scene_flow = torch.cat([flow, anchor_flow], dim=0)
            goal_scene_pc = scene_pc + scene_flow

            # center the point clouds
            center = scene_pc.mean(axis=0)
            scene_pc = scene_pc - center
            goal_scene_pc = goal_scene_pc - center

            # transform the point clouds
            T1 = random_se3(
                N=1,
                rot_var=self.dataset_cfg.rotation_variance,
                trans_var=self.dataset_cfg.translation_variance,
                rot_sample_method=self.dataset_cfg.anchor_transform_type,
            )
            scene_pc = T1.transform_points(scene_pc)
            goal_scene_pc = T1.transform_points(goal_scene_pc)
            T_goal2world = T1.inverse().compose(
                Translate(center.unsqueeze(0))
            )

            gt_flow = goal_scene_pc - scene_pc
            item["pc"] = goal_scene_pc # Scene points in goal position
            item["pc_action"] = scene_pc # Scene points in starting position
            item["seg"] = scene_seg
            item["flow"] = gt_flow
            item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0)
        # object-centric dataset
        else:
            goal_action_pc = action_pc + flow

            # center the point clouds
            if self.dataset_cfg.center_type == "action_center":
                center = action_pc.mean(axis=0)
            elif self.dataset_cfg.center_type == "anchor_center":
                center = anchor_pc.mean(axis=0)
            elif self.dataset_cfg.center_type == "anchor_random":
                center = anchor_pc[np.random.choice(len(anchor_pc))]
            elif self.dataset_cfg.center_type == "none":
                center = torch.zeros(3, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
            
            if self.dataset_cfg.action_context_center_type == "center":
                action_center = action_pc.mean(axis=0)
            elif self.dataset_cfg.action_context_center_type == "random":
                action_center = action_pc[np.random.choice(len(action_pc))]
            elif self.dataset_cfg.action_context_center_type == "none":
                action_center = torch.zeros(3, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown action context center type: {self.dataset_cfg.action_context_center_type}")
            
            goal_action_pc = goal_action_pc - center
            anchor_pc = anchor_pc - center
            action_pc = action_pc - action_center

            # transform the point clouds
            T0 = random_se3(
                N=1,
                rot_var=self.dataset_cfg.rotation_variance,
                trans_var=self.dataset_cfg.translation_variance,
                rot_sample_method=self.dataset_cfg.action_transform_type,
            )
            T1 = random_se3(
                N=1,
                rot_var=self.dataset_cfg.rotation_variance,
                trans_var=self.dataset_cfg.translation_variance,
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

            gt_flow = goal_action_pc - action_pc
            item["pc"] = goal_action_pc # Action points in goal position
            item["pc_action"] = action_pc # Action points in starting position for context
            item["pc_anchor"] = anchor_pc # Anchor points in goal position
            item["seg"] = action_seg
            item["seg_anchor"] = anchor_seg
            item["flow"] = gt_flow
            item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0)
            item["T_action2world"] = T_action2world.get_matrix().squeeze(0)
        return item




DATASET_FN = {
    # "cloth": ProcClothFlowDataset,
    # "cloth_point": ProcClothPointDataset,
    "flow": DeformablePlacementDataset, #ProcClothFlowDataset,
    "point": DeformablePlacementDataset,# ProcClothPointDataset,
}


# TODO: rename this to Deformable Data Module or something
class ProcClothFlowDataModule(L.LightningDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg):# type, scene):
        super().__init__()
        # self.root = root
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.stage = None
        self.dataset_cfg = dataset_cfg

        # setting root directory based on dataset type
        data_dir = os.path.expanduser(dataset_cfg.data_dir)
        self.cloth_geometry = self.dataset_cfg.cloth_geometry
        self.cloth_pose = self.dataset_cfg.cloth_pose
        self.anchor_geometry = self.dataset_cfg.anchor_geometry
        self.anchor_pose = self.dataset_cfg.anchor_pose
        self.hole = self.dataset_cfg.hole
        exp_dir = (
            f'cloth={self.cloth_geometry}-{self.cloth_pose} ' + \
            f'anchor={self.anchor_geometry}-{self.anchor_pose} ' + \
            f'hole={self.hole}'
        )
        self.root = Path(data_dir) / exp_dir

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = "fit"):
        self.stage = stage

        # dataset sanity checks
        if self.dataset_cfg.scene and not self.dataset_cfg.world_frame:
            raise ValueError("Scene inputs require a world frame.")

        # if not in train mode, don't use rotation augmentations
        if self.stage != "fit":
            print("-------Turning off rotation augmentation for validation/inference.-------")
            self.dataset_cfg.action_transform_type = "identity"
            self.dataset_cfg.anchor_transform_type = "identity"
            self.dataset_cfg.rotation_variance = 0.0
            self.dataset_cfg.translation_variance = 0.0
        # if world frame, don't mean-center the point clouds
        if self.dataset_cfg.world_frame:
            print("-------Turning off mean-centering for world frame predictions.-------")
            self.dataset_cfg.center_type = "none"
            self.dataset_cfg.action_context_center_type = "none"
            #self.dataset_cfg.action_transform_type = "identity"
            #self.dataset_cfg.anchor_transform_type = "identity"
            #self.dataset_cfg.rotation_variance = 0.0
            #self.dataset_cfg.translation_variance = 0.0
        
        
        self.train_dataset = DATASET_FN[self.dataset_cfg.type](
            self.root, self.dataset_cfg, "train_tax3d"
        )
        self.val_dataset = DATASET_FN[self.dataset_cfg.type](
            self.root, self.dataset_cfg, "val_tax3d"
        )
        self.val_ood_dataset = DATASET_FN[self.dataset_cfg.type](
            self.root, self.dataset_cfg, "val_ood_tax3d"
        )
    
    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True if self.stage == "train" else False,
            num_workers=self.num_workers,
            collate_fn=cloth_collate_fn,
        )
    
    def val_dataloader(self):
        val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=cloth_collate_fn,
        )
        val_ood_dataloader = data.DataLoader(
            self.val_ood_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=cloth_collate_fn,
        )
        return val_dataloader, val_ood_dataloader


# custom collate function to handle deform params
def cloth_collate_fn(batch):
    # batch is a list of dictionaries
    # we need to convert it to a dictionary of lists
    keys = batch[0].keys()
    out = {k: None for k in keys}
    for k in keys:
        if k == "deform_params":
            out[k] = [item[k] for item in batch]
        else:
            out[k] = torch.stack([item[k] for item in batch])
    return out