import os
from pathlib import Path

import numpy as np
import lightning as L
import torch
import torch.utils.data as data

from pytorch3d.transforms import Transform3d, Translate
from pytorch3d.transforms import matrix_to_quaternion, matrix_to_rotation_6d

from non_rigid.utils.transform_utils import random_se3
from non_rigid.utils.pointcloud_utils import downsample_pcd 
from non_rigid.utils.augmentation_utils import plane_occlusion

class DedoDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.dataset_dir = self.root / self.split
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        # TODO: print a message when loading dataset?
        self.dataset_cfg = dataset_cfg

        # determining dataset size - if not specified, use all demos in directory once
        size = self.dataset_cfg.train_size if "train" in self.split else self.dataset_cfg.val_size
        if size is not None:
            self.size = size
        else:
            self.size = self.num_demos

        # setting sample sizes
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor

        # additional dataset params
        self.scene = self.dataset_cfg.scene
        self.world_frame = self.dataset_cfg.world_frame
        self.scene_anchor = self.dataset_cfg.scene_anchor

        # scene and scene-anchor checks
        # TODO: maybe move this to the datamodule?
        if self.scene and self.scene_anchor:
            raise ValueError("Cannot have both scene and scene_anchor set to True.")
        
        # TODO: this is here for consistency; just error out for scene-level dataset
        # experiments from now on should just be object-level
        if self.scene:
            raise NotImplementedError("Scene-level DEDO dataset not yet implemented.")
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index, return_indices=False, use_indices=None):
        """
        Args:
            return_indices: if True, return the indices used to downsample the point clouds.
            use_indices: if not None, use these indices to downsample the point clouds. If indices are provided,
                sample_size_action and sample_size_anchor are ignored.
        """
        # loop over the dataset multiple times - allows for arbitrary dataset and batch size
        file_index = index % self.num_demos
        # load data
        demo = np.load(self.dataset_dir / f"demo_{file_index}.npz", allow_pickle=True)
        action_pc = torch.as_tensor(demo["action_pc"]).float()
        anchor_pc = torch.as_tensor(demo["anchor_pc"]).float()
        flow = torch.as_tensor(demo["flow"]).float()

        # initializing item dict
        # TODO: eventually, these keys will have to update with newer DEDO env
        item = {
            # "rot": torch.as_tensor(demo["rot"]).float(),
            # "trans": torch.as_tensor(demo["trans"]).float(),
            "deform_transform": demo["deform_transform"].item(),
            "rigid_transform": demo["rigid_transform"].item(),
            "deform_params": demo["deform_params"].item(),
            "rigid_params": demo["rigid_params"].item(),
        }

        # downsample action
        if use_indices is not None and "action_pc_indices" in use_indices:
            action_pc_indices = use_indices["action_pc_indices"]
            action_pc = action_pc[action_pc_indices]
            flow = flow[action_pc_indices]
        elif self.sample_size_action > 0 and action_pc.shape[0] > self.sample_size_action:
            action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), self.sample_size_action, type=self.dataset_cfg.downsample_type)
            action_pc_indices = action_pc_indices.squeeze(0)
            action_pc = action_pc.squeeze(0)
            flow = flow[action_pc_indices]
        else:
            action_pc_indices = torch.arange(action_pc.shape[0])

        # downsample anchor
        if use_indices is not None and "anchor_pc_indices" in use_indices:
            anchor_pc_indices = use_indices["anchor_pc_indices"]
            anchor_pc = anchor_pc[anchor_pc_indices]
        elif self.sample_size_anchor > 0 and anchor_pc.shape[0] > self.sample_size_anchor:
            anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), self.sample_size_anchor, type=self.dataset_cfg.downsample_type)
            anchor_pc_indices = anchor_pc_indices.squeeze(0)
            anchor_pc = anchor_pc.squeeze(0)
        else:
            anchor_pc_indices = torch.arange(anchor_pc.shape[0])

        # return indices if specified
        if return_indices:
            item["action_pc_indices"] = action_pc_indices
            item["anchor_pc_indices"] = anchor_pc_indices


        # # downsample action
        # if self.sample_size_action > 0 and action_pc.shape[0] > self.sample_size_action:
        #     action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), self.sample_size_action, type=self.dataset_cfg.downsample_type)
        #     action_pc = action_pc.squeeze(0)
        #     flow = flow[action_pc_indices.squeeze(0)]

        # # downsample anchor
        # if self.sample_size_anchor > 0 and anchor_pc.shape[0] > self.sample_size_anchor:
        #     anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), self.sample_size_anchor, type=self.dataset_cfg.downsample_type)
        #     anchor_pc = anchor_pc.squeeze(0)

        # randomly occlude the anchor
        if self.dataset_cfg.anchor_occlusion:
            anchor_pc_temp, temp_mask = plane_occlusion(anchor_pc, return_mask=True)
            if anchor_pc_temp.shape[0] > self.sample_size_anchor:
                anchor_pc = anchor_pc_temp

        # compute goal action point cloud
        goal_action_pc = action_pc + flow

        # manually creating seg tensors
        seg = torch.ones_like(action_pc[:, 0]).int()
        seg_anchor = torch.zeros_like(anchor_pc[:, 0]).int()

        # apply scene-level augmentation
        T = random_se3(
            N=1,
            rot_var=self.dataset_cfg.rotation_variance,
            trans_var=self.dataset_cfg.translation_variance,
            rot_sample_method=self.dataset_cfg.scene_transform_type,
        )
        action_pc = T.transform_points(action_pc)
        anchor_pc = T.transform_points(anchor_pc)
        goal_action_pc = T.transform_points(goal_action_pc)

        # center the point clouds
        if self.dataset_cfg.center_type == "action_center":
            center = action_pc.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_center":
            center = anchor_pc.mean(axis=0)
        elif self.dataset_cfg.center_type == "scene_center":
            center = torch.cat([action_pc, anchor_pc], dim=0).mean(axis=0)
        elif self.dataset_cfg.center_type == "none":
            center = torch.zeros(3, dtype=torch.float32)
        else:
            raise ValueError(f"Invalid center type: {self.dataset_cfg.center_type}")
        
        if self.dataset_cfg.action_context_center_type == "center":
            action_center = action_pc.mean(axis=0)
        elif self.dataset_cfg.action_context_center_type == "none":
            action_center = torch.zeros(3, dtype=torch.float32)
        else:
            raise ValueError(f"Invalid action context center type: {self.dataset_cfg.action_context_center_type}")
        
        # handle scene-anchor processing
        if self.scene_anchor:
            anchor_pc = torch.cat([action_pc, anchor_pc], dim=0)
            seg_anchor = torch.cat([seg, seg_anchor], dim=0)
        
        goal_action_pc = goal_action_pc - center
        anchor_pc = anchor_pc - center
        action_pc = action_pc - action_center

        # updating item
        T_goal2world = Translate(center.unsqueeze(0)).compose(T.inverse())
        T_action2world = Translate(action_center.unsqueeze(0)).compose(T.inverse())

        gt_flow = goal_action_pc - action_pc
        # TODO: eventually, rename this key to "point"
        item["pc_action"] = action_pc # Action points in initial position for context
        item["pc_anchor"] = anchor_pc # Anchor points in goal position
        item["pc"] = goal_action_pc # Ground-truth action points
        item["flow"] = gt_flow # Ground-truth flow (cross-frame) to action points
        item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0)
        item["T_action2world"] = T_action2world.get_matrix().squeeze(0)
        # TODO: eventually, remove "seg" keys
        #item["seg"] = torch.ones_like(action_pc[:, 0]).int()
        #item["seg_anchor"] = torch.ones_like(anchor_pc[:, 0]).int()
        item["seg"] = seg
        item["seg_anchor"] = seg_anchor

        # handle relative pose
        if self.dataset_cfg.rel_pose:
            if self.dataset_cfg.rel_pose_type == "translation":
                # relative translation between action and anchor
                rel_pose = action_center - center
                item["rel_pose"] = rel_pose
            else:
                raise ValueError(f"Invalid relative pose type: {self.dataset_cfg.rel_pose_type}")
        
        return item


class DedoDataModule(L.LightningDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dataset_cfg = dataset_cfg
        self.stage = None

        # setting root directory based on dataset type
        data_dir = os.path.expanduser(self.dataset_cfg.data_dir)
        exp_dir = (
            f"cloth={self.dataset_cfg.cloth_geometry}-{self.dataset_cfg.cloth_pose} " + \
            f"anchor={self.dataset_cfg.anchor_geometry}-{self.dataset_cfg.anchor_pose} " + \
            f"hole={self.dataset_cfg.hole} " + \
            f"robot={self.dataset_cfg.robot}"
        )
        self.root = Path(data_dir) / self.dataset_cfg.task / exp_dir
    
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = "fit"):
        self.stage = stage

        # if not in train mode, don't use rotation augmentations
        if self.stage != "fit":
            print("-------Turning off rotation augmentation for validation/inference.-------")
            self.dataset_cfg.scene_transform_type = "identity"
            self.dataset_cfg.rotation_variance = 0.0
            self.dataset_cfg.translation_variance = 0.0
        # if world frame, don't mean-center the point clouds
        if self.dataset_cfg.world_frame:
            print("-------Turning off mean-centering for world frame predictions.-------")
            self.dataset_cfg.center_type = "none"
            self.dataset_cfg.action_context_center_type = "none"

        # initializing datasets
        self.train_dataset = DedoDataset(self.root, self.dataset_cfg, "train_tax3d")
        self.val_dataset = DedoDataset(self.root, self.dataset_cfg, "val_tax3d")
        self.val_ood_dataset = DedoDataset(self.root, self.dataset_cfg, "val_ood_tax3d")

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
    # batch can contain a list of dictionaries
    # we need to convert those to a dictionary of lists
    dict_keys = ["deform_transform", "rigid_transform", "deform_params", "rigid_params"]
    keys = batch[0].keys()
    out = {k: None for k in keys}
    for k in keys:
        if k in dict_keys:
        #if k == "deform_params":
            out[k] = [item[k] for item in batch]
        else:
            out[k] = torch.stack([item[k] for item in batch])
    return out