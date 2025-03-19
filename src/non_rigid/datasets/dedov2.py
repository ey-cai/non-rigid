import os
from pathlib import Path

import numpy as np
import lightning as L
import torch
import torch.utils.data as data

from pytorch3d.transforms import Transform3d, Translate

from non_rigid.utils.transform_utils import random_se3
from non_rigid.utils.pointcloud_utils import downsample_pcd 
from non_rigid.utils.augmentation_utils import plane_occlusion

class Dedov2Dataset(data.Dataset):
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.dataset_dir = self.root / self.split
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
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

        # # additional dataset params
        # self.scene = self.dataset_cfg.scene
        # self.world_frame = self.dataset_cfg.world_frame
        # self.scene_anchor = self.dataset_cfg.scene_anchor

        assert self.dataset_cfg.tax3dv2, "Dedov2Dataset requires tax3dv2 model."

        # for now, anchor occlusion is disabled
        if self.dataset_cfg.anchor_occlusion:
            raise NotImplementedError("Anchor occlusion is not yet supported for DEDOv2 dataset.")
    
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
        action_seg = torch.as_tensor(demo["action_seg"]).int()
        anchor_seg = torch.as_tensor(demo["anchor_seg"]).int()

        # initializing item dict
        item = {
            "deform_data": demo["deform_data"].item(),
            "rigid_data": demo["rigid_data"].item(),
        }

        # downsample action 
        if use_indices is not None and "action_pc_indices" in use_indices:
            action_pc_indices = use_indices["action_pc_indices"]
            action_pc = action_pc[action_pc_indices]
            action_seg = action_seg[action_pc_indices]
            flow = flow[action_pc_indices]
        elif self.sample_size_action > 0 and action_pc.shape[0] > self.sample_size_action:
            action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), self.sample_size_action, type=self.dataset_cfg.downsample_type)
            action_pc_indices = action_pc_indices.squeeze(0)
            action_pc = action_pc.squeeze(0)
            action_seg = action_seg[action_pc_indices]
            flow = flow[action_pc_indices]
        else:
            action_pc_indices = torch.arange(action_pc.shape[0])

        # downsample anchor
        if use_indices is not None and "anchor_pc_indices" in use_indices:
            anchor_pc_indices = use_indices["anchor_pc_indices"]
            anchor_pc = anchor_pc[anchor_pc_indices]
            anchor_seg = anchor_seg[anchor_pc_indices]
        elif self.sample_size_anchor > 0 and anchor_pc.shape[0] > self.sample_size_anchor:
            anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), self.sample_size_anchor, type=self.dataset_cfg.downsample_type)
            anchor_pc_indices = anchor_pc_indices.squeeze(0)
            anchor_pc = anchor_pc.squeeze(0)
            anchor_seg = anchor_seg[anchor_pc_indices]
        else:
            anchor_pc_indices = torch.arange(anchor_pc.shape[0])

        # return indices if specified
        if return_indices:
            item["action_pc_indices"] = action_pc_indices
            item["anchor_pc_indices"] = anchor_pc_indices
        
        # compute goal action point cloud
        goal_action_pc = action_pc + flow

        # center scene point cloud, and apply random augmentation
        scene_center = torch.cat([action_pc, anchor_pc], dim=0).mean(axis=0)
        T = random_se3(
            N=1,
            rot_var=self.dataset_cfg.rotation_variance,
            trans_var=self.dataset_cfg.translation_variance,
            rot_sample_method=self.dataset_cfg.scene_transform_type,
        )
        goal_action_pc = T.transform_points(goal_action_pc - scene_center)
        action_pc = T.transform_points(action_pc - scene_center)
        anchor_pc = T.transform_points(anchor_pc - scene_center)

        # if diffusing reference frame, store the goal origin and update the goal action point cloud
        if self.dataset_cfg.diffuse_ref_frame:
            goal_origin = goal_action_pc.mean(axis=0)
            # handle noisy goal origin
            if self.dataset_cfg.noisy_goal_origin:
                goal_origin = goal_origin + torch.normal(mean=torch.zeros(3), std=torch.ones(3))
            goal_action_pc = goal_action_pc - goal_origin
            item["goal_origin"] = goal_origin
        
        # updating item dict
        T_goal2world = Translate(scene_center.unsqueeze(0)).compose(T.inverse())
        T_action2world = Translate(scene_center.unsqueeze(0)).compose(T.inverse())

        # TODO: this needs to handle flow prediction properly
        gt_flow = goal_action_pc - action_pc

        item["pc_action"] = action_pc # Initial action points in scene frame
        item["pc_anchor"] = anchor_pc # Anchor points in scene frame
        item["pc"] = goal_action_pc # Ground-truth goal action points in scene frame
        item["flow"] = gt_flow # Ground-truth goal flow
        item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0)
        item["T_action2world"] = T_action2world.get_matrix().squeeze(0)
        item["seg"] = action_seg
        item["seg_anchor"] = anchor_seg

        return item
    

class Dedov2DataModule(L.LightningDataModule):
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
            f"robot={self.dataset_cfg.robot} " + \
            f"num_anchors={self.dataset_cfg.num_anchors}"
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
        self.train_dataset = Dedov2Dataset(self.root, self.dataset_cfg, "train_tax3d")
        self.val_dataset = Dedov2Dataset(self.root, self.dataset_cfg, "val_tax3d")
        self.val_ood_dataset = Dedov2Dataset(self.root, self.dataset_cfg, "val_ood_tax3d")

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
    # dict_keys = ["deform_transform", "rigid_transform", "deform_params", "rigid_params"]
    dict_keys = ["deform_data", "rigid_data"]
    keys = batch[0].keys()
    out = {k: None for k in keys}
    for k in keys:
        if k in dict_keys:
        #if k == "deform_params":
            out[k] = [item[k] for item in batch]
        else:
            out[k] = torch.stack([item[k] for item in batch])
    return out