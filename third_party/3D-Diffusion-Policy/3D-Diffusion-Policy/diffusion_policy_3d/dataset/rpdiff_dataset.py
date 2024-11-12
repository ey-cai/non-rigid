from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.common.se3 import random_se3

from pytorch3d.transforms import (
    Transform3d,
    Rotate,
    axis_angle_to_matrix,
)

import os
import os.path as osp

from rpdiff.utils import config_util, path_util
from rpdiff.training import dataio_full_chunked as dataio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class RPDiffDataset(BaseDataset):
    def __init__(self,
            # zarr_path,
            root_dir=None, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            random_augment=False,
            obj_config='mug-rack-multi',
            add_multi_obj_mesh_file=False,
            output_format='taxpose_dataset',
            split='train',
            batch_size=16
            ):
        super().__init__()
        self.root_dir = root_dir
        self.task_name = task_name
        self.random_augment = random_augment
        self.obj_config = obj_config
        self.add_multi_obj_mesh_file = add_multi_obj_mesh_file
        self.output_format = output_format
        self.split = split
        self.batch_size = batch_size

        self.OBJ_CONFIG_PATHS = {
            'book-bookshelf': "book_on_bookshelf_cfgs/book_on_bookshelf_pose_diff_with_varying_crop_fixed_noise_var.yaml",
            'mug-rack-multi': "mug_on_rack_multi_cfgs/mug_on_rack_multi_pose_diff_with_varying_crop_fixed_noise_var.yaml",
            'can-cabinet': "can_on_cabinet_cfgs/can_on_cabinet_pose_diff_with_varying_crop_fixed_noise_var.yaml",
            'mug-rack-single-hardrack': "mug_on_rack_multi_cfgs/mug_on_rack_single_hardrack_pose_diff_with_varying_crop_fixed_noise_var.yaml",
        }
        
        # ------------ Setup ------------- # 
        config_fname = self.OBJ_CONFIG_PATHS[self.obj_config]
        train_args = config_util.load_config(osp.join(path_util.get_train_config_dir(), config_fname))

        train_args = config_util.recursive_attr_dict(train_args)
        data_args = train_args.data

        # don't add any noise to the demonstrations
        data_args.refine_pose.diffusion_steps = 0

        dataset_path = osp.join(
            path_util.get_rpdiff_data(), 
            data_args.data_root,
            data_args.dataset_path)

        assert osp.exists(dataset_path), f'Dataset path: {dataset_path} does not exist'

        self.dataset = dataio.FullRelationPointcloudPolicyDataset(
            dataset_path, 
            data_args,
            phase=self.split, 
            train_coarse_aff=False,
            train_refine_pose=True,
            train_success=False,
            mc_vis=False, 
            debug_viz=False)
    
    def get_dataloader(self):
        if self.split == 'train':
            return DataLoader(
                    RPDiffDatasetWrapper(self.dataset, rot_sample_method="axis_angle", output_format=self.output_format, add_multi_obj_mesh_file=self.add_multi_obj_mesh_file), 
                    batch_size=self.batch_size, 
                    shuffle=True,
                    num_workers=1,
                    drop_last=True)
        elif self.split == 'val':
            return DataLoader(
                    RPDiffDatasetWrapper(self.dataset, rot_sample_method="axis_angle", output_format=self.output_format, add_multi_obj_mesh_file=self.add_multi_obj_mesh_file), 
                    batch_size=self.batch_size, 
                    num_workers=1,
                    shuffle=False, 
                    drop_last=True)
        else:
            raise NotImplementedError


class RPDiffDatasetWrapper(Dataset):
    def __init__(self, 
                 original_dataset,
                 rotation_variance=np.pi,
                 translation_variance=0.5,
                 overfit=False,
                 num_overfit_transforms=3,
                 seed_overfit_transforms=False,
                 set_Y_transform_to_identity=False,
                 set_Y_transform_to_overfit=False,
                 rot_sample_method="axis_angle",
                 output_format="taxpose_dataset",
                 add_multi_obj_mesh_file=False
                 ):
        self.original_dataset = original_dataset

        # taxpose_dataset matches the data outputted by a taxpose dataloader
        # taxpose_raw_dataset matches the data required by a taxpose dataloader
        self.output_format = output_format
        assert self.output_format in ["taxpose_dataset", "taxpose_raw_dataset"]
        
        self.add_multi_obj_mesh_file = add_multi_obj_mesh_file

        if self.output_format == "taxpose_dataset":
            # The output of a taxpose dataloader would apply random transforms to the demonstrations
            self.rot_var = rotation_variance
            self.trans_var = translation_variance

            self.overfit = overfit
            self.seed_overfit_transforms = seed_overfit_transforms
            # identity has a higher priority than overfit
            self.set_Y_transform_to_identity = set_Y_transform_to_identity
            self.set_Y_transform_to_overfit = set_Y_transform_to_overfit
            if self.set_Y_transform_to_identity:
                self.set_Y_transform_to_overfit = True
            self.num_overfit_transforms = num_overfit_transforms
            self.T0_list = []
            self.T1_list = []
            self.T2_list = []

            self.rot_sample_method = rot_sample_method

            if self.overfit or self.set_Y_transform_to_overfit:
                self.get_fixed_transforms()

    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        sample = self.original_dataset[idx]

        # Parse out the demonstration data
        # These are given the batch dimension
        points_anchor = torch.tensor(sample[1][1]['parent_final_pcd'])[None,].float()
        points_action = torch.tensor(sample[1][1]['child_final_pcd'])[None,].float()

        if self.output_format == "taxpose_raw_dataset":
            clouds = torch.cat([points_anchor, points_action], dim=1)

            B = clouds.shape[0]
            classes = torch.cat([torch.ones((B, points_anchor.shape[1])), torch.zeros((B, points_action.shape[1]))], dim=1)

            data = {
                'clouds': clouds.squeeze(0),
                'classes': classes.squeeze(0),
                'symmetric_cls': torch.tensor([]),
            }
        elif self.output_format == "taxpose_dataset":
            ########
            # The below is transformation code copied from TAXPose's PointCloudDataset

            if self.overfit:
                transform_idx = torch.randint(
                    self.num_overfit_transforms, (1,)).item()
                T0 = self.T0_list[transform_idx]
                T1 = self.T1_list[transform_idx]
                if not self.set_Y_transform_to_overfit:
                    T2 = random_se3(1, rot_var=self.rot_var,
                                    trans_var=self.trans_var, device=points_anchor.device, rot_sample_method=self.rot_sample_method)
                else:
                    T2 = self.T2_list[transform_idx]
            else:
                T0 = random_se3(1, rot_var=self.rot_var,
                                trans_var=self.trans_var, device=points_action.device, rot_sample_method=self.rot_sample_method)
                T1 = random_se3(1, rot_var=self.rot_var,
                                trans_var=self.trans_var, device=points_anchor.device, rot_sample_method=self.rot_sample_method)
                if self.set_Y_transform_to_identity:
                    T2 = Rotate(torch.eye(3), device=points_anchor.device)
                elif self.set_Y_transform_to_overfit:
                    transform_idx = torch.randint(
                            self.num_overfit_transforms, (1,)).item()
                    T2 = self.T2_list[transform_idx]
                else:
                    T2 = random_se3(1, rot_var=self.rot_var,
                                    trans_var=self.trans_var, device=points_anchor.device, rot_sample_method=self.rot_sample_method)

            points_action_trans = T0.transform_points(points_action)
            points_anchor_trans = T1.transform_points(points_anchor)

            points_action_onetrans = T2.transform_points(points_action)
            points_anchor_onetrans = T2.transform_points(points_anchor)

            data = {
                'points_action': points_action.squeeze(0),
                'points_anchor': points_anchor.squeeze(0),
                'points_action_trans': points_action_trans.squeeze(0),
                'points_anchor_trans': points_anchor_trans.squeeze(0),
                'points_action_onetrans': points_action_onetrans.squeeze(0),
                'points_anchor_onetrans': points_anchor_onetrans.squeeze(0),
                'T0': T0.get_matrix().squeeze(0),
                'T1': T1.get_matrix().squeeze(0),
                'T2': T2.get_matrix().squeeze(0),
                'symmetric_cls': torch.tensor([]),
                # 'mug_id': filename.name.split("_")[0],
            }
            
        if self.add_multi_obj_mesh_file:
            data['multi_obj_mesh_file'] = sample[-2].item()
            data['multi_obj_final_obj_pose'] = sample[-1].item()

        return data