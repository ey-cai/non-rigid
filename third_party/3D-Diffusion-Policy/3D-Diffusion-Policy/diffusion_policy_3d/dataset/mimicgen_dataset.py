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

from pytorch3d.transforms import (
    Transform3d,
    Rotate,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    quaternion_to_matrix,
    matrix_to_quaternion,
)

import robosuite.utils.transform_utils as trans

import os

def random_so2(N=1):
    theta = torch.rand(N, 1) * 2 * np.pi
    axis_angle_z = torch.cat([torch.zeros(N, 2), theta], dim=1)
    R = axis_angle_to_matrix(axis_angle_z)
    return Rotate(R)


class MimicGenDataset(BaseDataset):
    def __init__(self,
            # zarr_path,
            root_dir, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            random_augment=False,
            ):
        super().__init__()
        self.root_dir = root_dir
        self.task_name = task_name
        self.random_augment = random_augment

        if self.random_augment:
            print('Training with random SO2 augment')
        else:
            print('Training without random SO2 augment')

        self.zarr_dir = root_dir
        train_zarr_path = os.path.join(self.zarr_dir, 'train.zarr')
        print(train_zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            train_zarr_path, keys=['point_cloud', 'state', 'action', 'action_pcd', 'anchor_pcd', 'ground_truth', 'gripper_pcd'])
        train_mask = np.ones(self.replay_buffer.n_episodes, dtype=bool)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_zarr_path = os.path.join(self.zarr_dir, 'val.zarr')
        val_set.replay_buffer = ReplayBuffer.copy_from_path(
            val_zarr_path, keys=['point_cloud', 'state', 'action'])
        val_mask = np.ones(val_set.replay_buffer.n_episodes, dtype=bool)
        val_set.sampler = SequenceSampler(
            replay_buffer=val_set.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
        )
        val_set.train_mask = val_mask
    
    def get_normalizer(self, mode='limits', **kwargs):
        # this function should only be called after action_pcd and anchor_pcd have already been combined
        data = {
            'action': self.replay_buffer['action'],
            # 'agent_pos': self.replay_buffer['state'][...,:],
            # 'point_cloud': self.replay_buffer['point_cloud'],
            # 'action_pcd': self.replay_buffer['action_pcd'],
            # 'anchor_pcd': self.replay_buffer['anchor_pcd'],
            # 'goal_pcd': self.replay_buffer['ground_truth']  # tax3d for predicted goal pcd
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data, last_n_dims=1, mode=mode,**kwargs)
        return normalizer
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32)
        action = sample['action'].astype(np.float32)
        point_cloud = sample['point_cloud'].astype(np.float32)
        action_pcd = sample['action_pcd'].astype(np.float32)
        anchor_pcd = sample['anchor_pcd'].astype(np.float32)
        gripper_pcd = sample['gripper_pcd'].astype(np.float32)
        goal_pcd = sample['ground_truth'].astype(np.float32)  # tax3d for predicted goal pcd

        data = {
            'obs': {
                'point_cloud': point_cloud, # 4, 1250, 3
                'agent_pos': agent_pos, # 4, 45
                'action_pcd': action_pcd, # 4, 625, 3
                'anchor_pcd': anchor_pcd, # 4, 625, 3
                'gripper_pcd': gripper_pcd, # 4, 625, 3
                'goal_pcd': goal_pcd # 4, 625, 3
            },
            'action': action # 4, 7
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, lambda x: torch.from_numpy(x))

        # if random se2 augment, center point cloud, rotate, and uncenter
        # also rotate action vectors
        if self.random_augment:
            # sample transform and compute mean across all timesteps
            T = random_so2()
            T_mat = T.get_matrix()[0, :3, :3]
            agent_pos = torch_data['obs']['agent_pos']
            action_pcd = torch_data['obs']['action_pcd']
            anchor_pcd = torch_data['obs']['anchor_pcd']
            gripper_pcd = torch_data['obs']['gripper_pcd']
            goal_pcd = torch_data['obs']['goal_pcd']
            action = torch_data['action']
            n_action_pcd = action_pcd.shape[1]
            n_anchor_pcd = anchor_pcd.shape[1]
            n_goal_pcd = goal_pcd.shape[1]
            
            all_point_cloud = torch.cat([action_pcd, anchor_pcd, goal_pcd], dim=1)
            all_point_cloud_mean = all_point_cloud.mean(dim=[0, 1], keepdim=True)

            # transform point cloud
            all_point_cloud = T.transform_points(all_point_cloud - all_point_cloud_mean) # + all_point_cloud_mean

            # transform agent pos
            agent_pos[:, 0:3] = T.transform_points(agent_pos[:, 0:3] - all_point_cloud_mean)
            origin_eef_mat = quaternion_to_matrix(agent_pos[:, 3:7])
            new_eef_mat = torch.matmul(T_mat, origin_eef_mat)
            new_quat = matrix_to_quaternion(new_eef_mat)
            agent_pos[:, 3:7] = new_quat
            agent_pos[:, 7:10] = T.transform_points(agent_pos[:, 7:10])

            # transform action
            action[:, 0:3] = T.transform_points(action[:, 0:3])
            origin_rot_mat = axis_angle_to_matrix(action[:, 3:6])
            new_rot_mat = torch.matmul(T_mat, origin_rot_mat)
            new_axis_angle = matrix_to_axis_angle(new_rot_mat)
            action[:, 3:6] = new_axis_angle
            # action[:, 3:6] = T.transform_points(action[:, 3:6])

            # update torch data
            torch_data['obs']['point_cloud'] = all_point_cloud
            torch_data['obs']['action_pcd'] = all_point_cloud[:, :n_action_pcd, :]
            torch_data['obs']['anchor_pcd'] = all_point_cloud[:, n_action_pcd:n_action_pcd+n_anchor_pcd, :]
            torch_data['obs']['goal_pcd'] = all_point_cloud[:, n_action_pcd+n_anchor_pcd:, :]
            torch_data['obs']['agent_pos'] = agent_pos
            torch_data['action'] = action
        return torch_data