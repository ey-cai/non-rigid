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
)
from non_rigid.utils.pointcloud_utils import downsample_pcd
import os

def random_so2(N=1):
    theta = torch.rand(N, 1) * 2 * np.pi
    axis_angle_z = torch.cat([torch.zeros(N, 2), theta], dim=1)
    R = axis_angle_to_matrix(axis_angle_z)
    return Rotate(R)


class DedoDataset(BaseDataset):
    def __init__(self,
            # zarr_path,
            root_dir, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            # seed=42,
            # val_ratio=0.0,
            # max_train_episodes=None,
            # task_name=None,
            random_augment=False,
            cloth_geometry='single',
            cloth_pose='fixed',
            anchor_geometry='single',
            anchor_pose='random',
            hole='single',
            goal_conditioning='none',
            robot=True,
            action_size=512,
            anchor_size=512,
            ):
        super().__init__()
        self.root_dir = root_dir
        # self.task_name = task_name
        self.random_augment = random_augment

        self.cloth_geometry = cloth_geometry
        self.cloth_pose = cloth_pose
        self.anchor_geometry = anchor_geometry
        self.anchor_pose = anchor_pose
        self.hole = hole
        self.goal_conditioning = goal_conditioning
        self.robot = robot
        self.action_size = action_size
        self.anchor_size = anchor_size

        if self.random_augment:
            print('Training with random SO2 augment')
        else:
            print('Training without random SO2 augment')

        dataset_dir = (
            f'cloth={self.cloth_geometry}-{self.cloth_pose} ' + \
            f'anchor={self.anchor_geometry}-{self.anchor_pose} ' + \
            f'hole={self.hole} ' + \
            f'robot={self.robot}'
        )
        self.zarr_dir = os.path.join(root_dir, dataset_dir)
        train_zarr_path = os.path.join(self.zarr_dir, 'train.zarr')

        self.zarr_keys = ['state', 'action', 'cloth', 'action_pcd', 'anchor_pcd']
        if self.goal_conditioning.startswith('gt'):
            self.zarr_keys.append('ground_truth')
        elif self.goal_conditioning.startswith('tax3d'):
            self.zarr_keys.append('tax3d_pred')

        self.replay_buffer = ReplayBuffer.copy_from_path(
            train_zarr_path, keys=self.zarr_keys)
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
            val_zarr_path, keys=self.zarr_keys)
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
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data, last_n_dims=1, mode=mode,**kwargs)
        # normalizer['action_pcd'] = SingleFieldLinearNormalizer().create_identity()
        # normalizer['anchor_pcd'] = SingleFieldLinearNormalizer().create_identity()

        return normalizer
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def _sample_to_data(self, sample):
        # TODO: might have to convert action and anchor pcds into a single point cloud here
        agent_pos = sample['state'].astype(np.float32)
        action = sample['action'].astype(np.float32)
        cloth = sample['cloth'].astype(np.int16)
        action_pcd = sample['action_pcd'].astype(np.float32)
        anchor_pcd = sample['anchor_pcd'].astype(np.float32)
        # ground_truth = sample['ground_truth'].astype(np.float32)

        # extracting goal based on goal-conditioning type
        if self.goal_conditioning.startswith('gt'):
            goal = sample['ground_truth'].astype(np.float32)
        elif self.goal_conditioning.startswith('tax3d'):
            goal = sample['tax3d_pred'].astype(np.float32)

        data = {
            'obs': {
                'agent_pos': agent_pos, # T, D_pos
                'action_pcd': action_pcd, # T, N_action, 3
                'anchor_pcd': anchor_pcd, # T, N_anchor, 3
                'cloth' : cloth,
            },
            'action': action # T, D_action
        }

        # extracting and adding goal based on goal-conditioning type
        if self.goal_conditioning != 'none':
            data['obs']['goal'] = goal
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

            # masking out padding for cloth point cloud
            cloth_size = torch_data['obs']['cloth'][0].item() # cloths across horizon should always be the same size
            action_pcd = torch_data['obs']['action_pcd'][:, :cloth_size, :]
            anchor_pcd = torch_data['obs']['anchor_pcd']
            # ground_truth = torch_data['obs']['goal'][:, :cloth_size, :]

            # downsampling point clouds; action and ground-truth maintain correspondences
            _, action_indices = downsample_pcd(action_pcd[[0], ...], self.action_size, type='fps')
            _, anchor_indices = downsample_pcd(anchor_pcd[[0], ...], self.anchor_size, type='fps')
            action_indices = action_indices.squeeze()
            anchor_indices = anchor_indices.squeeze()

            action_pcd = action_pcd[:, action_indices, :]
            anchor_pcd = anchor_pcd[:, anchor_indices, :]
            # ground_truth = ground_truth[:, action_indices, :]

            # scene-centering
            scene_center = torch.cat([action_pcd, anchor_pcd], dim=1).mean(dim=1, keepdim=True)
            action_pcd = action_pcd - scene_center
            anchor_pcd = anchor_pcd - scene_center
            # ground_truth = ground_truth - scene_center

            # transform point cloud
            action_pcd = T.transform_points(action_pcd)
            anchor_pcd = T.transform_points(anchor_pcd)
            # ground_truth = T.transform_points(ground_truth)

            # combining point clouds, and processing goal if necessary
            if self.goal_conditioning == 'none':
                point_cloud = torch.cat([action_pcd, anchor_pcd], dim=1)
            else:
                goal = torch_data['obs']['goal'][..., :cloth_size, :]
                # for tax3d predictions, randomly sample one of the goals
                if self.goal_conditioning.startswith('tax3d'):
                    random_goal = torch.randint(0, goal.shape[1], (1,))
                    goal = goal[:, random_goal, ...].squeeze(1)
                goal = goal[:, action_indices, :]
                goal = goal - scene_center
                goal = T.transform_points(goal)

                if self.goal_conditioning.endswith('pcd'):
                    point_cloud = torch.cat([action_pcd, anchor_pcd, goal], dim=1)
                elif self.goal_conditioning.endswith('flow'):
                    point_cloud = torch.cat([action_pcd, anchor_pcd, goal - action_pcd], dim=1)

            agent_pos = torch_data['obs']['agent_pos']
            action = torch_data['action']
            scene_center = scene_center.squeeze()

            # transform agent pos
            agent_pos[:, 0:3] = T.transform_points(agent_pos[:, 0:3] - scene_center)
            agent_pos[:, 6:9] = T.transform_points(agent_pos[:, 6:9] - scene_center)
            agent_pos[:, 3:6] = T.transform_points(agent_pos[:, 3:6])
            agent_pos[:, 9:12] = T.transform_points(agent_pos[:, 9:12])

            # transform action
            action[:, 0:3] = T.transform_points(action[:, 0:3])
            action[:, 3:6] = T.transform_points(action[:, 3:6])

            # # update torch data
            torch_data['obs']['point_cloud'] = point_cloud
            torch_data['obs']['agent_pos'] = agent_pos
            torch_data['action'] = action
        return torch_data