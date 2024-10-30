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
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            random_augment=False,
            cloth_geometry='single',
            cloth_pose='fixed',
            anchor_geometry='single',
            anchor_pose='random',
            hole='single',
            goal_conditioning='ground_truth',
            centroid='entire_scene'
            ):
        super().__init__()
        self.root_dir = root_dir
        self.task_name = task_name
        self.random_augment = random_augment

        self.cloth_geometry = cloth_geometry
        self.cloth_pose = cloth_pose
        self.anchor_geometry = anchor_geometry
        self.anchor_pose = anchor_pose
        self.hole = hole
        self.goal_conditioning = goal_conditioning
        self.centroid = centroid


        if self.random_augment:
            print('Training with random SO2 augment')
        else:
            print('Training without random SO2 augment')

        dataset_dir = (
            f'cloth={self.cloth_geometry}-{self.cloth_pose} ' + \
            f'anchor={self.anchor_geometry}-{self.anchor_pose} ' + \
            f'hole={self.hole}'
        )
        self.zarr_dir = os.path.join(root_dir, dataset_dir)
        train_zarr_path = os.path.join(self.zarr_dir, 'train.zarr')


        # Pointclooud includes action, anchor, and ground truth
        # Action is a force vector
        self.replay_buffer = ReplayBuffer.copy_from_path(
            train_zarr_path, keys=['point_cloud', 'state', 'action', 'action_pcd', 'anchor_pcd', 'cloth', 'ground_truth'])
        
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
        # val_mask = get_val_mask(
        #     n_episodes=self.replay_buffer.n_episodes, 
        #     val_ratio=val_ratio,
        #     seed=seed)
        # train_mask = ~val_mask
        # train_mask = downsample_mask(
        #     mask=train_mask, 
        #     max_n=max_train_episodes, 
        #     seed=seed)
        

        # breakpoint()

        # self.sampler = SequenceSampler(
        #     replay_buffer=self.replay_buffer, 
        #     sequence_length=horizon,
        #     pad_before=pad_before, 
        #     pad_after=pad_after,
        #     episode_mask=train_mask)
        # self.train_mask = train_mask
        # self.horizon = horizon
        # self.pad_before = pad_before
        # self.pad_after = pad_after

        # TODO: dont need to use get_val_mask, just load differnet split for val dataset :)
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_zarr_path = os.path.join(self.zarr_dir, 'val.zarr')
        val_set.replay_buffer = ReplayBuffer.copy_from_path(
            val_zarr_path, keys=['point_cloud', 'state', 'action', 'action_pcd', 'anchor_pcd', 'cloth', 'ground_truth'])
        val_mask = np.ones(val_set.replay_buffer.n_episodes, dtype=bool)
        val_set.sampler = SequenceSampler(
            replay_buffer=val_set.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
        )
        val_set.train_mask = val_mask


    # def get_validation_dataset(self):
    #     val_set = copy.copy(self)
    #     val_set.sampler = SequenceSampler(
    #         replay_buffer=self.replay_buffer, 
    #         sequence_length=self.horizon,
    #         pad_before=self.pad_before, 
    #         pad_after=self.pad_after,
    #         episode_mask=~self.train_mask
    #         )
    #     val_set.train_mask = ~self.train_mask
    #     return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        # Normalize force
        data = {
            'action': self.replay_buffer['action'],
            # 'agent_pos': self.replay_buffer['state'][...,:],
            # 'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data, last_n_dims=1, mode=mode,**kwargs)
        return normalizer
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def _sample_to_data(self, sample):
        # TODO: might have to convert action and anchor pcds into a single point cloud here
        agent_pos = sample['state'].astype(np.float32)
        action = sample['action'].astype(np.float32)
        point_cloud = sample['point_cloud'].astype(np.float32)  # [:, :1160, :]
        cloth = sample['cloth'].astype(np.int16)
        action_pcd = sample['action_pcd'].astype(np.float32)
        anchor_pcd = sample['anchor_pcd'].astype(np.float32)
        ground_truth = sample['ground_truth'].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 3, no rgb
                'agent_pos': agent_pos, # T, D_pos
                'action_pcd': action_pcd, # T, 580, 3
                'anchor_pcd': anchor_pcd, # T, 580, 3
                'cloth' : cloth,
                'ground_truth' : ground_truth,
            },
            'action': action, # T, D_action
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
            if self.cloth_geometry == 'multi':
                cloth_size = torch_data['obs']['cloth'][0].item() # cloths across horizon should always be the same size
                action_pcd = torch_data['obs']['action_pcd'][:, :cloth_size, :]
                anchor_pcd = torch_data['obs']['anchor_pcd']
                ground_truth = torch_data['obs']['ground_truth'][:, :cloth_size, :]

                if self.goal_conditioning == 'ground_truth':
                    point_cloud = torch.cat([action_pcd, anchor_pcd, ground_truth], dim=1) # includes action, achor, and ground truth

                elif self.goal_conditioning == 'none':
                    point_cloud = torch.cat([action_pcd, anchor_pcd], dim=1)

            elif self.cloth_geometry == 'single':
                if self.goal_conditioning == 'ground_truth':
                    point_cloud = torch_data['obs']['point_cloud']

                elif self.goal_conditioning == 'none':
                    point_cloud = torch_data['obs']['point_cloud'][:, :1160, :]

            agent_pos = torch_data['obs']['agent_pos']
            
            action = torch_data['action']

            # Entire Scene
            if self.centroid == 'entire_scene':
                point_cloud_mean = point_cloud.mean(dim=[0, 1], keepdim=True)
            elif self.centroid == 'cloth':
                point_cloud_mean = action_pcd.mean(dim=[0,1], keepdim=True)

            # transform point cloud
            point_cloud = T.transform_points(point_cloud - point_cloud_mean) # + point_cloud_mean

            # transform agent pos
            agent_pos[:, 0:3] = T.transform_points(agent_pos[:, 0:3] - point_cloud_mean) # + point_cloud_mean
            agent_pos[:, 6:9] = T.transform_points(agent_pos[:, 6:9] - point_cloud_mean) # + point_cloud_mean
            agent_pos[:, 3:6] = T.transform_points(agent_pos[:, 3:6])
            agent_pos[:, 9:12] = T.transform_points(agent_pos[:, 9:12])

            # transform action
            action[:, 0:3] = T.transform_points(action[:, 0:3])
            action[:, 3:6] = T.transform_points(action[:, 3:6])

            # update torch data
            if self.cloth_geometry == 'single':
                torch_data['obs']['point_cloud'] = point_cloud
                
            elif self.cloth_geometry == 'multi':
                if self.goal_conditioning == 'ground_truth':
                    torch_data['obs']['point_cloud'] = np.concatenate((point_cloud, 
                                                        np.zeros((point_cloud.shape[0], 1875 - point_cloud.shape[1], point_cloud.shape[2]))), 
                                                        axis=1)
                if self.goal_conditioning == 'none':
                    torch_data['obs']['point_cloud'] = np.concatenate((point_cloud, 
                                                        np.zeros((point_cloud.shape[0], 1250 - point_cloud.shape[1], point_cloud.shape[2]))), 
                                                        axis=1)
            torch_data['obs']['agent_pos'] = agent_pos
            torch_data['action'] = action
        return torch_data