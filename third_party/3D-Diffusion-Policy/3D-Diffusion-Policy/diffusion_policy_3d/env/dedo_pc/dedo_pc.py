from dedo.utils.args import get_args, args_postprocess, CAM_CONFIG_DIR
from dedo.utils.pcd_utils import visualize_data, render_video
from dedo.envs import DeformEnvTAX3D


import gym
from gym import spaces
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops

import rpad.visualize_3d.plots as vpl


def downsample_with_fps(points: np.ndarray, num_points: int = 1024):
    # fast point cloud sampling using torch3d
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    num_points = torch.tensor([num_points]).cuda()
    # remember to only use coord to sample
    _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=num_points)
    points = points.squeeze(0).cpu().numpy()
    points = points[sampled_indices.squeeze(0).cpu().numpy()]
    return points


class DedoEnv:
    def __init__(self, task_name, viz, control_type='position', tax3d=False):
        self.task_name = task_name
        self.control_type = control_type
        self.tax3d = tax3d

        ###############################
        # configuring DEDO TAX3D environment
        ###############################
        args = get_args()
        # TODO: can move some of this stuff to task config

        if task_name == "proccloth":
            args.env = 'HangProcClothRobot-v0'
            args.max_episode_len = 300
        elif task_name == "hangbag":
            args.env = 'HangBag-v0'
            args.max_episode_len = 300
        else:
            raise ValueError(f"Unknown task name: {task_name}")

        args.tax3d = True
        args.rollout_vid = True
        args.viz = viz
        args_postprocess(args)

        kwargs = {'args': args}
        self.args = args
        self.env = gym.make(args.env, **kwargs)
        self.seed()

        print('Created environment: ', args.env)

        # TODO: are these needed?
        self.action_space = self.env.action_space

        # defining point cloud observation space
        self.agent_pos_dim = 12
        if self.tax3d:
            self.num_action_points = 625 # no downsampling
            self.num_anchor_points = 512
            self.observation_space = spaces.Dict({
                'pc_action': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_action_points, 3),
                    dtype=np.float32
                ),
                'pc_anchor': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_anchor_points, 3),
                    dtype=np.float32
                ),
                # 'seg': spaces.Box(
                #     low=-np.inf,
                #     high=np.inf,
                #     shape=(self.num_action_points,),
                #     dtype=np.float32
                # ),
                # 'seg_anchor': spaces.Box(
                #     low=-np.inf,
                #     high=np.inf,
                #     shape=(self.num_anchor_points,),
                #     dtype=np.float32
                # ),
            })
        else:
            self.num_points = 1024
            self.observation_space = spaces.Dict({
                # 'point_cloud': spaces.Box(
                #     low=-np.inf,
                #     high=np.inf,
                #     shape=(self.num_points, 3),
                #     dtype=np.float32
                # ),
                'action_pcd': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_points, 3),
                    dtype=np.float32
                ),
                'anchor_pcd': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_points, 3),
                    dtype=np.float32
                ),
                'agent_pos': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.agent_pos_dim,),
                    dtype=np.float32
                ),
            })

    def step(self, action):
        _, reward, done, info = self.env.step(action, self.control_type, tax3d=self.tax3d)
        next_obs = self.get_obs()
        return next_obs, reward, done, info

    def reset(self, deform_data={}, rigid_data={}):
        self.env.reset(
            deform_data=deform_data,
            rigid_data=rigid_data,
        )
        return self.get_obs()

    def get_obs(self):
        obs = self.env.get_obs()
        action_pcd = obs['action_pcd']
        action_seg = obs['action_seg']
        anchor_pcd = obs['anchor_pcd']
        anchor_seg = obs['anchor_seg']
        # TODO: since we can have multiple anchors, this should get a seg from the actual env

        if self.tax3d:
            # tax3d-specific observation; action/anchor segmentation, and full action point cloud
            obs_dict = {
                'pc_action': action_pcd,
                'pc_anchor': anchor_pcd,
                # 'seg': np.ones(action_pcd.shape[0]),
                # 'seg_anchor': np.zeros(anchor_pcd.shape[0]),
            }
        else:
            # anchor_pcd = downsample_with_fps(anchor_pcd, 1024)
            # point_cloud = np.concatenate([action_pcd, anchor_pcd], axis=0)
            # point_cloud = np.concatenate([action_pcd, anchor_pcd], axis=0)

            # if point_cloud.shape[0] > self.num_points:
            #     point_cloud = downsample_with_fps(point_cloud, self.num_points)
            obs_dict = {
                # 'point_cloud': point_cloud,
                'action_pcd': action_pcd,
                'action_seg': action_seg,
                'anchor_pcd': anchor_pcd,
                'anchor_seg': anchor_seg,
                'agent_pos': obs['gripper_state'],
            }
        return obs_dict

    def seed(self, seed=None):
        # if seed is not provided, use the one from args
        if seed is None:
            seed = self.args.seed
        self.env.seed(seed)
    
    def render(self, mode='rgb_array', width=300, height=300):
        return self.env.render(mode=mode, width=width, height=height)