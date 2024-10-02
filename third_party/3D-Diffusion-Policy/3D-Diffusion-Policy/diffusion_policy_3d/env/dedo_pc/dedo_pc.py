from dedo.utils.args import get_args, args_postprocess
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
            args.env = 'HangProcCloth-v0'
            args.max_episode_len = 300
        elif task_name == "hangbag":
            args.env = 'HangBag-v0'
            args.max_episode_len = 300
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        # TODO": this is the error
        
        # args.env = 'HangProcCloth-v0'
        args.tax3d = True
        args.rollout_vid = True
        args.pcd = True
        args.logdir = 'rendered'
        args.cam_config_path = '/home/ktsim/Projects/non-rigid/third_party/dedo/dedo/utils/cam_configs/camview_0.json'
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
                'seg': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_action_points,),
                    dtype=np.float32
                ),
                'seg_anchor': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_anchor_points,),
                    dtype=np.float32
                ),
            })
        else:
            self.num_points = 1024
            self.observation_space = spaces.Dict({
                'point_cloud': spaces.Box(
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

    def reset(self, 
              cloth_rot=None, 
              rigid_trans=None, 
              rigid_rot=None, 
              deform_params={}):
        self.env.reset(cloth_rot=cloth_rot,
                       rigid_trans=rigid_trans,
                       rigid_rot=rigid_rot,
                       deform_params=deform_params)
        return self.get_obs()

    def get_obs(self):
        obs = self.env.get_obs()

        action_pcd = obs['action_pcd']
        anchor_pcd = obs['anchor_pcd']

        if self.tax3d:
            # tax3d-specific observation; action/anchor segmentation, and full action point cloud
            obs_dict = {
                'pc_action': action_pcd,
                'pc_anchor': anchor_pcd,
                'seg': np.ones(action_pcd.shape[0]),
                'seg_anchor': np.zeros(anchor_pcd.shape[0]),
            }
        else:
            point_cloud = np.concatenate([action_pcd, anchor_pcd], axis=0)

            if point_cloud.shape[0] > self.num_points:
                point_cloud = downsample_with_fps(point_cloud, self.num_points)
            obs_dict = {
                'point_cloud': point_cloud,
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
    

# add a main function just to debug for now
if __name__ == '__main__':
    args = get_args()
    args.env = 'HangProcCloth-v0'
    args.tax3d = True
    args.pcd = True
    args.logdir = 'rendered'
    args.cam_config_path = '/home/eycai/Documents/dedo/dedo/utils/cam_configs/camview_0.json'
    args_postprocess(args)

    num_holes = 1
    deform_params = { # for single-cloth datasets
        'num_holes': num_holes,
        'node_density': 25,
        'w': 1.0,
        'h': 1.0,
        'holes': [{'x0': 8, 'x1': 16, 'y0': 9, 'y1': 13}]
    }

    kwargs = {'args': args}
    env = DedoEnv(args)
    env.seed()

    # reset
    obs = env.reset(deform_params=deform_params)
    # get action
    act = env.env.pseudo_expert_action(0)
    step = 0
    # rollout
    traj = []
    while True:
        assert (not isinstance(env.env.action_space, gym.spaces.Discrete))
        next_obs, rwd, done, info = env.step(act, 'velocity')
        traj.append(obs) # just obs for now, for debugging
        if done:
            break
        obs = next_obs
        step += 1

    # for o in traj:
    #     print(o['action_pcd'].shape, o['anchor_pcd'].shape, o['action_n'], o['anchor_n'])
    
    for i in range(0, len(traj), 20):
        obs_i = traj[i]
        obs_i = env.filter_obs(obs_i)
        action_pcd = obs_i['action_pcd']
        anchor_pcd = obs_i['anchor_pcd']
        action_seg = np.ones(action_pcd.shape[0])
        anchor_seg = np.zeros(anchor_pcd.shape[0])
        fig = vpl.segmentation_fig(
            np.concatenate([action_pcd, anchor_pcd], axis=0),
            np.concatenate([action_seg, anchor_seg], axis=0).astype(np.int32)
        )
        fig.show()
        # visualizer.visualize_pointcloud(
        #     np.concatenate([action_pcd, anchor_pcd], axis=0)
        # )