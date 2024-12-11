import wandb
import numpy as np
import torch
import collections
import tqdm
from termcolor import cprint
import os
import imageio

import torch.utils.data as data

# from diffusion_policy_3d.env import DedoEnv
import mimicgen.utils.robomimic_utils as RobomimicUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util

from non_rigid.utils.vis_utils import plot_diffusion

from PIL import Image

class MimicGenRunner(BaseRunner):
    def __init__(self,
                 mg_config,
                 output_dir,
                 n_episodes=20,
                 # max_steps=200, # TODO: also don't need max steps, env has it already
                 n_obs_steps=8, # don't need multi step
                 n_action_steps=8, # don't need multi step
                 fps=10,
                 # crf=22, # unclear what this is for
                 # render_size=84, # unclear what this is for
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 viz=False,
                 control_type='position', # position or velocity
                 tax3d=False,
                 tax3d_pred=False,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.tax3d = tax3d
        self.use_goal_pred = tax3d_pred
        self.vid_speed = 3
        self.diffusion_gif_speed = 2

        # get env information
        source_dataset_path = os.path.expandvars(os.path.expanduser(mg_config.experiment.source.dataset_path))
        env_meta = get_env_metadata_from_dataset(dataset_path=source_dataset_path)

        # steps_per_render = max(10 // fps, 1)

        def env_fn():
            env = RobomimicUtils.create_env(
                env_meta=env_meta,
                env_class=None,
                env_name=mg_config.experiment.task.name,
                robot=mg_config.experiment.task.robot,
                gripper=mg_config.experiment.task.gripper,
                camera_names=mg_config.obs.camera_names,
                camera_height=mg_config.obs.camera_height,
                camera_width=mg_config.obs.camera_width,
                render=viz,  # False 
                render_offscreen=False,
                use_image_obs=True,
                use_depth_obs=True,
            )
            # return env
            return MultiStepWrapper(
                env,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                # max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.n_episodes = n_episodes
        self.env = env_fn()

        self.fps = fps
        #self.crf = crf
        #self.n_obs_steps = n_obs_steps
        #self.n_action_steps = n_action_steps
        #self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        # TODO: this can also take as input the specific environment settings to run on
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_successes = []
        centroid_dists = []


        for episode_id in tqdm.tqdm(
            range(self.n_episodes), 
            desc=f"MIMICGEN {self.task_name} Env", leave=False, 
            mininterval=self.tqdm_interval_sec,
        ):
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            # don't need to iterate through max steps
            while True:
                # create obs dict
                np_obs_dict = dict(obs)

                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))
                
                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)


                # device transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)

                if done:
                    break
            
            # update metrics
            all_successes.append(info['is_success'])
            centroid_dists.append(info['centroid_dist'])
        
        # log 
        log_data = dict()

        log_data['mean_success'] = np.mean(all_successes)
        log_data['mean_centroid_dist'] = np.mean(centroid_dists)

        log_data['test_mean_score'] = np.mean(all_successes)

        self.logger_util_test.record(np.mean(all_successes))
        self.logger_util_test10.record(np.mean(all_successes))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        del env
        return log_data
    

    def run_dataset(self, policy: BasePolicy, dataset: data.Dataset, dataset_name: str):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        output_save_dir = os.path.join(self.output_dir, dataset_name)
        rollout_save_dir = os.path.join(output_save_dir, f'rollout_videos')

        # creating directory for outputs
        if os.path.exists(output_save_dir):
            cprint(f"Output directory {output_save_dir} already exists. Overwriting...", 'red')
            os.system('rm -rf {}'.format(output_save_dir))
            os.system('rm -rf {}'.format(rollout_save_dir))
        os.makedirs(output_save_dir, exist_ok=True)
        os.makedirs(rollout_save_dir, exist_ok=True)

        # video_path = os.path.join(output_save_dir, f'rollout_video.mp4')
        # video_writer = imageio.get_writer(video_path, fps=20)

        all_successes = []
        centroid_dists = []
        num_successes = 0


        pbar = tqdm.tqdm(
            range(len(dataset)),
            desc=f"MIMICGEN {self.task_name} Env", leave=False,
            mininterval=self.tqdm_interval_sec,
        )

        # for id in tqdm.tqdm(
        #     range(len(dataset)),
        #     desc=f"DEDO {self.task_name} Env", leave=False,
        #     mininterval=self.tqdm_interval_sec,
        # ):
        for id in pbar:
            pbar.set_description(f"MIMICGEN {self.task_name} Env ({num_successes})")

            # set video writer
            video_path = os.path.join(rollout_save_dir, f'demo_{id}.mp4')
            video_writer = imageio.get_writer(video_path, fps=20)

            # get initial state
            demo = dataset[id]
            initial_state = {'states': demo['state']}
            if self.use_goal_pred:
                goal_pc = demo['tax3d']
            else:
                goal_pc = demo['action_pc'] + demo['flow']
            goal_pc = torch.from_numpy(goal_pc).to(device=device)

            # load the initial state
            env.reset()
            obs = env.reset_to(initial_state)
            policy.reset()

            done = False
            success = { k: False for k in env.is_success() }
            video_count = 0
            # don't need to iterate through max steps
            while True:
                # create obs dict
                np_obs_dict = dict(obs)

                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x.copy()).to(
                                          device=device))
                
                
                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    if self.tax3d:
                        obs_dict_input['pc_action'] = obs_dict['pc_action'].float()
                        obs_dict_input['pc_anchor'] = obs_dict['pc_anchor'].float()
                        obs_dict_input['seg'] = obs_dict['seg'].int()
                        obs_dict_input['seg_anchor'] = obs_dict['seg_anchor'].int()
                        action_dict = policy.predict_action(obs_dict_input, deform_params)
                    else:
                        obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)  # [1, 2, 14]
                        obs_dict_input['action_pcd'] = obs_dict['pointcloud'][:, 0, :, :].unsqueeze(0)  # [1, 2, 625, 3]
                        obs_dict_input['anchor_pcd'] = obs_dict['pointcloud'][:, 1, :, :].unsqueeze(0)  # [1, 2, 625, 3]

                        bsz = obs_dict_input['anchor_pcd'].shape[0]
                        hor = obs_dict_input['anchor_pcd'].shape[1]
                        obs_dict_input['goal_pcd'] = goal_pc.unsqueeze(0).repeat(bsz,hor,1,1)
                        obs_dict_input['point_cloud'] = torch.cat([obs_dict_input['action_pcd'],
                                                                   obs_dict_input['anchor_pcd'],
                                                                   obs_dict_input['goal_pcd']], dim=-2)

                        # import open3d as o3d
                        # import numpy as np
                        # point_geometry = o3d.geometry.PointCloud()
                        # goal_geometry = o3d.geometry.PointCloud()
                        # anchor_geometry = o3d.geometry.PointCloud()
                        # point_geometry.points = o3d.utility.Vector3dVector(obs_dict_input['action_pcd'][0][0].cpu().numpy())
                        # point_geometry.paint_uniform_color(np.array([0, 0, 1]))
                        # goal_geometry.points = o3d.utility.Vector3dVector(obs_dict_input['goal_pcd'][0][0].cpu().numpy())
                        # goal_geometry.paint_uniform_color(np.array([1, 0, 0]))
                        # anchor_geometry.points = o3d.utility.Vector3dVector(obs_dict_input['anchor_pcd'][0][0].cpu().numpy())
                        # anchor_geometry.paint_uniform_color(np.array([0, 1, 0]))
                        # o3d.visualization.draw_geometries([point_geometry, goal_geometry, anchor_geometry])
                        # exit()

                        action_dict = policy.predict_action(obs_dict_input)

                # device transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                # step env
                obs, reward, done, info = env.step(action)

                # update success
                cur_success_metrics = env.is_success()
                for k in success:
                    success[k] = success[k] or cur_success_metrics[k]
                info['is_success'] = bool(success["task"])

                env.render(mode="human", camera_name='agentview')
                # # video render
                # print(video_count)
                # if video_count % 5 == 0:
                #     video_img = []
                #     camera_names = ['agentview']
                #     for cam_name in camera_names:
                #         video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                #     video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                #     print(video_img.shape)
                #     exit()
                #     video_writer.append_data(video_img)
                # video_count += 1

                if done:
                    if info['is_success']:
                        num_successes += 1
                    break

            # update metrics
            all_successes.append(info['is_success'])
            # centroid_dists.append(info['centroid_dist'])
            video_writer.close()

        # log
        log_data = dict()

        log_data['mean_success'] = np.mean(all_successes)
        # log_data['mean_centroid_dist'] = np.mean(centroid_dists)

        log_data['test_mean_score'] = np.mean(all_successes)

        self.logger_util_test.record(np.mean(all_successes))
        self.logger_util_test10.record(np.mean(all_successes))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        del env
        return log_data