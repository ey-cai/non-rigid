import wandb
import numpy as np
import torch
import collections
import tqdm
from termcolor import cprint
import os

import torch.utils.data as data

from diffusion_policy_3d.env import DedoEnv

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util

from non_rigid.utils.vis_utils import plot_diffusion

from PIL import Image

class DedoRunner(BaseRunner):
    def __init__(self,
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
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.tax3d = tax3d
        self.vid_speed = 3
        self.diffusion_gif_speed = 2

        # steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                DedoEnv(task_name=task_name, viz=viz, control_type=control_type, tax3d=tax3d),
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

        #################################################
        # determining experiment type based on task  name
        #################################################
        self.deform_params = {}
        if self.task_name == 'dedo':
            num_holes = 1
            self.deform_params = { # for single-cloth datasets
                'num_holes': num_holes,
                'node_density': 25,
                'w': 1.0,
                'h': 1.0,
                'holes': [{'x0': 8, 'x1': 16, 'y0': 9, 'y1': 13}]
            }


    def run(self, policy: BasePolicy):
        # TODO: this can also take as input the specific environment settings to run on
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_successes = []
        centroid_dists = []


        for episode_id in tqdm.tqdm(
            range(self.n_episodes), 
            desc=f"DEDO {self.task_name} Env", leave=False, 
            mininterval=self.tqdm_interval_sec,
        ):
            # start rollout
            # TODO: env reset should take in deform params and configuration
            obs = env.reset(deform_params=self.deform_params)
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
                    # TODO: add batch dim
                    # TODO: flush unused keys
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

        ##################################################
        # SOME REALLY HACKY PRE-PROCESSING STUFF
        ##################################################

        # still determining right amount of steps for episode
        output_save_dir = os.path.join(self.output_dir, dataset_name)
        # output_save_dir = os.path.join(self.output_dir, f"{dataset_name}_200_original_gains")
        # self.env.env.env.max_episode_len = 200

        # set environment to randomize cloth color

        ##################################################
        # END OF REALLY HACKY PRE-PROCESSING STUFF
        ##################################################

        # creating directory for outputs
        if os.path.exists(output_save_dir):
            cprint(f"Output directory {output_save_dir} already exists. Overwriting...", 'red')
            os.system('rm -rf {}'.format(output_save_dir))
        os.makedirs(output_save_dir, exist_ok=True)

        all_successes = []
        centroid_dists = []
        num_successes = 0


        pbar = tqdm.tqdm(
            range(len(dataset)),
            desc=f"DEDO {self.task_name} Env", leave=False,
            mininterval=self.tqdm_interval_sec,
        )

        for id in pbar:
            pbar.set_description(f"DEDO {self.task_name} Env ({num_successes})")
            # get rot, trans, deform params
            demo = dataset[id]
            rot = demo['rot']
            trans = demo['trans']
            deform_params = demo['deform_params'][()]
            goal_pc = demo['action_pc'] + demo['flow']
            goal_pc = torch.from_numpy(goal_pc).to(device=device)

            obs = env.reset(
                rigid_rot=rot,
                rigid_trans=trans,
                deform_params=deform_params,
            )
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
                    if self.tax3d:
                        obs_dict_input['pc_action'] = obs_dict['pc_action'].float()
                        obs_dict_input['pc_anchor'] = obs_dict['pc_anchor'].float()
                        obs_dict_input['seg'] = obs_dict['seg'].int()
                        obs_dict_input['seg_anchor'] = obs_dict['seg_anchor'].int()
                        action_dict = policy.predict_action(obs_dict_input, deform_params)
                    else:
                        obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                        obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                        
                        bsz = obs_dict_input['point_cloud'].shape[0]
                        hor = obs_dict_input['point_cloud'].shape[1]
                        goal_pointcloud = goal_pc.unsqueeze(0).unsqueeze(0).repeat(bsz,hor,1,1)
                        # obs_dict_input['point_cloud'] = torch.cat([obs_dict_input['point_cloud'],
                        #                                            goal_pointcloud,], dim=-2)

                        obs_dict_input['point_cloud'] = torch.cat([obs_dict_input['point_cloud'],
                                                                   goal_pointcloud, np.zeros((obs_dict_input['point_cloud'].shape[0], 1875 - obs_dict_input['point_cloud'].shape[1], obs_dict_input['point_cloud'].shape[2]))], dim=-2)
                        
                        # print(obs_dict_input['point_cloud'].shape)
                        action_dict = policy.predict_action(obs_dict_input)
                        


                # device transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                # step env
                obs, reward, done, info = env.step(action)

                if done:
                    # saving rollout video
                    vid_frames = info['vid_frames'].squeeze(0)
                    vid_frames_list = [
                        Image.fromarray(frame) for frame in vid_frames
                    ]
                    vid_tag = "success" if info['is_success'] else "fail"
                    vid_save_path = os.path.join(output_save_dir, f'{id}_{vid_tag}.gif')
                    vid_frames_list[0].save(vid_save_path, save_all=True,
                                            append_images=vid_frames_list[self.vid_speed::self.vid_speed], 
                                            duration=33, loop=0)
                    # saving first frame
                    vid_frames_list[0].save(os.path.join(output_save_dir, f'{id}_{vid_tag}_first_frame.png'))
                    # saving last frame
                    vid_frames_list[-1].save(os.path.join(output_save_dir, f'{id}_{vid_tag}_last_frame.png'))
                    # saving pre-release frame
                    pre_release_frame = Image.fromarray(info['pre_release_frame'].squeeze(0))
                    pre_release_frame.save(os.path.join(output_save_dir, f'{id}_{vid_tag}_pre_release_frame.png'))


                    # if tax3d, also save the diffusion visualization
                    # grab the first frame, and then plot the time series of results
                    if self.tax3d:
                        color_key = info["color_key"].squeeze(0)
                        viewmat = info["viewmat"].squeeze(0)

                        # get img from vid_frames
                        # get results from action_dict
                        img = vid_frames[0]
                        results = policy.results_world
                        diffusion_frames = plot_diffusion(img, results, viewmat, color_key)
                        diffusion_save_path = os.path.join(output_save_dir, f'{id}_{vid_tag}_diffusion.gif')
                        diffusion_frames[0].save(diffusion_save_path, save_all=True,
                                                append_images=diffusion_frames[self.diffusion_gif_speed::self.diffusion_gif_speed], 
                                                duration=33, loop=0)
                        # saving last diffusion frame
                        diffusion_frames[-1].save(os.path.join(output_save_dir, f'{id}_{vid_tag}_diffusion_last_frame.png'))

                    if info['is_success']:
                        num_successes += 1
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