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
                    # action_dict = policy.predict_action(obs_dict_input)
                    print(policy.model_predict(obs_dict_input))        
                    
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
        pointCloudList = []
        for id in pbar:
            pbar.set_description(f"DEDO {self.task_name} Env ({num_successes})")
            # get rot, trans, deform params
            demo = dataset[id]
            rot = demo['rot']
            trans = demo['trans']
            deform_params = demo['deform_params'][()]

            obs = env.reset(
                rigid_rot=rot,
                rigid_trans=trans,
                deform_params=deform_params,
            )
            policy.reset()

            done = False
            np_obs_dict = dict(obs)

            # device transfer
            obs_dict = dict_apply(np_obs_dict,
                                    lambda x: torch.from_numpy(x).to(
                                        device=device))
            
            
            # run policy
            with torch.no_grad():
                obs_dict_input = {}  # flush unused keys
                obs_dict_input['pc_action'] = obs_dict['pc_action'].float()
                obs_dict_input['pc_anchor'] = obs_dict['pc_anchor'].float()
                obs_dict_input['seg'] = obs_dict['seg'].int()
                obs_dict_input['seg_anchor'] = obs_dict['seg_anchor'].int()
                goal_PC= policy.model_predict(obs_dict_input)
                pointCloudList.append(goal_PC)
            
                   
        del env
        return pointCloudList