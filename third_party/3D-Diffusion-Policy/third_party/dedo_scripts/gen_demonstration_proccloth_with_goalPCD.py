
import os
import numpy as np
import sys
import pathlib
import argparse
import hydra
import gym
import torch
import pytorch3d.ops as torch3d_ops
from tqdm import tqdm
import zarr
from termcolor import cprint
from omegaconf import OmegaConf
from dedo.utils.args import get_args, args_postprocess
from dedo.envs import DeformEnvTAX3D
from trainGoalPC import EvalTAX3DWorkspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from PIL import Image


OmegaConf.register_new_resolver("eval", eval, replace=True)
    
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env_name', type=str, default='single_cloth', help='environment to run')
    parser.add_argument('--root_dir', type=str, default='data', help='directory to save data')
    parser.add_argument('--num_episodes', type=int, default=64, help='number of episodes to run')
    parser.add_argument('--action_num_points', type=int, default=580, help='number of points in action point cloud')
    parser.add_argument('--anchor_num_points', type=int, default=580, help='number of points in anchor point cloud')
    # parser.add_argument('--anchor_config', type=str, default='fixed', help='anchor configuration')
    parser.add_argument('--split', type=str, default='train', help='train/val/val_ood split')
    # Args for experiment settings.
    parser.add_argument('--random_cloth_geometry', action='store_true', help='randomize cloth geometry')
    parser.add_argument('--random_cloth_pose', action='store_true', help='randomize cloth pose')
    parser.add_argument('--random_anchor_geometry', action='store_true', help='randomize anchor geometry')
    parser.add_argument('--random_anchor_pose', action='store_true', help='randomize anchor pose')
    parser.add_argument('--cloth_hole', type=str, default='single', help='number of holes in cloth')
    parser.add_argument('--tag', type=str, default='', help='additional tag for dataset description')
    # Args for demo generation.
    parser.add_argument('--vid_speed', type=int, default=3, help='speed of rollout video')
    #parser.add_argument('--save_demos', action='store_true', help='generate DEDO demos')
    #parser.add_argument('--save_tax3d', action='store_true', help='save TAX3D demos')
    #parser.add_argument('--save_rollout_vids', action='store_true', help='save rollout videos')
    args, _ = parser.parse_known_args()
    # TODO: maybe add task as an argument, so that all the demonstration generation can be consolidated into one script

    return args

def downsample_with_fps(points: np.ndarray, num_points: int = 512):
    # fast point cloud sampling using torch3d
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    num_points = torch.tensor([num_points]).cuda()
    # remember to only use coord to sample
    _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=num_points)
    points = points.squeeze(0).cpu().numpy()
    points = points[sampled_indices.squeeze(0).cpu().numpy()]
    return points

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.parent.joinpath(
        '3D-Diffusion-Policy', 'diffusion_policy_3d', 'config'))
)
def main(cfg):
    ###############################
    # parse DP3 demo args
    ###############################
    args = parse_args()
    num_episodes = args.num_episodes
    action_num_points = args.action_num_points
    anchor_num_points = args.anchor_num_points
    # anchor_config = args.anchor_config
    split = args.split
    vid_speed = args.vid_speed

    #save_demos = args.save_demos
    #save_tax3d = args.save_tax3d
    #save_rollout_vids = args.save_rollout_vids

    random_cloth_geometry = args.random_cloth_geometry # set it to true later
    random_cloth_pose = args.random_cloth_pose
    random_anchor_geometry = args.random_anchor_geometry
    # random_anchor_pose = args.random_anchor_pose
    random_anchor_pose = True
    cloth_hole = args.cloth_hole
    tag = args.tag

    if cloth_hole not in ['single', 'double']:
        raise ValueError(f'Invalid cloth hole configuration: {cloth_hole}')


    ##############################################
    # Creating experiment name and directories
    ##############################################
    cloth_geometry = 'multi' if random_cloth_geometry else 'single'
    cloth_pose = 'random' if random_cloth_pose else 'fixed'
    anchor_geometry = 'multi' if random_anchor_geometry else 'single'
    anchor_pose = 'random' if random_anchor_pose else 'fixed'
    # experiment name
    exp_name_dir = (
        f'cloth={cloth_geometry}-{cloth_pose} ' + \
        f'anchor={anchor_geometry}-{anchor_pose} ' + \
        f'hole={cloth_hole}{tag}'
    )

    # creating directories
    args.root_dir = os.path.expanduser(args.root_dir)
    save_dir = os.path.join(args.root_dir, exp_name_dir, f'{split}.zarr')
    tax3d_save_dir = os.path.join(args.root_dir, exp_name_dir, f'{split}_tax3d/')
    rollout_vids_save_dir = os.path.join(args.root_dir, exp_name_dir, f'{split}_rollout_vids/')
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        user_input = input()
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
        else:
            cprint('Exiting', 'red')
            exit(0)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tax3d_save_dir, exist_ok=True)
    os.makedirs(rollout_vids_save_dir, exist_ok=True)

    ###############################
    # parse DEDO demo args and create DEDO env
    ###############################
    dedo_args = get_args()
    dedo_args.env = 'HangProcCloth-v0'
    dedo_args.tax3d = True
    dedo_args.rollout_vid = True
    dedo_args.pcd = True
    dedo_args.logdir = 'rendered'
    dedo_args.cam_config_path = '/home/ktsim/Projects/non-rigid/third_party/dedo/dedo/utils/cam_configs/camview_0.json'
    dedo_args.viz = True
    dedo_args.max_episode_len = 300
    args_postprocess(dedo_args)

    # TODO: based on env name, there should be some logic here handling resetting of the environment
    # potentially have to load env configuration from TAX3D datasets
    num_holes = 1 if cloth_hole == 'single' else 2
    # deform_params = { # for single-cloth datasets
    #     'num_holes': num_holes,
    #     'node_density': 25,
    #     'w': 1.0,
    #     'h': 1.0,
    #     'holes': [{'x0': 8, 'x1': 16, 'y0': 9, 'y1': 13}]
    # }
    # check that num_holes divides num_episodes
    if num_episodes % num_holes != 0:
        raise ValueError(f'num_episodes ({num_episodes}) must be divisible by num_holes ({num_holes})')

    kwargs = {'args': dedo_args}
    env = gym.make(dedo_args.env, **kwargs)


    # settings seed based on split
    if split == 'train':
        seed = 0
    elif split == 'val':
        seed = 10
    elif split == 'val_ood':
        seed = 20
    env.seed(seed)

    ###############################
    # run episodes
    ###############################
    success_list = []
    reward_list = []

    action_pcd_arrays = []
    anchor_pcd_arrays = []
    ground_truth_arrays = []
    tax3d_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    total_count = 0

    # Configure Tax3D inference
    policy = EvalTAX3DWorkspace(cfg).model
    policy.eval()
    policy.cuda()

    with tqdm(total=num_episodes) as pbar:
        num_success = 0
        while num_success < num_episodes:
            # randomizing cloth geometry
            if random_cloth_geometry:
                deform_params = {
                    'num_holes': num_holes,
                    'node_density': 25,
                    #'w': 1.0,
                    #'h': 1.0,
                }
            else:
                if num_holes == 1:
                    holes = [
                        # {'x0': 9, 'x1': 13, 'y0': 11, 'y1': 14}
                        {'x0': 8, 'x1': 16, 'y0': 9, 'y1': 13}
                    ]
                else:
                    holes = [
                        {'x0': 8, 'x1': 15, 'y0': 7, 'y1': 10}, 
                        {'x0': 11, 'x1': 17, 'y0': 16, 'y1': 19}
                    ]
                deform_params = {
                    'num_holes': num_holes,
                    'node_density': 25,
                    'w': 1.0,
                    'h': 1.0,
                    # 'holes': [{'x0': 8, 'x1': 16, 'y0': 9, 'y1': 13}]
                    'holes': holes,
                }

            # randomizing cloth pose
            if random_cloth_pose:
                raise NotImplementedError("Need to implement random cloth pose")
            
            # randomizing cloth geometry
            if random_anchor_geometry:
                raise NotImplementedError("Need to implement random anchor geometry")
            else:
                anchor_params = {
                    'hanger_scale': 1.0,
                    'tallrod_scale': 1.0,
                }
            
            # randomizing anchor pose
            if random_anchor_pose:
                if split == 'val_ood':
                    rigid_rot, rigid_trans = env.random_anchor_transform_ood()
                else:
                    rigid_rot, rigid_trans = env.random_anchor_transform()
                rigid_rot = rigid_rot.as_euler('xyz')
            else:
                raise ValueError("Only generating datasets for random anchor poses")

            action_pcd_arrays_sub_list = []
            anchor_pcd_arrays_sub_list = []
            state_arrays_sub_list = []
            action_arrays_sub_list = []
        
            total_count_sub_list = []
            success_sub_list = []
            reward_sum_list = []

            tax3d_demo_list = []
            rollout_vid_list = []

            for hole in range(num_holes):
                # reset the environment
                obs = env.reset(
                    rigid_rot=rigid_rot,
                    rigid_trans=rigid_trans,
                    deform_params=deform_params,
                    anchor_params=anchor_params,
                )

                # initializing tax3d demo
                tax3d_demo = {
                    'action_pc': obs['action_pcd'],
                    'action_seg': np.ones(obs['action_pcd'].shape[0]),
                    'anchor_pc': obs['anchor_pcd'],
                    'anchor_seg': np.ones(obs['anchor_pcd'].shape[0]),
                    'speed_factor': 1.0, # this is legacy?
                    'rot': rigid_rot,
                    'trans': rigid_trans,
                    'deform_params': deform_params,
                    'anchors': env.anchors, # this is legacy?
                }

                obs_dict = {}  # flush unused keys
                obs_dict['pc_action'] = np.expand_dims(tax3d_demo['action_pc'], axis=0)
                obs_dict['pc_anchor'] = np.expand_dims(tax3d_demo['anchor_pc'], axis=0)
                obs_dict['seg'] = np.expand_dims(tax3d_demo['action_seg'], axis=0)
                obs_dict['seg_anchor'] = np.expand_dims(tax3d_demo['anchor_seg'], axis=0)

                obs_dict = dict_apply(obs_dict,
                                    lambda x: torch.from_numpy(x).to(device=policy.device))
                
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['pc_action'] = obs_dict['pc_action'].float()
                    obs_dict_input['pc_anchor'] = obs_dict['pc_anchor'].float()
                    obs_dict_input['seg'] = obs_dict['seg'].int()
                    obs_dict_input['seg_anchor'] = obs_dict['seg_anchor'].int()
                    # breakpoint()
                    tax3D_goalPCD = policy.model_predict(obs_dict_input).cpu().data.numpy()
                

                # episode data
                action_pcd_arrays_sub = []
                anchor_pcd_arrays_sub = []
                state_arrays_sub = []
                action_arrays_sub = []
              
                success = False
                total_count_sub = 0
                reward_sum = 0

                # rollout the policy for this hole
                while True:
                    # get action
                    action = env.pseudo_expert_force(hole, rigid_rot=rigid_rot, rigid_trans=rigid_trans)
                    total_count_sub += 1

                    # downsample point clouds for demos (not tax3d demos)
                    obs_action_pcd = obs['action_pcd']
                    obs_anchor_pcd = obs['anchor_pcd']
                    gripper_state = obs['gripper_state']

                    # if obs_action_pcd.shape[0] > 512:
                    #     obs_action_pcd = downsample_with_fps(obs_action_pcd, action_num_points)
                    if obs_anchor_pcd.shape[0] > 512:
                        obs_anchor_pcd = downsample_with_fps(obs_anchor_pcd, anchor_num_points)

                    # update episode data
                    action_pcd_arrays_sub.append(obs_action_pcd)
                    anchor_pcd_arrays_sub.append(obs_anchor_pcd)
                    state_arrays_sub.append(gripper_state)
                    action_arrays_sub.append(action)

                    # step environment
                    obs, reward, done, info = env.step(action, action_type='force')
           
                    reward_sum += reward
                    if done:
                        success = info['is_success']
                        success_sub_list.append(int(success))
                        break

                if success:
                    print("Success!")
                    # updating successful demo
                    action_pcd_arrays_sub_list.extend(action_pcd_arrays_sub)
                    anchor_pcd_arrays_sub_list.extend(anchor_pcd_arrays_sub)
                    state_arrays_sub_list.extend(state_arrays_sub)
                    action_arrays_sub_list.extend(action_arrays_sub)
                    total_count_sub_list.append(total_count_sub)
                    reward_sum_list.append(reward_sum)
                    # force_arrays_sub_list.extend(forces_arrays_sub)

                    # updating successful tax3d demo
                    tax3d_demo["flow"] = obs["action_pcd"] - tax3d_demo["action_pc"]
                    tax3d_demo_list.append(tax3d_demo)

                                       # updating successful rollout video
                    vid_frames = [
                        Image.fromarray(frame) for frame in info["vid_frames"]
                    ]
                    rollout_vid_list.append(vid_frames)
                else:
                    print("Failed.")
                    break

            # in order to save data, policy must be successful on all holes
            if np.sum(success_sub_list) == num_holes:
                # update episode ends, successes, and rewards
                episode_ends_arrays.extend(np.cumsum(total_count_sub_list) + total_count)
                total_count += np.sum(total_count_sub_list)
                # update success and reward lists
                success_list.extend(success_sub_list)
                reward_list.extend(reward_sum_list)

                # update action, anchor, state arrays
                action_pcd_arrays.extend(action_pcd_arrays_sub_list)
                anchor_pcd_arrays.extend(anchor_pcd_arrays_sub_list)
                state_arrays.extend(state_arrays_sub_list)
                action_arrays.extend(action_arrays_sub_list)

                # also adding goal pcd from ground truth and tax3d
                len_episode = len(action_pcd_arrays_sub)
                ground_truth_subarray = np.tile(np.expand_dims(obs["action_pcd"], axis=0), (len_episode, 1, 1))
                ground_truth_arrays.extend(ground_truth_subarray)

                tax3d_subarray = np.tile(tax3D_goalPCD, (len_episode, 1, 1))
                tax3d_arrays.extend(tax3d_subarray)

                # save tax3d demos and rollout vids
                for i in range(len(tax3d_demo_list)):
                    tax3d_demo = tax3d_demo_list[i]
                    vid_frames = rollout_vid_list[i]

                    np.savez(
                        os.path.join(tax3d_save_dir, f'demo_{num_success + i}.npz'),
                        **tax3d_demo
                    )
                    vid_frames[0].save(
                        os.path.join(rollout_vids_save_dir, f'demo_{num_success + i}.gif'),
                        save_all=True,
                        append_images=vid_frames[vid_speed::vid_speed],
                        duration=33,
                        loop=0,
                    )
                num_success += num_holes
                pbar.update(num_holes)
            else:
                print("Failed on at least one hole, retrying...")

    ###############################
    # save data
    ###############################
    # create zarr file
    zarr_root = zarr.group(save_dir, overwrite=True)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save point cloud and action arrays into data, and episode ends arrays into meta
    action_pcd_arrays = np.stack(action_pcd_arrays, axis=0)
    anchor_pcd_arrays = np.stack(anchor_pcd_arrays, axis=0)
    state_arrays = np.stack(state_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    ground_truth_arrays = np.stack(ground_truth_arrays, axis=0)
    tax3d_arrays = np.stack(tax3d_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    # as an additional step, create point clouds that combine action and anchor
    point_cloud_arrays = np.concatenate([action_pcd_arrays, anchor_pcd_arrays, ground_truth_arrays], axis=1)


    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    # for now, use chunk size of 100
    action_pcd_chunk_size = (100, action_pcd_arrays.shape[1], action_pcd_arrays.shape[2])
    anchor_pcd_chunk_size = (100, anchor_pcd_arrays.shape[1], anchor_pcd_arrays.shape[2])
    state_chunk_size = (100, state_arrays.shape[1])
    action_chunk_size = (100, action_arrays.shape[1])
    ground_truth_chunk_size = (100, ground_truth_arrays.shape[1], ground_truth_arrays.shape[2])
    tax3d_chunk_size = (100, tax3d_arrays.shape[1], tax3d_arrays.shape[2])
    
    zarr_data.create_dataset('action_pcd', data=action_pcd_arrays, chunks=action_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('anchor_pcd', data=anchor_pcd_arrays, chunks=anchor_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('ground_truth', data=ground_truth_arrays, chunks=ground_truth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('tax3d', data=tax3d_arrays, chunks=tax3d_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    
    cprint(f'action point cloud shape: {action_pcd_arrays.shape}, range: [{np.min(action_pcd_arrays)}, {np.max(action_pcd_arrays)}]', 'green')
    cprint(f'anchor point cloud shape: {anchor_pcd_arrays.shape}, range: [{np.min(anchor_pcd_arrays)}, {np.max(anchor_pcd_arrays)}]', 'green')
    cprint(f'point cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'goal truth shape: {ground_truth_arrays.shape}, range: [{np.min(ground_truth_arrays)}, {np.max(ground_truth_arrays)}]', 'green')
    cprint(f'tax3d cloud shape: {tax3d_arrays.shape}, range: [{np.min(tax3d_arrays)}, {np.max(tax3d_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')

    # clean up
    del action_pcd_arrays, anchor_pcd_arrays, ground_truth_arrays, action_arrays, episode_ends_arrays
    del zarr_root, zarr_data, zarr_meta
    del env


if __name__ == "__main__":
    main()