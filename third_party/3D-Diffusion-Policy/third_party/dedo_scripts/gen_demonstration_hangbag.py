import os
import numpy as np

import argparse
import gym
import torch
import pytorch3d.ops as torch3d_ops
from tqdm import tqdm
import zarr
from termcolor import cprint

from dedo.utils.args import get_args, args_postprocess, CAM_CONFIG_DIR
from dedo.envs import DeformEnvTAX3D

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env_name', type=str, default='single_cloth', help='environment to run')
    parser.add_argument('--root_dir', type=str, default='data', help='directory to save data')
    parser.add_argument('--num_episodes', type=int, default=5, help='number of episodes to run')
    parser.add_argument('--action_num_points', type=int, default=512, help='number of points in action point cloud')
    parser.add_argument('--anchor_num_points', type=int, default=512, help='number of points in anchor point cloud')
    # parser.add_argument('--anchor_config', type=str, default='fixed', help='anchor configuration')
    parser.add_argument('--split', type=str, default='train', help='train/val/val_ood split')
    # Args for experiment settings.
    parser.add_argument('--random_cloth_geometry', action='store_true', help='randomize cloth geometry')
    parser.add_argument('--random_cloth_pose', action='store_true', help='randomize cloth pose')
    parser.add_argument('--random_anchor_geometry', action='store_true', help='randomize anchor geometry')
    parser.add_argument('--random_anchor_pose', action='store_true', help='randomize anchor pose')
    parser.add_argument('--cloth_hole', type=str, default='single', help='number of holes in cloth')
    # Args for demo generation.
    parser.add_argument('--vid_speed', type=int, default=3, help='speed of rollout video')
    parser.add_argument('--save_demos', action='store_true', help='generate DEDO demos')
    parser.add_argument('--save_tax3d', action='store_true', help='save TAX3D demos')
    parser.add_argument('--save_rollout_vids', action='store_true', help='save rollout videos')
    args, _ = parser.parse_known_args()
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


if __name__ == '__main__':
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

    save_demos = args.save_demos
    save_tax3d = args.save_tax3d
    save_rollout_vids = args.save_rollout_vids

    random_cloth_geometry = args.random_cloth_geometry
    random_cloth_pose = args.random_cloth_pose
    random_anchor_geometry = args.random_anchor_geometry
    random_anchor_pose = args.random_anchor_pose
    cloth_hole = args.cloth_hole

    if cloth_hole not in ['single', 'double']:
        raise ValueError(f'Invalid cloth hole configuration: {cloth_hole}')


    ##############################################
    # Creating experiment name and save directory
    ##############################################
    cloth_geometry = 'multi' if random_cloth_geometry else 'single'
    cloth_pose = 'random' if random_cloth_pose else 'fixed'
    anchor_geometry = 'multi' if random_anchor_geometry else 'single'
    anchor_pose = 'random' if random_anchor_pose else 'fixed'

    exp_name_dir = (
        f'cloth={cloth_geometry}-{cloth_pose} ' + \
        f'anchor={anchor_geometry}-{anchor_pose} ' + \
        f'hole={cloth_hole}'
    )
    # TODO: get rid of the env_name arg
    
    # save_dir = os.path.join(args.root_dir, 'dedo_'+args.env_name+'_'+anchor_config+'_expert_'+split+'.zarr')
    # tax3d_save_dir = os.path.join(args.root_dir, 'tax3d/multi_cloth_1/'+split+'_1/')

    save_dir = os.path.join(args.root_dir, exp_name_dir, f'{split}.zarr')
    tax3d_save_dir = os.path.join(args.root_dir, exp_name_dir, f'{split}_tax3d/')
    rollout_vids_save_dir = os.path.join(args.root_dir, exp_name_dir, f'{split}_rollout_vids/')
    # TODO: eventually, this demo generation should be more general; handle unified experiment names, and train/val/val_ood splits

    # creating directory for rollouts
    if save_demos:
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

    # creating directory for tax3d data
    if save_tax3d:
        os.makedirs(tax3d_save_dir, exist_ok=True)

    # creating directory for rollout videos
    if save_rollout_vids:
        os.makedirs(rollout_vids_save_dir, exist_ok=True)

    ###############################
    # parse DEDO demo args and create DEDO env
    ###############################
    dedo_args = get_args()
    # dedo_args.env = 'HangProcCloth-v0'
    dedo_args.env = 'HangBag-v0'
    dedo_args.tax3d = True
    dedo_args.rollout_vid = True
    dedo_args.pcd = True
    dedo_args.logdir = 'rendered'
    dedo_args.cam_config_path = f"{CAM_CONFIG_DIR}/camview_0.json"
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
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    total_count = 0

    with tqdm(total=num_episodes) as pbar:
        num_success = 0
        while num_success < num_episodes:
            # randomizing cloth geometry
            if random_cloth_geometry:
                # TODO: randomly pick cloth geometry
                raise NotImplementedError("Need to implement random cloth geometry")
            else:
                deform_params = {
                    'id0': 1,
                    'id1': 34,
                }

            # randomizing cloth pose
            if random_cloth_pose:
                raise NotImplementedError("Need to implement random cloth pose")
            
            # randomizing cloth geometry
            if random_anchor_geometry:
                raise NotImplementedError("Need to implement random anchor geometry")
            else:
                anchor_params = {}
            
            # randomizing anchor pose
            if random_anchor_pose:
                if split == 'val_ood':
                    rigid_rot, rigid_trans = env.random_anchor_transform_ood()
                else:
                    rigid_rot, rigid_trans = env.random_anchor_transform()
                rigid_rot = rigid_rot.as_euler('xyz')
            else:
                raise ValueError("Only generating datasets for random anchor poses")


            obs = env.reset(
                rigid_rot=rigid_rot,
                rigid_trans=rigid_trans,
                deform_params=deform_params,
                # deform_params={'id0': 1, 'id1': 34},
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



            # episode data
            action_pcd_arrays_sub = []
            anchor_pcd_arrays_sub = []
            state_arrays_sub = []
            action_arrays_sub = []

            success = False
            total_count_sub = 0
            reward_sum = 0


            # get action from pseudo-expert
            random_hole = np.random.randint(num_holes)
            action = env.pseudo_expert_action(random_hole, rigid_rot=rigid_rot, rigid_trans=rigid_trans)
            while True:
                action = env.pseudo_expert_action(random_hole, rigid_rot=rigid_rot, rigid_trans=rigid_trans)
                # increment total
                total_count_sub += 1

                # update episode data

                # expected keys from obs:
                # 'action_pcd', 'anchor_pcd'
                obs_action_pcd = obs['action_pcd']
                obs_anchor_pcd = obs['anchor_pcd']
                gripper_state = obs['gripper_state']

                # downsampling just for demos (not tax3d)
                if obs_action_pcd.shape[0] > 512:
                    obs_action_pcd = downsample_with_fps(obs_action_pcd, action_num_points)
                if obs_anchor_pcd.shape[0] > 512:
                    obs_anchor_pcd = downsample_with_fps(obs_anchor_pcd, anchor_num_points)

                action_pcd_arrays_sub.append(obs_action_pcd)
                anchor_pcd_arrays_sub.append(obs_anchor_pcd)
                state_arrays_sub.append(gripper_state)
                action_arrays_sub.append(action)
                # TODO: do we need to keep track of rewards?

                # obs, reward, done, info = env.step(action, action_type='velocity')
                obs, reward, done, info = env.step(action, action_type='position')
                reward_sum += reward
                if done:
                    success = info['is_success']
                    break

            if success:
                print('Success!')
                total_count += total_count_sub
                episode_ends_arrays.append(total_count)
                reward_list.append(reward_sum)
                success_list.append(int(success))

                action_pcd_arrays.extend(action_pcd_arrays_sub)
                anchor_pcd_arrays.extend(anchor_pcd_arrays_sub)
                state_arrays.extend(state_arrays_sub)
                action_arrays.extend(action_arrays_sub)


                # TODO: this is incredibly hacky, but save demo id??
                # updating tax3d demo with flow, and saving
                tax3d_demo['flow'] = obs["action_pcd"] - tax3d_demo['action_pc']
                if save_tax3d:
                    np.savez(
                        os.path.join(tax3d_save_dir, f'demo_{num_success}.npz'),
                        **tax3d_demo
                    )

                # saving rollout video
                if save_rollout_vids:
                    vid_frames = info['vid_frames']
                    vid_frames_list = [
                        Image.fromarray(frame) for frame in vid_frames
                    ]
                    vid_frames_list[0].save(
                        os.path.join(rollout_vids_save_dir, f'demo_{num_success}.gif'),
                        save_all=True,
                        append_images=vid_frames_list[vid_speed::vid_speed],
                        duration=33,
                        loop=0,
                    )

                num_success += 1
                pbar.update(1)
            else:
                print('Failed episode, retrying...')


    ###############################
    # save data
    ###############################
    if not save_demos:
        cprint('Not saving demos', 'red')
        exit(0)

    # create zarr file
    zarr_root = zarr.group(save_dir, overwrite=True)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save point cloud and action arrays into data, and episode ends arrays into meta
    action_pcd_arrays = np.stack(action_pcd_arrays, axis=0)
    anchor_pcd_arrays = np.stack(anchor_pcd_arrays, axis=0)
    state_arrays = np.stack(state_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    # as an additional step, create point clouds that combine action and anchor
    point_cloud_arrays = np.concatenate([action_pcd_arrays, anchor_pcd_arrays], axis=1)


    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    # for now, use chunk size of 100
    action_pcd_chunk_size = (100, action_pcd_arrays.shape[1], action_pcd_arrays.shape[2])
    anchor_pcd_chunk_size = (100, anchor_pcd_arrays.shape[1], anchor_pcd_arrays.shape[2])
    state_chunk_size = (100, state_arrays.shape[1])
    action_chunk_size = (100, action_arrays.shape[1])

    zarr_data.create_dataset('action_pcd', data=action_pcd_arrays, chunks=action_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('anchor_pcd', data=anchor_pcd_arrays, chunks=anchor_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    
    cprint(f'action point cloud shape: {action_pcd_arrays.shape}, range: [{np.min(action_pcd_arrays)}, {np.max(action_pcd_arrays)}]', 'green')
    cprint(f'anchor point cloud shape: {anchor_pcd_arrays.shape}, range: [{np.min(anchor_pcd_arrays)}, {np.max(anchor_pcd_arrays)}]', 'green')
    cprint(f'point cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')

    # clean up
    del action_pcd_arrays, anchor_pcd_arrays, action_arrays, episode_ends_arrays
    del zarr_root, zarr_data, zarr_meta
    del env