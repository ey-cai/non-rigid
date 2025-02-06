import os
import numpy as np

import argparse
import gym
import torch
import pytorch3d.ops as torch3d_ops
from tqdm import tqdm
import zarr
from termcolor import cprint

from dedo.utils.args import get_args, args_postprocess
from PIL import Image

from non_rigid.utils.pointcloud_utils import downsample_pcd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='~/data/proccloth', help='directory to save data')
    parser.add_argument('--num_episodes', type=int, default=5, help='number of episodes to run')
    parser.add_argument('--action_num_points', type=int, default=625, help='number of points in action point cloud')
    parser.add_argument('--anchor_num_points', type=int, default=1024, help='number of points in anchor point cloud')
    parser.add_argument('--split', type=str, default='train', help='train/val/val_ood split')
    # Args for experiment settings.
    parser.add_argument('--random_cloth_geometry', action='store_true', help='randomize cloth geometry')
    parser.add_argument('--random_cloth_pose', action='store_true', help='randomize cloth pose')
    parser.add_argument('--random_anchor_geometry', action='store_true', help='randomize anchor geometry')
    parser.add_argument('--random_anchor_pose', action='store_true', help='randomize anchor pose')
    parser.add_argument('--num_anchors', type=str, default='1', help='number of anchors')
    parser.add_argument('--cloth_hole', type=str, default='single', help='number of holes in cloth')
    parser.add_argument('--tag', type=str, default='', help='additional tag for dataset description')
    parser.add_argument('--robot_env', action='store_true', help='use robot environment')
    # Args for demo generation.
    parser.add_argument('--debug_viz', action='store_true', help='debug mode (turns on GUI)')
    parser.add_argument('--vid_speed', type=int, default=3, help='speed of rollout video')
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
    split = args.split
    vid_speed = args.vid_speed
    random_cloth_geometry = args.random_cloth_geometry
    random_cloth_pose = args.random_cloth_pose
    random_anchor_geometry = args.random_anchor_geometry
    random_anchor_pose = args.random_anchor_pose
    num_anchors = int(args.num_anchors)
    cloth_hole = args.cloth_hole
    tag = args.tag
    use_robot_env = args.robot_env
    action_type = 'ee_position' if use_robot_env else 'position'

    if cloth_hole not in ['single', 'double']:
        raise ValueError(f'Invalid cloth hole configuration: {cloth_hole}')

    # parse num_anchors
    # anchor_num_list = list(map(int, num_anchors.split(',')))


    ##############################################
    # Creating experiment name and directories
    ##############################################
    cloth_geometry = 'multi' if random_cloth_geometry else 'single'
    cloth_pose = 'random' if random_cloth_pose else 'fixed'
    anchor_geometry = 'multi' if random_anchor_geometry else 'single'
    anchor_pose = 'random' if random_anchor_pose else 'fixed'
    num_holes = 1 if cloth_hole == 'single' else 2
    # check that num_holes divides num_episodes
    if num_episodes % (num_holes * num_anchors) != 0:
        raise ValueError(f'num_episodes ({num_episodes}) must be divisible by num_holes ({num_holes}) x num_anchors ({num_anchors})')

    # experiment name
    exp_name_dir = (
        f'cloth={cloth_geometry}-{cloth_pose} ' + \
        f'anchor={anchor_geometry}-{anchor_pose} ' + \
        f'hole={cloth_hole}{tag} ' + \
        f'robot={use_robot_env} ' + \
        # f'num_anchors={num_anchors}'
        f'num_anchors={num_anchors} ' + \
        'debug'
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
    dedo_args.env = 'HangProcClothRobot-v0' if use_robot_env else 'HangProcCloth-v0'
    dedo_args.tax3d = True
    dedo_args.rollout_vid = True
    dedo_args.viz = args.debug_viz
    dedo_args.max_episode_len = 300
    args_postprocess(dedo_args)

    # creating env
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
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    cloth_size_arrays = []
    total_count = 0
    total_successes = 0
    total_failures = 0
    counter = 0

    with tqdm(total=num_episodes) as pbar:
        num_success = 0
        while num_success < num_episodes:
            # ---------------- Deformable Data ----------------
            # randomizing cloth geometry
            if random_cloth_geometry:
                deform_params = {
                    'num_holes': num_holes,
                    'node_density': 25,
                }
            else:
                if num_holes == 1:
                    holes = [
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
                    'holes': holes,
                }

            # randomizing cloth pose
            if random_cloth_pose:
                deform_rotation, deform_translation = env.random_deform_transform()
                deform_rotation = deform_rotation.as_euler('xyz')
                deform_transform = {
                    'rotation': deform_rotation,
                    'translation': deform_translation,
                }
            else:
                deform_transform = {}
            
            deform_data = {
                'deform_transform': deform_transform,
                'deform_params': deform_params,
            }
            
            # ---------------- Rigid Data ----------------
            # randomizing anchor geometry
            if random_anchor_geometry:
                raise NotImplementedError("Need to implement random anchor geometry")
            else:
                rigid_params = {
                    'hanger_scale': 1.5,
                    'tallrod_scale': 1.5,
                }
            
            # randomizing anchor pose
            if random_anchor_pose:
                if split == 'val_ood':
                    rigid_rotation, rigid_translation = env.random_anchor_transform_ood(num_anchors)
                else:
                    rigid_rotation, rigid_translation = env.random_anchor_transform(num_anchors)
            else:
                raise ValueError("Only generating datasets for random anchor poses")

            rigid_data = {
                i: {
                    'rigid_transform': {
                        'rotation': rigid_rotation[i],
                        'translation': rigid_translation[i],
                    },
                    'rigid_params': rigid_params,
                }
                for i in range(num_anchors)
            }

            action_pcd_arrays_sub_list = []
            anchor_pcd_arrays_sub_list = []
            state_arrays_sub_list = []
            action_arrays_sub_list = []
            cloth_size_arrays_sub_list = []

            total_count_sub_list = []
            success_sub_list = []
            reward_sum_list = []

            tax3d_demo_list = []
            rollout_vid_list = []

            for hole in range(num_holes):
                for anchor in range(num_anchors):
                    # reset the environment
                    obs = env.reset(
                        deform_data=deform_data,
                        rigid_data=rigid_data,
                    )

                    # also downsampling tax3d demo - temporary debugging code for reference frame prediction
                    anchor_pc_tax3d = torch.tensor(obs['anchor_pcd'])
                    anchor_seg_tax3d = obs['anchor_seg']
                    anchor_pc_tax3d, anchor_indices = downsample_pcd(anchor_pc_tax3d.unsqueeze(0), 1024, "fps")
                    anchor_pc_tax3d = anchor_pc_tax3d.squeeze(0).numpy()
                    anchor_seg_tax3d = anchor_seg_tax3d[anchor_indices.squeeze(0).numpy()]


                    # initializing tax3d demo
                    tax3d_demo = {
                        'action_pc': obs['action_pcd'],
                        'action_seg': obs['action_seg'],
                        # 'anchor_pc': obs['anchor_pcd'],
                        # 'anchor_seg': obs['anchor_seg'],
                        'anchor_pc': anchor_pc_tax3d,
                        'anchor_seg': anchor_seg_tax3d,
                        'deform_data': deform_data,
                        'rigid_data': rigid_data,
                    }

                    # episode data
                    action_pcd_arrays_sub = []
                    anchor_pcd_arrays_sub = []
                    state_arrays_sub = []
                    action_arrays_sub = []
                    cloth_size_arrays_sub = []

                    success = False
                    total_count_sub = 0
                    reward_sum = 0

                    # rollout the policy for this hole
                    while True:
                        # get action
                        action = env.pseudo_expert_action(anchor, hole)
                        total_count_sub += 1

                        # downsample point clouds for demos (not tax3d demos)
                        obs_action_pcd = obs['action_pcd']
                        obs_anchor_pcd = obs['anchor_pcd']
                        gripper_state = obs['gripper_state']

                        cloth_size = obs_action_pcd.shape[0]
                        if cloth_size < action_num_points:
                            # pad
                            pad = np.zeros((action_num_points - cloth_size, 3))
                            obs_action_pcd = np.concatenate([obs_action_pcd, pad], axis=0)

                        if obs_anchor_pcd.shape[0] > anchor_num_points:
                            obs_anchor_pcd = downsample_with_fps(obs_anchor_pcd, anchor_num_points)

                        # update episode data
                        action_pcd_arrays_sub.append(obs_action_pcd)
                        anchor_pcd_arrays_sub.append(obs_anchor_pcd)
                        state_arrays_sub.append(gripper_state)
                        action_arrays_sub.append(action)
                        cloth_size_arrays_sub.append([cloth_size])

                        # step environment
                        obs, reward, done, info = env.step(action, action_type=action_type)
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
                        cloth_size_arrays_sub_list.extend(cloth_size_arrays_sub)
                        reward_sum_list.append(reward_sum)

                        # updating successful tax3d demo
                        tax3d_demo["flow"] = obs["action_pcd"] - tax3d_demo["action_pc"]
                        tax3d_demo_list.append(tax3d_demo)

                        # updating successful rollout video
                        vid_frames = [
                            Image.fromarray(frame) for frame in info["vid_frames"]
                        ]
                        rollout_vid_list.append(vid_frames)
                        total_successes += 1
                    else:
                        print("Failed.")
                        total_failures += 1
                        break

            # in order to save data, policy must be successful on all holes
            if np.sum(success_sub_list) == (num_holes * num_anchors):
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
                cloth_size_arrays.extend(cloth_size_arrays_sub_list)

                len_episode = len(action_pcd_arrays_sub)
                ground_truth_subarray = np.tile(np.expand_dims(obs["action_pcd"], axis=0), (len_episode, 1, 1))

                if obs_action_pcd.shape[0] == 625 and random_cloth_geometry:
                    # print('Padding ground truth with zeroes')
                    pad_gt = np.tile(ground_truth_subarray[:, 0].reshape(-1, 1, 3), (1, action_num_points - ground_truth_subarray.shape[1], 1))
                    ground_truth_subarray = np.concatenate([ground_truth_subarray, pad_gt], axis=1)

                ground_truth_arrays.extend(ground_truth_subarray)

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
                num_success += (num_holes * num_anchors)
                pbar.update(num_holes * num_anchors)
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
    action_arrays = action_arrays.reshape(action_arrays.shape[0], -1) # Flatten
    episode_ends_arrays = np.array(episode_ends_arrays)
    ground_truth_arrays = np.stack(ground_truth_arrays, axis=0)
    cloth_size_arrays = np.stack(cloth_size_arrays, axis=0)

    episode_length = 301
    velocity_arrays = np.zeros((len(state_arrays), 6))    
    velocity_arrays[:, 0:3] = state_arrays[:, 3:6]
    velocity_arrays[:, 3:6] = state_arrays[:, 9:12]

    # as an additional step, create point clouds that combine action and anchor
    point_cloud_arrays = np.concatenate([action_pcd_arrays, anchor_pcd_arrays], axis=1)


    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    # for now, use chunk size of 100
    action_pcd_chunk_size = (100, action_pcd_arrays.shape[1], action_pcd_arrays.shape[2])
    anchor_pcd_chunk_size = (100, anchor_pcd_arrays.shape[1], anchor_pcd_arrays.shape[2])
    state_chunk_size = (100, state_arrays.shape[1])
    action_chunk_size = (100, action_arrays.shape[1])
    cloth_chunk_size = (100, action_arrays.shape[1])
    ground_truth_chunk_size = (100, ground_truth_arrays.shape[1], ground_truth_arrays.shape[2])
    velocity_chunk_size = (100, velocity_arrays.shape[1])

    zarr_data.create_dataset('action_pcd', data=action_pcd_arrays, chunks=action_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('anchor_pcd', data=anchor_pcd_arrays, chunks=anchor_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('cloth', data=cloth_size_arrays, chunks=cloth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('ground_truth', data=ground_truth_arrays, chunks=ground_truth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('velocity', data=velocity_arrays, chunks=velocity_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    
    cprint(f'action point cloud shape: {action_pcd_arrays.shape}, range: [{np.min(action_pcd_arrays)}, {np.max(action_pcd_arrays)}]', 'green')
    cprint(f'anchor point cloud shape: {anchor_pcd_arrays.shape}, range: [{np.min(anchor_pcd_arrays)}, {np.max(anchor_pcd_arrays)}]', 'green')
    cprint(f'point cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'ground truth shape: {ground_truth_arrays.shape}, range: [{np.min(ground_truth_arrays)}, {np.max(ground_truth_arrays)}]', 'green')
    cprint(f'cloth shape: {cloth_size_arrays.shape}, size: [{np.min(cloth_size_arrays)}, {np.max(cloth_size_arrays)}]', 'green')
    cprint(f'velocity shape: {velocity_arrays.shape}, range: [{np.min(velocity_arrays)}, {np.max(velocity_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')
    cprint(f'Successes: {total_successes}, Failures: {total_failures}', 'green')

    # clean up
    del action_pcd_arrays, anchor_pcd_arrays, action_arrays, episode_ends_arrays, ground_truth_arrays, velocity_arrays, cloth_size_arrays
    del zarr_root, zarr_data, zarr_meta
    del env