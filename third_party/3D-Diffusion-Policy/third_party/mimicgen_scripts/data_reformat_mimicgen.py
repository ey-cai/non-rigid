"""
Helper script to report dataset information. By default, will print trajectory length statistics,
the maximum and minimum action element in the dataset, filter keys present, environment
metadata, and the structure of the first demonstration. If --verbose is passed, it will
report the exact demo keys under each filter key, and the structure of all demonstrations
(not just the first one).

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, report statistics on the subset of trajectories
        in the file that correspond to this filter key

    verbose (bool): if flag is provided, print more details, like the structure of all
        demonstrations (not just the first one)

Example usage:

    # run script on example hdf5 packaged with repository
    python get_dataset_info.py --dataset ../../tests/assets/test.hdf5

    # run script only on validation data
    python get_dataset_info.py --dataset ../../tests/assets/test.hdf5 --filter_key valid
"""
import h5py
import json
import argparse
import numpy as np
import os
import zarr
from termcolor import cprint

import mimicgen.utils.file_utils as MG_FileUtils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        help="path to hdf5 dataset folder",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) if provided, report statistics on the subset of trajectories \
            in the file that correspond to this filter key",
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="verbose output",
    )
    args = parser.parse_args()

    # merge all demos into one file
    # MG_FileUtils.merge_all_hdf5(
    #     folder=args.dataset_folder,
    #     new_hdf5_path=args.dataset,
    #     delete_folder=False,
    # )

    # extract demonstration list from file
    filter_key = args.filter_key
    all_filter_keys = None
    f = h5py.File(args.dataset, "r")
    if filter_key is not None:
        # use the demonstrations from the filter key instead
        print("NOTE: using filter key {}".format(filter_key))
        demos = sorted([elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])])
    else:
        # use all demonstrations
        demos = sorted(list(f["data"].keys()))

        # extract filter key information
        if "mask" in f:
            all_filter_keys = {}
            for fk in f["mask"]:
                fk_demos = sorted([elem.decode("utf-8") for elem in np.array(f["mask/{}".format(fk)])])
                all_filter_keys[fk] = fk_demos

    # put demonstration list in increasing episode order
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # extract length of each trajectory in the file
    traj_lengths = []
    action_min = np.inf
    action_max = -np.inf
    for ep in demos:
        traj_lengths.append(f["data/{}/actions".format(ep)].shape[0])
        action_min = min(action_min, np.min(f["data/{}/actions".format(ep)][()]))
        action_max = max(action_max, np.max(f["data/{}/actions".format(ep)][()]))
    traj_lengths = np.array(traj_lengths)

    # report statistics on the data
    print("")
    print("total transitions: {}".format(np.sum(traj_lengths)))
    print("total trajectories: {}".format(traj_lengths.shape[0]))
    print("traj length mean: {}".format(np.mean(traj_lengths)))
    print("traj length std: {}".format(np.std(traj_lengths)))
    print("traj length min: {}".format(np.min(traj_lengths)))
    print("traj length max: {}".format(np.max(traj_lengths)))
    print("action min: {}".format(action_min))
    print("action max: {}".format(action_max))
    print("")
    print("==== Filter Keys ====")
    if all_filter_keys is not None:
        for fk in all_filter_keys:
            print("filter key {} with {} demos".format(fk, len(all_filter_keys[fk])))
    else:
        print("no filter keys")
    print("")
    if args.verbose:
        if all_filter_keys is not None:
            print("==== Filter Key Contents ====")
            for fk in all_filter_keys:
                print("filter_key {} with {} demos: {}".format(fk, len(all_filter_keys[fk]), all_filter_keys[fk]))
        print("")
    env_meta = json.loads(f["data"].attrs["env_args"])
    print("==== Env Meta ====")
    print(json.dumps(env_meta, indent=4))
    print("")

    print("==== Dataset Structure ====")
    split = 'train'  # 'train', 'val'
    tax3d_save_dir = os.path.join(os.path.dirname(args.dataset), f'{split}_tax3d/')
    save_dir = os.path.join(os.path.dirname(args.dataset), f'{split}.zarr')

    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        user_input = input()
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
            cprint('Overwriting {}'.format(tax3d_save_dir), 'red')
            os.system('rm -rf {}'.format(tax3d_save_dir))
        else:
            cprint('Exiting', 'red')
            exit(0)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tax3d_save_dir, exist_ok=True)

    num_demo = 0
    total_count = 0

    action_pcd_arrays = []
    anchor_pcd_arrays = []
    gripper_pcd_arrays = []
    ground_truth_arrays = []
    action_arrays = []
    state_arrays = []
    episode_ends_arrays = []

    for ep in demos:
        num_demo += 1
        if split == 'train':
            demo_start = 0
            if num_demo > 100:
                break
        elif split == 'val':
            demo_start = 900
            if num_demo <= 900:
                continue
        print("episode {} with {} transitions".format(ep, f["data/{}".format(ep)].attrs["num_samples"]))
        pointcloud = f["data/{}/obs/pointcloud".format(ep)]
        states = f["data/{}/states".format(ep)][()]
        actions = f["data/{}/actions".format(ep)][()]
        initial_state = states[0]

        # prepare eef states for agent_pos
        robot0_eef_pos = f["data/{}/obs/robot0_eef_pos".format(ep)]
        robot0_eef_quat = f["data/{}/obs/robot0_eef_quat".format(ep)]
        robot0_eef_vel_lin = f["data/{}/obs/robot0_eef_vel_lin".format(ep)]
        robot0_gripper_qpos = f["data/{}/obs/robot0_gripper_qpos".format(ep)]
        robot0_gripper_qvel = f["data/{}/obs/robot0_gripper_qvel".format(ep)]
        agent_pos = np.hstack([robot0_eef_pos, robot0_eef_quat, robot0_eef_vel_lin, robot0_gripper_qpos, robot0_gripper_qvel])

        # prepare ground truth
        len_episode = states.shape[0]
        total_count += len_episode
        ground_truth_subarray = np.tile(np.expand_dims(pointcloud[-1][0], axis=0), (len_episode, 1, 1))

        # update trajectory demos
        action_pcd_arrays.extend(pointcloud[:, 0, :, :])
        anchor_pcd_arrays.extend(pointcloud[:, 1, :, :])
        gripper_pcd_arrays.extend(pointcloud[:, 2, :, :])
        ground_truth_arrays.extend(ground_truth_subarray)
        action_arrays.extend(actions)
        state_arrays.extend(agent_pos)
        episode_ends_arrays.append(total_count)

        # save tax3d demos
        tax3d_demo = {
            'action_pc': pointcloud[0][0],
            'action_seg': np.ones(pointcloud[0][0].shape[0]),
            'anchor_pc': pointcloud[0][1],
            'anchor_seg': np.ones(pointcloud[0][1].shape[0]),
            'gripper_pc': pointcloud[0][2],
            'gripper_seg': np.ones(pointcloud[0][2].shape[0]),
            'state': initial_state,
            'flow': pointcloud[-1][0] - pointcloud[0][0],
        }
        cur_demo_num = num_demo - 1 - demo_start
        np.savez(
            os.path.join(tax3d_save_dir, f'demo_{cur_demo_num}.npz'),
            **tax3d_demo
        )

        # debug
        # import open3d as o3d
        # import numpy as np
        # point_geometry = o3d.geometry.PointCloud()
        # anchor_geometry = o3d.geometry.PointCloud()
        # goal_geometry = o3d.geometry.PointCloud()
        # point_geometry.points = o3d.utility.Vector3dVector(pointcloud[0][0])
        # point_geometry.paint_uniform_color(np.array([0, 0, 1]))
        # anchor_geometry.points = o3d.utility.Vector3dVector(tax3d_demo['anchor_pc'])
        # anchor_geometry.paint_uniform_color(np.array([0, 1, 0]))
        # goal_geometry.points = o3d.utility.Vector3dVector(pointcloud[-1][0])
        # goal_geometry.paint_uniform_color(np.array([1, 0, 0]))
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([point_geometry, anchor_geometry, goal_geometry, mesh_frame])


    f.close()

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
    gripper_pcd_arrays = np.stack(gripper_pcd_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    state_arrays = np.stack(state_arrays, axis=0)
    ground_truth_arrays = np.stack(ground_truth_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)
    # tax3d_arrays = np.stack(tax3d_arrays, axis=0)  # todo
    # as an additional step, create point clouds that combine action and anchor
    point_cloud_arrays = np.concatenate([action_pcd_arrays, anchor_pcd_arrays], axis=1)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    # for now, use chunk size of 100
    action_pcd_chunk_size = (100, action_pcd_arrays.shape[1], action_pcd_arrays.shape[2])
    anchor_pcd_chunk_size = (100, anchor_pcd_arrays.shape[1], anchor_pcd_arrays.shape[2])
    gripper_pcd_chunk_size = (100, gripper_pcd_arrays.shape[1], gripper_pcd_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    ground_truth_chunk_size = (100, ground_truth_arrays.shape[1], ground_truth_arrays.shape[2])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    # tax3d_chunk_size = (100, tax3d_arrays.shape[1], tax3d_arrays.shape[2])
    zarr_data.create_dataset('action_pcd', data=action_pcd_arrays, chunks=action_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('anchor_pcd', data=anchor_pcd_arrays, chunks=anchor_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('gripper_pcd', data=gripper_pcd_arrays, chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('ground_truth', data=ground_truth_arrays, chunks=ground_truth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('tax3d', data=tax3d_arrays, chunks=tax3d_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    
    cprint(f'action point cloud shape: {action_pcd_arrays.shape}, range: [{np.min(action_pcd_arrays)}, {np.max(action_pcd_arrays)}]', 'green')
    cprint(f'anchor point cloud shape: {anchor_pcd_arrays.shape}, range: [{np.min(anchor_pcd_arrays)}, {np.max(anchor_pcd_arrays)}]', 'green')
    cprint(f'gripper point cloud shape: {gripper_pcd_arrays.shape}, range: [{np.min(gripper_pcd_arrays)}, {np.max(gripper_pcd_arrays)}]', 'green')
    cprint(f'point cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'goal truth shape: {ground_truth_arrays.shape}, range: [{np.min(ground_truth_arrays)}, {np.max(ground_truth_arrays)}]', 'green')
    # cprint(f'tax3d goal point cloud shape: {tax3d_arrays.shape}, range: [{np.min(tax3d_arrays)}, {np.max(tax3d_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')
    # clean up
    del action_pcd_arrays, anchor_pcd_arrays, gripper_pcd_arrays, point_cloud_arrays, ground_truth_arrays, episode_ends_arrays, action_arrays, state_arrays
    del zarr_root, zarr_data, zarr_meta

    # maybe display error message
    print("")
    if (action_min < -1.) or (action_max > 1.):
        raise Exception("Dataset should have actions in [-1., 1.] but got bounds [{}, {}]".format(action_min, action_max))
