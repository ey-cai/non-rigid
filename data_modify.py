# This file helps add tax3d predicted point cloud to .npz files for evaluation.

import numpy as np
import open3d as o3d
import torch
import os
from tqdm import tqdm
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

split = 'val_ood'
data_dir = "/home/yingyuan/non-rigid/datasets/eric_flow/cloth=multi-fixed anchor=single-random hole=single"
zarr_path = os.path.join(data_dir, '{}.zarr'.format(split))
replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['point_cloud', 'state', 'action', 'action_pcd', 'anchor_pcd', 'ground_truth', 'tax3d'])
predicted_pcd = replay_buffer['tax3d']
# print(predicted_pcd[0].shape)
# print(predicted_pcd[301][:5])
# print(predicted_pcd[602][:5])

demo_dir = data_dir + "/{}_tax3d".format(split)
demo_write_dir = data_dir + "/{}_tax3d_with_pred".format(split)
if not os.path.exists(demo_write_dir):
    os.mkdir(demo_write_dir)
num_demos = {'train': 64, 'val': 40, 'val_ood': 40}
for index in tqdm(range(num_demos[split])):
    demo = np.load(f"{demo_dir}/demo_{index}.npz", allow_pickle=True)
    tax3d_pred = predicted_pcd[301 * index]
    new_demo = {
        'action_pc': demo['action_pc'],
        'action_seg': demo['action_seg'],
        'anchor_pc': demo['anchor_pc'],
        'anchor_seg': demo['anchor_seg'],
        'speed_factor': demo['speed_factor'],
        'rot': demo['rot'],
        'trans': demo['trans'],
        'deform_params': demo['deform_params'],
        'anchors': demo['anchors'],
        'flow': demo['flow'],
        'tax3d': tax3d_pred,
    }
    np.savez(
        os.path.join(demo_write_dir, f'demo_{index}.npz'),
        **new_demo
    )

    # # debug
    # goal_pc = new_demo['action_pc'] + new_demo['flow']
    # point_geometry = o3d.geometry.PointCloud()
    # goal_geometry = o3d.geometry.PointCloud()
    # point_geometry.points = o3d.utility.Vector3dVector(tax3d_pred)
    # point_geometry.paint_uniform_color(np.array([0, 0, 1]))
    # goal_geometry.points = o3d.utility.Vector3dVector(goal_pc)
    # goal_geometry.paint_uniform_color(np.array([1, 0, 0]))
    # o3d.visualization.draw_geometries([point_geometry, goal_geometry])
    # exit()