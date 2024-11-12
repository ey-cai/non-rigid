# An example code of loading RPDiff Dataset
from diffusion_policy_3d.dataset.rpdiff_dataset import RPDiffDataset
import tqdm

dataset = RPDiffDataset(output_format='taxpose_dataset')
dataloader = dataset.get_dataloader()
for data_dict in tqdm.tqdm(dataloader):
    print(data_dict.keys())
    for key, data in data_dict.items():
        print(key, data.shape)

    # Comment out below if you do not have desktop visualizer.
    import open3d as o3d
    import numpy as np
    point_geometry = o3d.geometry.PointCloud()
    anchor_geometry = o3d.geometry.PointCloud()
    goal_geometry = o3d.geometry.PointCloud()
    point_geometry.points = o3d.utility.Vector3dVector(data_dict['points_action_trans'][0].cpu().numpy())
    point_geometry.paint_uniform_color(np.array([0, 0, 1]))
    anchor_geometry.points = o3d.utility.Vector3dVector(data_dict['points_anchor'][0].cpu().numpy())
    anchor_geometry.paint_uniform_color(np.array([0, 1, 0]))
    goal_geometry.points = o3d.utility.Vector3dVector(data_dict['points_action'][0].cpu().numpy())
    goal_geometry.paint_uniform_color(np.array([1, 0, 0]))
    o3d.visualization.draw_geometries([point_geometry, anchor_geometry, goal_geometry]) 
    exit()