import pytorch3d
from pytorch3d.transforms import Transform3d, Rotate, Translate, \
    rotation_6d_to_matrix, axis_angle_to_matrix, so3_rotation_angle, quaternion_to_matrix
# from pytorch3d.transforms import se3_exp_map

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

mse_criterion = nn.MSELoss(reduction='sum')


def random_se3(N, rot_var=np.pi/180 * 5, trans_var=0.1, device=None, fix_random=False, rot_sample_method="axis_angle"):
    assert rot_sample_method in ["axis_angle", "axis_angle_uniform_z", "quat_uniform", "random_flat_upright", "random_upright"]
    
    if rot_sample_method == "quat_uniform" and np.isclose(rot_var, np.pi):
        # This true uniform SE(3) sampling tends to make it hard to train the models
        # In contrast, the axis angle sampling tends to leave the objects close to upright
        quat = torch.randn(N, 4, device=device)
        quat = quat / torch.linalg.norm(quat)
        R = quaternion_to_matrix(quat)
        random_translation = torch.randn(N, 3, device=device)
        translation_ratio = trans_var / \
            torch.norm(random_translation, dim=1).max().item()
        t = torch.rand(1).item()*translation_ratio*random_translation
    elif rot_sample_method == "axis_angle_uniform_z":
        # this is random axis angle sampling
        axis_angle_random = torch.randn(N, 3, device=device)
        rot_ratio = torch.rand(1).item()*rot_var / \
            torch.norm(axis_angle_random, dim=1).max().item()
        constrained_axix_angle = rot_ratio*axis_angle_random  # max angle is rot_var
        R_random = axis_angle_to_matrix(constrained_axix_angle)
        
        # this is uniform z axis rotation sampling
        theta = torch.rand(N, 1, device=device) * 2 * np.pi
        axis_angle_z = torch.cat([torch.zeros(N, 2, device=device), theta], dim=1)
        R_z = axis_angle_to_matrix(axis_angle_z)
        
        R = torch.bmm(R_z, R_random)
        random_translation = torch.randn(N, 3, device=device)
        translation_ratio = trans_var / \
            torch.norm(random_translation, dim=1).max().item()
        t = torch.rand(1).item()*translation_ratio*random_translation
    elif rot_sample_method == "random_upright":
        # Random rotation around z axis and xy translation (no z translation)
        theta = torch.rand(N, 1, device=device) * 2 * np.pi
        axis_angle_z = torch.cat([torch.zeros(N, 2, device=device), theta], dim=1)
        R = axis_angle_to_matrix(axis_angle_z)
        
        random_translation = torch.randn(N, 3, device=device)
        translation_ratio = trans_var / \
            torch.norm(random_translation, dim=1).max().item()
        t = torch.rand(1).item()*translation_ratio*random_translation
    elif rot_sample_method == "random_flat_upright":
        # Random rotation around z axis and xy translation (no z translation)
        theta = torch.rand(N, 1, device=device) * 2 * np.pi
        axis_angle_z = torch.cat([torch.zeros(N, 2, device=device), theta], dim=1)
        R = axis_angle_to_matrix(axis_angle_z)
        
        random_translation = torch.randn(N, 3, device=device)
        random_translation[:, 2] = 0
        translation_ratio = trans_var / \
            torch.norm(random_translation, dim=1).max().item()
        t = torch.rand(1).item()*translation_ratio*random_translation
    else:
        # this is random axis angle sampling (rot_sample_method == "axis_angle")
        axis_angle_random = torch.randn(N, 3, device=device)
        rot_ratio = torch.rand(1).item()*rot_var / \
            torch.norm(axis_angle_random, dim=1).max().item()
        constrained_axix_angle = rot_ratio*axis_angle_random  # max angle is rot_var
        R = axis_angle_to_matrix(constrained_axix_angle)
        random_translation = torch.randn(N, 3, device=device)
        translation_ratio = trans_var / \
            torch.norm(random_translation, dim=1).max().item()
        t = torch.rand(1).item()*translation_ratio*random_translation
    return Rotate(R, device=device).translate(t)