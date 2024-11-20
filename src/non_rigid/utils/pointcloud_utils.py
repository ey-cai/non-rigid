import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import Transform3d

from non_rigid.utils.transform_utils import random_se3


def expand_pcd(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Expands a batch of point clouds by a fixed factor. This is useful when sampling
    multiple diffusion predictions for a batch of point clouds.

    Args:
        points (torch.Tensor): [B, N, C]  or [B, N] Point cloud to expand.
        num_samples (int): Number of samples to expand to.
    """
    assert points.dim() == 3 or points.dim() == 2
    if points.dim() == 2:
        B, N = points.shape
        points = (
            points.unsqueeze(1).expand(B, num_samples, N).reshape(B * num_samples, N)
        )
    elif points.dim() == 3:
        B, N, C = points.shape
        points = (
            points.unsqueeze(1)
            .expand(B, num_samples, N, C)
            .reshape(B * num_samples, N, C)
        )
    else:
        raise ValueError("Invalid input dimensions.")
    return points


def downsample_pcd(
    points: torch.Tensor, num_points: int = 1024, type: str = "fps"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Downsamples a pointcloud using a specified method.

    Args:
        points (torch.Tensor): [B, N, 3] Pointcloud to downsample.
        num_points (int): Number of points to downsample to.
        type (str): Method of downsampling the point cloud.

    Returns:
        (torch.Tensor): Downsampled pointcloud.
        (torch.Tensor): Indices of the downsampled points in the original pointcloud.
    """

    if re.match(r"^fps$", type) is not None:
        return sample_farthest_points(points, K=num_points, random_start_point=True)  # type: ignore
    elif re.match(r"^random$", type) is not None:
        random_idx = torch.randperm(points.shape[1])[:num_points]
        return points[:, random_idx], random_idx
    elif re.match(r"^random_0\.[0-9]$", type) is not None:
        prob = float(re.match(r"^random_(0\.[0-9])$", type).group(1))  # type: ignore
        if np.random.random() > prob:
            return sample_farthest_points(points, K=num_points, random_start_point=True)  # type: ignore
        else:
            random_idx = torch.randperm(points.shape[1])[:num_points]
            return points[:, random_idx], random_idx
    elif re.match(r"^[0-9]+N_random_fps$", type) is not None:
        random_num_points = (
            int(re.match(r"^([0-9]+)N_random_fps$", type).group(1)) * num_points  # type: ignore
        )
        random_idx = torch.randperm(points.shape[1])[:random_num_points]
        random_points = points[:, random_idx]
        return sample_farthest_points(random_points, K=num_points, random_start_point=True)  # type: ignore
    else:
        raise NotImplementedError(f"Downsample type {type} not implemented")


def points_to_axis_aligned_rect(
    points: torch.Tensor, buffer: float = 0.1
) -> torch.Tensor:
    """
    Given a point cloud, return the axis-aligned bounding box of the point cloud with a buffer.

    Args:
        points (torch.Tensor): [B, N, 3] Point cloud to get the bounding box of.
        buffer (float): The buffer to add to the bounding box.

    Returns:
        (torch.Tensor): [B, 6] Axis-aligned bounding box of the point cloud.
    """
    assert points.ndim == 3
    rect_prism = torch.hstack([points.min(axis=1)[0], points.max(axis=1)[0]])  # type: ignore
    buffer_w = (rect_prism[:, 3:6] - rect_prism[:, 0:3]) * buffer
    rect_prism = rect_prism + torch.hstack([-buffer_w, buffer_w])

    return rect_prism


def axis_aligned_rect_intersect(
    rect_prism1: torch.Tensor, rect_prism2: torch.Tensor
) -> torch.Tensor:
    """
    Given two axis-aligned rectangular prisms, return whether they intersect.

    Args:
        rect_prism1 (torch.Tensor): [B, 6] Axis-aligned rectangular prism.
        rect_prism2 (torch.Tensor): [B, 6] Axis-aligned rectangular prism.

    Returns:
        (torch.Tensor): [B] Whether the two rectangular prisms intersect.
    """
    conditions = (
        (rect_prism1[:, 0] <= rect_prism2[:, 3]).int()
        + (rect_prism1[:, 3] >= rect_prism2[:, 0]).int()
        + (rect_prism1[:, 1] <= rect_prism2[:, 4]).int()
        + (rect_prism1[:, 4] >= rect_prism2[:, 1]).int()
        + (rect_prism1[:, 2] <= rect_prism2[:, 5]).int()
        + (rect_prism1[:, 5] >= rect_prism2[:, 2]).int()
    )
    return conditions >= 6


def combine_axis_aligned_rect(rect_prisms: List[torch.Tensor]) -> torch.Tensor:
    """
    Given a list of axis-aligned rectangular prisms, return the axis-aligned bounding box of all the prisms.

    Args:
        rect_prisms (List[torch.Tensor]): List of axis-aligned rectangular prisms.

    Returns:
        (torch.Tensor): [B, 6] Axis-aligned bounding box of all the prisms.
    """
    assert len(rect_prisms) > 0
    assert all([r.ndim == 2 for r in rect_prisms])
    assert all([r.shape[1] == 6 for r in rect_prisms])
    return torch.hstack(
        [
            torch.min(torch.stack([r[:, 0:3] for r in rect_prisms], dim=0), dim=0)[0],
            torch.max(torch.stack([r[:, 3:6] for r in rect_prisms], dim=0), dim=0)[0],
        ]
    )


def get_nonintersecting_anchor(
    points_anchor_base: torch.Tensor,
    rect_prisms_base: torch.Tensor,
    rot_var: float = np.pi,
    trans_var: float = 1,
    return_debug: bool = False,
    rot_sample_method: str = "axis_angle",
) -> Tuple[torch.Tensor, Transform3d, dict]:
    """
    Given a base point cloud and a set of axis-aligned rectangular prisms, generate a non-intersecting anchor.

    Args:
        points_anchor_base (torch.Tensor): [B, N, 3] Base point cloud.
        rect_prisms_base (torch.Tensor): [B, 6] Axis-aligned rectangular prisms.
        rot_var (float): Maximum rotation in radians
        trans_var (float): Maximum translation
        return_debug (bool): Whether to return debug information.
        rot_sample_method (str): Method to sample the rotation. Options are:
            - "axis_angle": Random axis angle sampling
            - "axis_angle_uniform_z": Random axis angle sampling with uniform z axis rotation
            - "quat_uniform": Uniform SE(3) sampling
            - "random_flat_upright": Random rotation around z axis and xy translation (no z translation)
            - "random_upright": Random rotation around z axis and xyz translation

    Returns:
        (torch.Tensor): [B, N, 3] Non-intersecting anchor point cloud.
        (Transform3d): Transformation applied to the base point cloud to get the non-intersecting anchor.
        (dict): Debug information.
    """

    points_anchor1 = points_anchor_base
    rect_prisms1 = rect_prisms_base

    success = torch.tensor(
        [False] * points_anchor1.shape[0], device=points_anchor1.device
    )
    tries = 0

    points_anchor2 = torch.empty_like(points_anchor1)
    T = None

    # Empirically, the success rate for any single augmentation for this env is 1/1.3778 = approx 72.5%
    # Success rate for batch size 16 is 0.725^16 = 0.58%
    while not torch.all(success):  # and tries < 10:
        T_temp = random_se3(
            N=points_anchor_base.shape[0],
            rot_var=rot_var,
            trans_var=trans_var,
            rot_sample_method=rot_sample_method,
            device=points_anchor1.device,
        )
        points_anchor2_temp = T_temp.transform_points(points_anchor1)
        rect_prisms2 = points_to_axis_aligned_rect(points_anchor2_temp)

        intersects_temp = axis_aligned_rect_intersect(rect_prisms1, rect_prisms2)

        points_anchor2[intersects_temp == False] = points_anchor2_temp[
            intersects_temp == False
        ]
        # This doesnt work:
        T = T_temp

        success = torch.logical_or(success, torch.logical_not(intersects_temp))
        tries += 1

    if return_debug:
        rect_prisms2 = points_to_axis_aligned_rect(points_anchor2)
        debug = dict(tries=tries, rect_prisms1=rect_prisms1, rect_prisms2=rect_prisms2)
    else:
        debug = {}

    return points_anchor2, T, debug


def get_multi_anchor_scene(
    points_gripper: Optional[torch.Tensor],
    points_action: torch.Tensor,
    points_anchor_base: torch.Tensor,
    rot_var: float = np.pi,
    trans_var: float = 1.0,
    transform_base: bool = False,
    return_debug: bool = False,
    rot_sample_method: str = "axis_angle",
    num_anchors_to_add: int = 1,
) -> Tuple[
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    List[torch.Tensor],
    List[Transform3d],
    dict,
]:
    """
    Generate a scene of multiple non-intersecting anchors.

    Args:
        points_gripper (torch.Tensor): [B, N, 3] Gripper point cloud.
        points_action (torch.Tensor): [B, N, 3] Action point cloud.
        points_anchor_base (torch.Tensor): [B, N, 3] Anchor point cloud.
        rot_var (float): Maximum rotation in radians
        trans_var (float): Maximum translation
        transform_base (bool): Whether to transform the base point cloud.
        return_debug (bool): Whether to return debug information.
        rot_sample_method (str): Method to sample the rotation. Options are:
            - "axis_angle": Random axis angle sampling
            - "axis_angle_uniform_z": Random axis angle sampling with uniform z axis rotation
            - "quat_uniform": Uniform SE(3) sampling
            - "random_flat_upright": Random rotation around z axis and xy translation (no z translation)
            - "random_upright": Random rotation around z axis and xyz translation
        num_anchors_to_add (int): Number of anchors to add.

    Returns:
        (torch.Tensor): [B, N, 3] Gripper point cloud.
        (torch.Tensor): [B, N, 3] Action point cloud.
        (torch.Tensor): [B, N, 3] Base point cloud.
        (List[torch.Tensor]): List of non-intersecting anchor point clouds.
        (List[Transform3d]): List of transformations applied to the base point cloud to get the non-intersecting anchors.
        (dict): Debug information.
    """

    debug = {}
    if transform_base:
        N = points_anchor_base.shape[0]
        T = random_se3(
            N,
            rot_var=rot_var,
            trans_var=trans_var,
            device=points_anchor_base.device,  # type: ignore
        )

        points_anchor_base = T.transform_points(points_anchor_base)
        points_action = T.transform_points(points_action)
        if points_gripper is not None:
            points_gripper = T.transform_points(points_gripper)
        debug["transform_base_T"] = T
    else:
        debug["transform_base_T"] = None

    points_anchor1 = points_anchor_base

    rect_prisms_base = combine_axis_aligned_rect(
        [
            points_to_axis_aligned_rect(points_anchor_base),
            points_to_axis_aligned_rect(points_action),
        ]
        + (
            [
                points_to_axis_aligned_rect(points_gripper),
            ]
            if points_gripper is not None
            else []
        )
    )

    new_points_list = []
    T_aug_list = []

    for i in range(num_anchors_to_add):
        points_anchor2, T2, debug_temp = get_nonintersecting_anchor(
            points_anchor_base,
            rect_prisms_base,
            rot_var=rot_var,
            trans_var=trans_var,
            return_debug=return_debug,
            rot_sample_method=rot_sample_method,
        )
        debug.update(debug_temp)

        if return_debug:
            # If rotate_base=True, these axis aligned rects don't exactly match the intersection checks that were done during
            # the non-intersecting anchor generation because a rotation was applied afterwards
            # However, it's a good visualization
            rect_prisms1 = combine_axis_aligned_rect(
                [
                    points_to_axis_aligned_rect(points_anchor1),
                    points_to_axis_aligned_rect(points_action),
                ]
                + (
                    [
                        points_to_axis_aligned_rect(points_gripper),
                    ]
                    if points_gripper is not None
                    else []
                )
            )
            rect_prisms2 = points_to_axis_aligned_rect(points_anchor2)
            debug_temp = dict(
                tries=debug["tries"],
                rect_prisms1=rect_prisms1,
                rect_prisms2=rect_prisms2,
            )
            debug.update(debug_temp)

        new_points_list.append(points_anchor2)
        T_aug_list.append(T2)

        # Update rect_prisms_base to also include the new anchor
        rect_prisms_base = combine_axis_aligned_rect(
            [
                rect_prisms_base,
                points_to_axis_aligned_rect(points_anchor2),
            ]
        )

    return (
        points_gripper,
        points_action,
        points_anchor1,
        new_points_list,
        T_aug_list,
        debug,
    )
