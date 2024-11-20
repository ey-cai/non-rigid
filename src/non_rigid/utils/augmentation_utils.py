from typing import Any, Dict, Optional, Tuple, Union

import torch
from pytorch3d.ops import ball_query
from torch.nn import functional as F


def ball_occlusion(
    points: torch.Tensor, radius: float = 0.05, return_mask: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    Occludes a ball shaped region of the point cloud with radius up to `radius`.

    Args:
        points: [N, 3] tensor of points
        radius: maximum radius of the occlusion ball
        return_mask: if True, returns the mask of the occluded points

    Returns:
        points: [N', 3] tensor of points
        mask: [N] tensor of bools indicating which points were occluded
    """
    idx = torch.randint(points.shape[0], [1])
    center = points[idx]
    sampled_radius = (radius - 0.025) * torch.rand(1) + 0.025
    ret = ball_query(
        center.unsqueeze(0),
        points.unsqueeze(0),
        radius=sampled_radius,
        K=points.shape[0],
    )
    mask = torch.isin(
        torch.arange(points.shape[0], device=points.device), ret.idx[0], invert=True
    )
    if return_mask:
        return points[mask], mask
    return points[mask]


def plane_occlusion(
    points: torch.Tensor, stand_off: float = 0.02, return_mask: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    Occludes a plane shaped region of the point cloud with stand-off distance `stand_off`.

    Args:
        points: [N, 3] tensor of points
        stand_off: distance of the plane from the point cloud
        return_mask: if True, returns the mask of the occluded points

    Returns:
        points: [N', 3] tensor of points
        mask: [N] tensor of bools indicating which points were occluded
    """
    idx = torch.randint(points.shape[0], [1])
    pt = points[idx]
    center = points.mean(dim=0, keepdim=True)
    plane_norm = F.normalize(pt - center, dim=-1)
    plane_orig = pt - stand_off * plane_norm
    points_vec = F.normalize(points - plane_orig, dim=-1)
    split = plane_norm @ points_vec.transpose(-1, -2)
    mask = split[0] < 0
    if return_mask:
        return points[mask], mask
    return points[mask]


def maybe_apply_augmentations(
    points: torch.Tensor,
    min_num_points: int,
    ball_occlusion_param: Dict[str, Any],
    plane_occlusion_param: Dict[str, Any],
) -> torch.Tensor:
    """
    Potentially applies augmentations to the point cloud, considering the dataset configuration e.g. min. number of points.

    Args:
        points: [N, 3] tensor of points
        min_num_points: minimum number of points required
        ball_occlusion_param: parameters for ball occlusion
        plane_occlusion_param: parameters for plane occlusion

    Returns:
        points: [N', 3] tensor of points
    """

    if points.shape[0] < min_num_points:
        return points

    new_points = points
    # Maybe apply ball occlusion
    if torch.rand(1) < ball_occlusion_param["ball_occlusion"]:
        temp_points = ball_occlusion(
            new_points, radius=ball_occlusion_param["ball_radius"], return_mask=False
        )
        assert isinstance(temp_points, torch.Tensor)
        if temp_points.shape[0] > min_num_points:
            new_points = temp_points

    # Maybe apply plane occlusion
    if torch.rand(1) < plane_occlusion_param["plane_occlusion"]:
        temp_points = plane_occlusion(
            new_points,
            stand_off=plane_occlusion_param["plane_standoff"],
            return_mask=False,
        )
        assert isinstance(temp_points, torch.Tensor)
        if temp_points.shape[0] > min_num_points:
            new_points = temp_points

    return new_points
