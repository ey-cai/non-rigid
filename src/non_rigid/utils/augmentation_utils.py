import omegaconf
from pytorch3d.ops import ball_query
import torch
from torch.nn import functional as F
from typing import Tuple, Optional, Dict, Any


def ball_occlusion(
    init_points: torch.Tensor,
    final_points: torch.Tensor, 
    radius: float = 0.05, 
    return_mask: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
    idx = torch.randint(init_points.shape[0], [1])
    sampled_radius = (radius - 0.025) * torch.rand(1) + 0.025

    # for init pcd
    init_center = init_points[idx]
    init_ret = ball_query(
        init_center.unsqueeze(0),
        init_points.unsqueeze(0),
        radius=sampled_radius,
        K=init_points.shape[0],
    )
    init_mask = torch.isin(
        torch.arange(init_points.shape[0], device=init_points.device), init_ret.idx[0], invert=True
    )
    init_points_aug = init_points[init_mask]

    # for final pcd, assuming 1-1 correspondance!
    if final_points is not None:
        final_center = final_points[idx]
        final_ret = ball_query(
            final_center.unsqueeze(0),
            final_points.unsqueeze(0),
            radius=sampled_radius,
            K=final_points.shape[0],
        )
        final_mask = torch.isin(
            torch.arange(final_points.shape[0], device=final_points.device), final_ret.idx[0], invert=True
        )
        final_points_aug = final_points[final_mask]
    else:
        final_mask = None
        final_points_aug = None

    if return_mask:
        return init_points_aug, init_mask, final_points_aug, final_mask
    return init_points_aug, final_points_aug


def plane_occlusion(
    init_points: torch.Tensor,
    final_points: torch.Tensor, 
    stand_off: float = 0.02, 
    return_mask: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
    idx = torch.randint(init_points.shape[0], [1])

    # for init pcd
    init_pt = init_points[idx]
    init_center = init_points.mean(dim=0, keepdim=True)
    init_plane_norm = F.normalize(init_pt - init_center, dim=-1)
    init_plane_orig = init_pt - stand_off * init_plane_norm
    init_points_vec = F.normalize(init_points - init_plane_orig, dim=-1)
    init_split = init_plane_norm @ init_points_vec.transpose(-1, -2)
    init_mask = init_split[0] < 0
    init_points_aug = init_points[init_mask]

    # for final pcd, assuming 1-1 correspondance!
    if final_points is not None:
        final_pt = final_points[idx]
        final_center = final_points.mean(dim=0, keepdim=True)
        final_plane_norm = F.normalize(final_pt - final_center, dim=-1)
        final_plane_orig = final_pt - stand_off * final_plane_norm
        final_points_vec = F.normalize(final_points - final_plane_orig, dim=-1)
        final_split = final_plane_norm @ final_points_vec.transpose(-1, -2)
        final_mask = final_split[0] < 0
        final_points_aug = final_points[final_mask]
    else:
        final_mask = None
        final_points_aug = None

    if return_mask:
        return init_points_aug, init_mask, final_points_aug, final_mask
    return init_points_aug, final_points_aug

def random_drop(
    init_points: torch.Tensor,
    final_points: torch.Tensor, 
    percent: float = 0.1, 
    return_mask: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Randomly drop percentage of the input point cloud with percentage `percent`.

    Args:
        points: [N, 3] tensor of points
        percent: percentage of points of the point cloud to be dropped
        return_mask: if True, returns the mask of the occluded points

    Returns:
        points: [N', 3] tensor of points
        mask: [N] tensor of bools indicating which points were occluded
    """
    percent = max(0.0, min(1.0, percent))
    
    num_points = init_points.shape[0]
    num_drop = int(num_points * percent)
    
    drop_indices = torch.randperm(num_points)[:num_drop]

    mask = torch.ones(num_points, dtype=torch.bool)
    mask[drop_indices] = False

    # for init pcd
    init_mask = mask
    init_points_aug = init_points[init_mask]

    # for final pcd, assuming 1-1 correspondance!
    if final_points is not None:
        final_mask = mask
        final_points_aug = final_points[mask]
    else:
        final_mask = None
        final_points_aug = None   

    if return_mask:
        return init_points_aug, init_mask, final_points_aug, final_mask
    return init_points_aug, final_points_aug


def maybe_apply_augmentations(
    init_points: torch.Tensor,
    final_points: torch.Tensor,
    min_num_points: int,
    ball_occlusion_param: Dict[str, Any],
    plane_occlusion_param: Dict[str, Any],
    random_drop_param: Dict[str, Any],
) -> torch.Tensor:
    """
    Potentially applies augmentations to the point cloud, considering the dataset configuration e.g. min. number of points.

    Args:
        points: [N, 3] tensor of points
        min_num_points: minimum number of points required
        ball_occlusion_param: parameters for ball occlusion
        plane_occlusion_param: parameters for plane occlusion
        random_drop_param: parameters for random drop

    Returns:
        points: [N', 3] tensor of points
    """

    if init_points.shape[0] < min_num_points:
        return init_points, final_points, None
    
    if final_points is not None and final_points.shape[0] < min_num_points:
        return init_points, final_points, None

    new_init_points = init_points
    new_final_points = final_points

    # Maybe apply ball occlusion
    if torch.rand(1) < ball_occlusion_param["ball_occlusion"]:
        temp_init_points, temp_final_points = ball_occlusion(
            new_init_points, 
            new_final_points, 
            radius=ball_occlusion_param["ball_radius"], 
            return_mask=False
        )
        if temp_init_points.shape[0] > min_num_points:
            new_init_points = temp_init_points
        if temp_final_points is not None and temp_final_points.shape[0] > min_num_points:
            new_final_points = temp_final_points

    # Maybe apply plane occlusion
    if torch.rand(1) < plane_occlusion_param["plane_occlusion"]:
        temp_init_points, temp_final_points = plane_occlusion(
            new_init_points, 
            new_final_points,
            stand_off=plane_occlusion_param["plane_standoff"],
            return_mask=False,
        )
        if temp_init_points.shape[0] > min_num_points:
            new_init_points = temp_init_points
        if temp_final_points is not None and temp_final_points.shape[0] > min_num_points:
            new_final_points = temp_final_points

    # Maybe apply random drops
    if torch.rand(1) < random_drop_param["random_drop_probability"]:
        temp_init_points, temp_final_points = random_drop(
            new_init_points, 
            new_final_points,
            percent=random_drop_param["random_drop_percent"],
            return_mask=False,
        )
        if temp_init_points.shape[0] > min_num_points:
            new_init_points = temp_init_points
        if temp_final_points is not None and temp_final_points.shape[0] > min_num_points:
            new_final_points = temp_final_points

    return new_init_points, new_final_points
