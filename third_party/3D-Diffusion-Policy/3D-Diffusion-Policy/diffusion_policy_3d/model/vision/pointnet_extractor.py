import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

from diffusion_policy_3d.model.vision.pointnet2_utils import PointNet2_small, PointNet2_small2, PointNet2ssg_small
from diffusion_policy_3d.model.vision.point_transformer import PointTransformerSeg, TrivialLocallyTransformer
from diffusion_policy_3d.common.network_helper import replace_bn_with_gn
from diffusion_policy_3d.model.vision.layers import RelativeCrossAttentionModule
from diffusion_policy_3d.model.vision.position_encodings import RotaryPositionEncoding3D

import einops


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        # assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()


class PointNetAttention(nn.Module):  
    def __init__(self, in_channels=3, out_channels=1024, **kwargs):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNetAttention, self).__init__()

        print(f'PointNetAttention')

        mlp_out_dim = out_channels
        self.local_mlp_k = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )
        self.local_mlp_v = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )
        self.local_mlp_q = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )
        # self.global_mlp = nn.Linear(n_points * mlp_out_dim, mlp_out_dim)
        self.softmax  = nn.Softmax(dim=-1)
        self.reset_parameters_()
        self.frame_count = 0

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: [B, N, 3]
        '''     
        # Local
        q = self.local_mlp_q(x).permute(0, 2, 1)
        k = self.local_mlp_k(x)
        v = self.local_mlp_v(x)
        #print(q.shape, k.shape, v.shape)
        energy = torch.bmm(q, k)  # transpose check
        #print(energy.shape)
        attention = self.softmax(energy).permute(0, 2, 1)
        out = torch.bmm(v, attention)
        # gloabal max pooling
        # x = torch.max(x, dim=1)[0]
        out, indices = torch.max(out, dim=1)
        # _, indices = torch.max(out, dim=1)
        # out = self.global_mlp(out.reshape(out.shape[0], -1))
        return out


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNet2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1024, **kwargs):
        super(PointNet2, self).__init__()
        normal_channel = True if in_channels == 6 else False
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channels, mlp=[64, 64, 128], group_all=False)  # 512 may be bug
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, out_channels)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        xyz = xyz.permute(0, 2, 1)
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x


class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        points = points[:, :1024, :]
        pn_feat = self.extractor(points)    # B * out_channel
            
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels


class TAX3DEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.action_pcd_key = 'action_pcd'
        self.anchor_pcd_key = 'anchor_pcd'
        self.goal_pcd_key = 'goal_pcd'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.extractor_mode = pointcloud_encoder_cfg.extractor_mode
        self.use_goal_pc = pointcloud_encoder_cfg.use_goal_pc
        self.use_onehot = pointcloud_encoder_cfg.use_onehot
        self.use_flow = pointcloud_encoder_cfg.use_flow
        
        if self.extractor_mode == "simple":
            print(self.pointnet_type)
            if self.pointnet_type == "pointnet":
                if self.use_onehot:
                    pointcloud_encoder_cfg.in_channels = 6
                    if self.use_flow:
                        pointcloud_encoder_cfg.in_channels += 3
                    self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)  # xyz
                else:
                    pointcloud_encoder_cfg.in_channels = 3
                    if self.use_flow:
                        pointcloud_encoder_cfg.in_channels += 3
                    self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
            elif self.pointnet_type == "pointnet2":  # this encoder only considers xyz
                pointcloud_encoder_cfg.in_channels = 3
                if self.use_flow:
                    pointcloud_encoder_cfg.in_channels += 3
                self.extractor = PointNet2_small2(num_classes=pointcloud_encoder_cfg.out_channels, in_channels=pointcloud_encoder_cfg.in_channels)
                self.extractor = replace_bn_with_gn(self.extractor,features_per_group=4)
                # if self.use_onehot:
                #     pointcloud_encoder_cfg.in_channels = 6
                #     self.extractor = PointNet2(**pointcloud_encoder_cfg)
                # else:
                #     pointcloud_encoder_cfg.in_channels = 3
                #     self.extractor = PointNet2(**pointcloud_encoder_cfg)
            elif self.pointnet_type == "pointnet2ssg":  # this encoder only considers xyz
                pointcloud_encoder_cfg.in_channels = 3
                if self.use_flow:
                    pointcloud_encoder_cfg.in_channels += 3
                self.extractor = PointNet2ssg_small(num_classes=pointcloud_encoder_cfg.out_channels, in_channels=pointcloud_encoder_cfg.in_channels)
                self.extractor = replace_bn_with_gn(self.extractor,features_per_group=4)
            elif self.pointnet_type == "pointnet_attention": 
                if self.use_onehot:
                    pointcloud_encoder_cfg.in_channels = 6
                    if self.use_flow:
                        pointcloud_encoder_cfg.in_channels += 3
                    self.extractor = PointNetAttention(**pointcloud_encoder_cfg)
                else:
                    pointcloud_encoder_cfg.in_channels = 3
                    if self.use_flow:
                        pointcloud_encoder_cfg.in_channels += 3
                    self.extractor = PointNetAttention(**pointcloud_encoder_cfg)
            elif self.pointnet_type == 'point_transformer':
                pointcloud_encoder_cfg.in_channels = 3
                if self.use_flow:
                    pointcloud_encoder_cfg.in_channels += 3
                self.extractor = PointTransformerSeg(
                    npoints=1024,  # without goal pcd
                    n_c=pointcloud_encoder_cfg.out_channels,
                    nblocks=3,
                    nneighbor=16,
                    d_points=pointcloud_encoder_cfg.in_channels,
                    transformer_dim=32,
                    base_dim=8,
                    downsample_ratio=8,
                    hidden_dim=128
                )
                self.extractor = replace_bn_with_gn(self.extractor, features_per_group=4)
            else:
                raise NotImplementedError
            
        elif self.extractor_mode == "incremental":
            pointcloud_encoder_cfg.in_channels = 3
            if self.use_flow:
                pointcloud_encoder_cfg.in_channels += 3
            encoder_output_dim = pointcloud_encoder_cfg.out_channels
            hidden_layer_dim = encoder_output_dim
            vision_encoder = nn.Sequential(
                nn.Linear(pointcloud_encoder_cfg.in_channels, hidden_layer_dim),
                nn.ReLU(),
                nn.Linear(hidden_layer_dim, hidden_layer_dim),
                nn.ReLU(),
                nn.Linear(hidden_layer_dim, encoder_output_dim)
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)

            attention_num_heads = 2
            attention_num_layers = 1
            attn_layers = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
            attn_layers = replace_bn_with_gn(attn_layers)
            self_attn_layers = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
            self_attn_layers = replace_bn_with_gn(self_attn_layers)
            goal_attn_layers = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
            goal_attn_layers = replace_bn_with_gn(goal_attn_layers)
            goal_self_attn_layers = RelativeCrossAttentionModule(encoder_output_dim, attention_num_heads, attention_num_layers)
            goal_self_attn_layers = replace_bn_with_gn(goal_self_attn_layers)

            self.nets = nn.ModuleDict({
                'vision_encoder': vision_encoder,
                'relative_pe_layer': RotaryPositionEncoding3D(encoder_output_dim),
                'attn_layers': attn_layers,
                'self_attn_layers': self_attn_layers,
                'goal_attn_layers': goal_attn_layers,
                'goal_self_attn_layers': goal_self_attn_layers,
                'final_layer': nn.Linear(580*encoder_output_dim, encoder_output_dim),
            })
            # self.n_output_channels *= 580
        
        elif self.extractor_mode == "all":
            raise NotImplementedError
        
        else:
            raise NotImplementedError


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        action_pcd = points[:, :580, :]
        anchor_pcd = points[:, 580:580*2, :]

        if self.use_goal_pc:
            goal_pcd = points[:, 580*2:, :]

        n_action_pcd = action_pcd.shape[1]
        n_anchor_pcd = anchor_pcd.shape[1]
        if self.use_goal_pc:
            n_goal_pcd = goal_pcd.shape[1]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        # if self.use_imagined_robot:
        #     img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
        #     points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        if self.extractor_mode == "simple":
            if self.use_onehot:
                one_hot = torch.zeros_like(points)
                one_hot[:, :n_action_pcd, 0] = 1
                one_hot[:, n_action_pcd:n_action_pcd+n_anchor_pcd, 1] = 1
                if self.use_goal_pc:
                    one_hot[:, n_action_pcd+n_anchor_pcd:, 2] = 1
                points = torch.cat([points, one_hot], dim=-1)
            if self.use_goal_pc:

                # import open3d as o3d
                # import numpy as np
                # point_geometry = o3d.geometry.PointCloud()
                # goal_geometry = o3d.geometry.PointCloud()
                # point_geometry.points = o3d.utility.Vector3dVector(points[0, :, :3].cpu().numpy())
                # point_geometry.paint_uniform_color(np.array([0, 0, 1]))
                # goal_geometry.points = o3d.utility.Vector3dVector(goal_pcd[0, :, :3].cpu().numpy())
                # goal_geometry.paint_uniform_color(np.array([1, 0, 0]))
                # o3d.visualization.draw_geometries([point_geometry, goal_geometry])
                # exit()
                
                pn_feat = self.extractor(points)    # B * out_channel
            elif self.use_flow:
                flow_action = points[:, n_action_pcd+n_anchor_pcd:, :3] - points[:, :n_action_pcd, :3]
                flow_anchor = torch.zeros_like(flow_action)
                action_input = torch.cat([points[:, :n_action_pcd, :3], flow_action, points[:, :n_action_pcd, 3:]], dim=-1)
                anchor_input = torch.cat([points[:, n_action_pcd:n_action_pcd+n_anchor_pcd, :3], 
                                          flow_anchor, 
                                          points[:, n_action_pcd:n_action_pcd+n_anchor_pcd, 3:]], dim=-1)
                all_points = torch.cat([action_input, anchor_input], dim=1)
                pn_feat = self.extractor(all_points)

                # # debug
                # import matplotlib.pyplot as plt
                # fig = plt.figure(figsize=(10, 7))
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(points[0, :580, 0].cpu().numpy(), points[0, :580, 1].cpu().numpy(), points[0, :580, 2].cpu().numpy(), c='blue', label='Action Point Cloud')
                # ax.scatter(points[0, 2*580:, 0].cpu().numpy(), points[0, 2*580:, 1].cpu().numpy(), points[0, 2*580:, 2].cpu().numpy(), c='red', label='Goal Point Cloud')
                # for i in range(580):
                #     ax.quiver(points[0, i, 0].cpu().numpy(), points[0, i, 1].cpu().numpy(), points[0, i, 2].cpu().numpy(),
                #             flow_action[0, i, 0].cpu().numpy(), flow_action[0, i, 1].cpu().numpy(), flow_action[0, i, 2].cpu().numpy(),
                #             color='green', arrow_length_ratio=0.1)
                # ax.set_xlabel('X')
                # ax.set_ylabel('Y')
                # ax.set_zlabel('Z')
                # ax.legend()
                # ax.set_title('Point Clouds and Relative Flow')
                # plt.show()
                # exit()
            else:
                pn_feat = self.extractor(points[:, :n_action_pcd+n_anchor_pcd, :])
        
        elif self.extractor_mode == "incremental":
            B, N, C = points.shape  # 256, 1740, 3
            rgb_obs_flatten = points.reshape(-1, C)
            rgb_features_flatten = self.nets['vision_encoder'](rgb_obs_flatten)
            rgb_features = rgb_features_flatten.reshape(B, N, -1)  # shape B N encoder_output_dim
            rgb_features = einops.rearrange(rgb_features, "B N encoder_output_dim -> N B encoder_output_dim") # 1740, 256, 64
            
            point_cloud_rel_pos_embedding = self.nets['relative_pe_layer'](points)[:, :, :rgb_features.shape[-1], :] # 256, 1740, 64, 2

            # cross attention between action pcd and anchor pcd
            attn_output = self.nets['attn_layers'](
                query=rgb_features[:580], value=rgb_features[580:1160],
                query_pos=point_cloud_rel_pos_embedding[:, :580], value_pos=point_cloud_rel_pos_embedding[:, 580:1160],
            )[-1] # 580, 256, 64

            # self attention
            self_attn_output = self.nets['self_attn_layers'](
                query=attn_output, value=attn_output,
                query_pos=point_cloud_rel_pos_embedding[:, :580], value_pos=point_cloud_rel_pos_embedding[:, :580],
            )[-1] # 580, 256, 64
            # new_rgb_features = einops.rearrange(
            #     self_attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1) # 256, 37120
            
            # cross attention between action pcd and goal pcd
            goal_attn_output = self.nets['goal_attn_layers'](
                query=self_attn_output, value=rgb_features[1160:],
                query_pos=point_cloud_rel_pos_embedding[:, :580], value_pos=point_cloud_rel_pos_embedding[:, 1160:],
            )[-1] # 580, 256, 64

            # self attention
            goal_self_attn_output = self.nets['goal_self_attn_layers'](
                query=goal_attn_output, value=goal_attn_output,
                query_pos=point_cloud_rel_pos_embedding[:, :580], value_pos=point_cloud_rel_pos_embedding[:, :580],
            )[-1] # 580, 256, 64
            pn_feat = einops.rearrange(
                goal_self_attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1) # 256, 37120
            # pn_feat = torch.cat([new_rgb_features, new_goal_features], dim=-1)
            
            pn_feat = self.nets['final_layer'](pn_feat)
        
        elif self.extractor_mode == "all":
            raise NotImplementedError
        
        else:
            raise NotImplementedError
            
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels