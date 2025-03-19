import numpy as np
import torch
from non_rigid.nets.pn2 import PN2Dense
from non_rigid.nets.dgcnn import DGCNN

from non_rigid.models.dit.models import DiT_PointCloud_Cross
from omegaconf import OmegaConf


class FramePredictorSimple(torch.nn.Module):
    def __init__(self, out_channels):
        super(FramePredictorSimple, self).__init__()
        self.out_channels = out_channels
        self.in_channels = 3
        self.pn2 = PN2Dense(in_channels=3, out_channels=out_channels)

    def forward(self, batch):
        # make sure all of the point clouds in the batch have the same number of points
        ptr_diffs = torch.unique(batch.ptr[1:] - batch.ptr[:-1])
        if len(ptr_diffs) > 1:
            raise ValueError("All point clouds in the batch must have the same number of points.")
        else:
            num_points = ptr_diffs.item()

        output = self.pn2(batch)
        # reshape output to get (B, N, C) shape
        output = output.reshape(-1, num_points, self.out_channels)

        logits = output[..., [0]]
        residuals = output[..., 1:4]
        vars = output[..., 4:]

        # run vars through softplus to ensure positive values
        vars = torch.nn.functional.softplus(vars)

        # add mean residuals to points to get mean predictions
        means = residuals + batch.pos.reshape(-1, num_points, 3)

        # converting logits to probabilities
        probs = torch.softmax(logits, dim=1)

        return {
            "probs": probs,
            "means": means,
            "vars": vars,
        }
    
class FramePredictorDGCNN(torch.nn.Module):
    def __init__(self, out_channels):
        super(FramePredictorDGCNN, self).__init__()
        self.out_channels = out_channels
        self.dgcnn = DGCNN(emb_dims=512)
        self.final = torch.nn.Conv1d(
            in_channels=512,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, batch):
        # make sure all of the point clouds in the batch have the same number of points
        ptr_diffs = torch.unique(batch.ptr[1:] - batch.ptr[:-1])
        if len(ptr_diffs) > 1:
            raise ValueError("All point clouds in the batch must have the same number of points.")
        else:
            num_points = ptr_diffs.item()
        
        input = batch.pos.reshape(-1, num_points, 3).permute(0, 2, 1)
        output = self.dgcnn(input)
        output = self.final(output)
        # reshape output to get (B, N, C) shape
        output = output.permute(0, 2, 1)

        logits = output[..., [0]]
        residuals = output[..., 1:4]
        vars = output[..., 4:]

        # run vars through softplus to ensure positive values
        vars = torch.nn.functional.softplus(vars)

        # add mean residuals to points to get mean predictions
        means = residuals + batch.pos.reshape(-1, num_points, 3)

        # converting logits to probabilities
        probs = torch.softmax(logits, dim=1)

        return {
            "probs": probs,
            "means": means,
            "vars": vars,
        }

class FramePredictorMLPTransformer(torch.nn.Module):
    def __init__(self):
        super(FramePredictorMLPTransformer, self).__init__()
        model_cfg = {
            "rotary": False,
            "x_encoder": "mlp",
            "x0_encoder": None,
            "y_encoder": "mlp",
            "extra_features": False,
            "rel_pose": True,
            "rel_pose_type": "translation",
            "center_noise": False,
        }
        # omegaconf from dict
        model_cfg = OmegaConf.create(model_cfg)
        self.model = DiT_PointCloud_Cross(
            depth=5, hidden_size=128, num_heads=4, in_channels=3, learn_sigma=True, model_cfg=model_cfg
        )
    
    def forward(self, batch):
        # make sure all of the point clouds in the batch have the same number of points
        ptr_diffs = torch.unique(batch.ptr[1:] - batch.ptr[:-1])
        if len(ptr_diffs) > 1:
            raise ValueError("All point clouds in the batch must have the same number of points.")
        else:
            num_points = ptr_diffs.item()
        
        # assume action and anchor are the same size
        pc_anchor = batch.pos.reshape(-1, num_points, 3).permute(0, 2, 1)
        pc_action = batch.action.reshape(-1, num_points, 3).permute(0, 2, 1)
        bs = pc_anchor.shape[0]
        rel_pose = batch.rel_pose.reshape(bs, -1)
        model_kwargs = {
            "x": pc_anchor, # anchor
            "y": pc_action, # action
            "t": torch.zeros(bs).to(pc_anchor.device), # zeros
            "rel_pose": rel_pose,# relpose
        }
        output = self.model(**model_kwargs).permute(0, 2, 1)

        logits = output[..., [0]]
        residuals = output[..., 1:4]
        vars = output[..., 4:5] # ignore last output

        # run vars through softplus to ensure positive values
        vars = torch.nn.functional.softplus(vars)

        # add mean residuals to points to get mean predictions
        means = residuals + batch.pos.reshape(-1, num_points, 3)

        # converting logits to probabilities
        probs = torch.softmax(logits, dim=1)

        return {
            "probs": probs,
            "means": means,
            "vars": vars,
        }