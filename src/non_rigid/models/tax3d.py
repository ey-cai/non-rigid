from typing import Any, Dict

import lightning as L
import numpy as np
import omegaconf
import plotly.express as px
import rpad.pyg.nets.dgcnn as dgcnn
import rpad.visualize_3d.plots as vpl
import torch
import torch.nn.functional as F
import torch_geometric.data as tgd
import torch_geometric.transforms as tgt
import torchvision as tv
import wandb
from diffusers import get_cosine_schedule_with_warmup
from pytorch3d.transforms import Transform3d, Translate
from pytorch3d.transforms import matrix_to_quaternion, matrix_to_rotation_6d
from torch import nn, optim
from torch_geometric.nn import fps

from non_rigid.metrics.error_metrics import get_pred_pcd_rigid_errors
from non_rigid.metrics.flow_metrics import flow_cos_sim, flow_rmse, pc_nn
from non_rigid.models.dit.diffusion import create_diffusion
# from non_rigid.models.dit.models import DiT_PointCloud_Unc as DiT_pcu
from non_rigid.models.dit.models import (
    DiT_PointCloud_Unc_Cross,
    Rel3D_DiT_PointCloud_Unc_Cross,
    DiT_PointCloud_Cross,
    DiT_PointCloud
)
from non_rigid.utils.logging_utils import viz_predicted_vs_gt
from non_rigid.utils.pointcloud_utils import expand_pcd


def DiT_pcu_S(**kwargs):
    return DiT_pcu(depth=12, hidden_size=384, num_heads=6, **kwargs)


def DiT_pcu_xS(**kwargs):
    return DiT_pcu(depth=5, hidden_size=128, num_heads=4, **kwargs)


def DiT_pcu_cross_xS(**kwargs):
    return DiT_PointCloud_Unc_Cross(depth=5, hidden_size=128, num_heads=4, **kwargs)


def Rel3D_DiT_pcu_cross_xS(**kwargs):
    # Embed dim divisible by 3 for 3D positional encoding and divisible by num_heads for multi-head attention
    return Rel3D_DiT_PointCloud_Unc_Cross(
        depth=5, hidden_size=132, num_heads=4, **kwargs
    )

def DiT_PointCloud_Cross_xS(use_rotary, **kwargs):
    # hidden size divisible by 3 for rotary embedding, and divisible by num_heads for multi-head attention
    hidden_size = 132 if use_rotary else 128
    return DiT_PointCloud_Cross(depth=5, hidden_size=hidden_size, num_heads=4, **kwargs)

def DiT_PointCloud_xS(use_rotary, **kwargs):
    # hidden size divisible by 3 for rotary embedding, and divisible by num_heads for multi-head attention
    hidden_size = 132 if use_rotary else 128
    return DiT_PointCloud(depth=5, hidden_size=hidden_size, num_heads=4, **kwargs)

# TODO: clean up all unused functions
DiT_models = {
    "DiT_pcu_S": DiT_pcu_S,
    "DiT_pcu_xS": DiT_pcu_xS,
    "DiT_pcu_cross_xS": DiT_pcu_cross_xS,
    "Rel3D_DiT_pcu_cross_xS": Rel3D_DiT_pcu_cross_xS,
    # there is no Rel3D_DiT_pcu_xS
    "DiT_PointCloud_Cross_xS": DiT_PointCloud_Cross_xS,
    # TODO: add the SD model here
    "DiT_PointCloud_xS": DiT_PointCloud_xS,
}


def get_model(model_cfg):
    #rotary = "Rel3D_" if model_cfg.rotary else ""
    cross = "Cross_" if model_cfg.name == "df_cross" else ""
    # model_name = f"{rotary}DiT_pcu_{cross}{model_cfg.size}"
    model_name = f"DiT_PointCloud_{cross}{model_cfg.size}"
    return DiT_models[model_name]


class DiffusionTransformerNetwork(nn.Module):
    """
    Network containing the specified Diffusion Transformer architecture.
    """
    def __init__(self, model_cfg=None):
        super().__init__()
        self.dit = get_model(model_cfg)(
            use_rotary=model_cfg.rotary,
            in_channels=model_cfg.in_channels,
            learn_sigma=model_cfg.learn_sigma,
            model_cfg=model_cfg,
        )
    
    def forward(self, x, t, **kwargs):
        return self.dit(x, t, **kwargs)
    


class DenseDisplacementDiffusionModule(L.LightningModule):
    """
    Generalized Dense Displacement Diffusion (DDD) module that handles model training, inference, 
    evaluation, and visualization. This module is inherited and overriden by scene-level and 
    object-centric modules.
    """
    def __init__(self, network, cfg) -> None:
        super().__init__()
        self.network = network
        self.model_cfg = cfg.model
        self.prediction_type = self.model_cfg.type # flow or point
        self.mode = cfg.mode # train or eval

        # prediction type-specific processing
        # TODO: eventually, this should be removed by updating dataset to use "point" instead of "pc"
        if self.prediction_type == "flow":
            self.label_key = "flow"
        elif self.prediction_type == "point":
            self.label_key = "pc"
        else:
            raise ValueError(f"Invalid prediction type: {self.prediction_type}")
        
        # mode-specific processing
        if self.mode == "train":
            self.run_cfg = cfg.training
            # training-specific params
            self.lr = self.run_cfg.lr
            self.weight_decay = self.run_cfg.weight_decay
            self.num_training_steps = self.run_cfg.num_training_steps
            self.lr_warmup_steps = self.run_cfg.lr_warmup_steps
            self.additional_train_logging_period = self.run_cfg.additional_train_logging_period
        elif self.mode == "eval":
            self.run_cfg = cfg.inference
            # inference-specific params
            self.num_trials = self.run_cfg.num_trials
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # data params
        self.batch_size = self.run_cfg.batch_size
        self.val_batch_size = self.run_cfg.val_batch_size
        # TODO: it is debatable if the module needs to know about the sample size
        self.sample_size = self.run_cfg.sample_size
        self.sample_size_anchor = self.run_cfg.sample_size_anchor

        # diffusion params
        # self.noise_schedule = model_cfg.diff_noise_schedule
        # self.noise_scale = model_cfg.diff_noise_scale
        self.diff_steps = self.model_cfg.diff_train_steps # TODO: rename to diff_steps
        self.num_wta_trials = self.run_cfg.num_wta_trials
        self.diffusion = create_diffusion(
            timestep_respacing=None,
            diffusion_steps=self.diff_steps,
            # noise_schedule=self.noise_schedule,
        )

    def configure_optimizers(self):
        assert self.mode == "train", "Can only configure optimizers in training mode."
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return [optimizer], [lr_scheduler]

    def get_model_kwargs(self, batch, nun_samples=None):
        """
        Get the model kwargs for the forward pass.
        """
        raise NotImplementedError("This should be implemented in the derived class.")
    
    def get_world_preds(self, batch, num_samples, pc_action, pred_dict):
        """
        Get world frame predictions from the given batch and predictions.
        """
        raise NotImplementedError("This should be implemented in the derived class.")
    
    def get_viz_args(self, batch, viz_idx):
        """
        Get visualization arguments for wandb logging.
        """
        raise NotImplementedError("This should be implemented in the derived class.")

    def forward(self, batch, t):
        """
        Forward pass to compute diffusion training loss.
        """
        ground_truth = batch[self.label_key].permute(0, 2, 1) # channel first
        model_kwargs = self.get_model_kwargs(batch)

        # run diffusion
        # noise = torch.randn_like(ground_truth) * self.noise_scale
        loss_dict = self.diffusion.training_losses(
            model=self.network,
            x_start=ground_truth,
            t=t,
            model_kwargs=model_kwargs,
            # noise=noise,
        )
        loss = loss_dict["loss"].mean()
        return None, loss

    @torch.no_grad()
    def predict(self, batch, num_samples, unflatten=False, progress=True, full_prediction=True):
        """
        Compute prediction for a given batch.

        Args:
            batch: the input batch
            num_samples: the number of samples to generate
            progress: whether to show progress bar
            full_prediction: whether to return full prediction (flow and point, goal and world frame)
        """
        # TODO: replace bs with batch_size?
        bs, sample_size = batch["pc_action"].shape[:2]
        model_kwargs = self.get_model_kwargs(batch, num_samples)

        # generating latents and running diffusion
        z = torch.randn(bs * num_samples, 3, sample_size, device=self.device)
        pred, results = self.diffusion.p_sample_loop(
            self.network,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress,
            device=self.device,
        )
        pred = pred.permute(0, 2, 1)

        if not full_prediction:
            # only return the prediction type in the goal frame
            return {self.prediction_type: {"pred": pred}}
        else:
            # return full prediction (flow and point, goal and world frame)
            pc_action = model_kwargs["x0"].permute(0, 2, 1)

            # computing flow and point predictions
            if self.prediction_type == "flow":
                pred_flow = pred
                pred_point = pc_action + pred_flow
                # for flow predictions, convert results to point predictions
                results = [
                    pc_action + res.permute(0, 2, 1) for res in results
                ]
            elif self.prediction_type == "point":
                pred_point = pred
                pred_flow = pred_point - pc_action
                results = [
                    res.permute(0, 2, 1) for res in results
                ]

            pred_dict = {
                "flow": {
                    "pred": pred_flow,
                },
                "point": {
                    "pred": pred_point,
                },
                "results": results,
            }

            # compute world frame predictions
            pred_flow_world, pred_point_world, results_world = self.get_world_preds(
                batch, num_samples, pc_action, pred_dict
            )
            pred_dict["flow"]["pred_world"] = pred_flow_world
            pred_dict["point"]["pred_world"] = pred_point_world
            pred_dict["results_world"] = results_world
            return pred_dict

    def predict_wta(self, batch, num_samples):
        """
        Predict WTA (winner-take-all) samples, and compute WTA metrics. Unlike predict, this 
        function assumes the ground truth is available.

        Args:
            batch: the input batch
            num_samples: the number of samples to generate
        """
        ground_truth = batch[self.label_key].to(self.device)
        seg = batch["seg"].to(self.device)

        # re-shaping and expanding for winner-take-all
        bs = ground_truth.shape[0]
        ground_truth = expand_pcd(ground_truth, num_samples)
        seg = expand_pcd(seg, num_samples)

        # generating diffusion predictions
        # TODO: this should probably specific full_prediction=False
        pred_dict = self.predict(
            batch, num_samples, unflatten=False, progress=True
        )
        pred = pred_dict[self.prediction_type]["pred"]

        # computing error metrics
        rmse = flow_rmse(pred, ground_truth, mask=True, seg=seg).reshape(bs, num_samples)
        pred = pred.reshape(bs, num_samples, -1, 3)

        # computing winner-take-all metrics
        winner = torch.argmin(rmse, dim=-1)
        rmse_wta = rmse[torch.arange(bs), winner]
        pred_wta = pred[torch.arange(bs), winner]
        return {
            "pred": pred,
            "pred_wta": pred_wta,
            "rmse": rmse,
            "rmse_wta": rmse_wta,
        }

    def log_viz_to_wandb(self, batch, pred_wta_dict, tag):
        """
        Log visualizations to wandb.

        Args:
            batch: the input batch
            pred_wta_dict: the prediction dictionary
            tag: the tag to use for logging
        """
        # pick a random sample in the batch to visualize
        viz_idx = np.random.randint(0, batch["pc"].shape[0])
        pred_viz = pred_wta_dict["pred"][viz_idx, 0, :, :3]
        pred_wta_viz = pred_wta_dict["pred_wta"][viz_idx, :, :3]
        viz_args = self.get_viz_args(batch, viz_idx)

        # getting predicted action point cloud
        if self.prediction_type == "flow":
            pred_action_viz = viz_args["pc_action_viz"] + pred_viz
            pred_action_wta_viz = viz_args["pc_action_viz"] + pred_wta_viz
        elif self.prediction_type == "point":
            pred_action_viz = pred_viz
            pred_action_wta_viz = pred_wta_viz

        # logging predicted vs ground truth point cloud
        viz_args["pred_action_viz"] = pred_action_viz
        predicted_vs_gt = viz_predicted_vs_gt(**viz_args)
        wandb.log({f"{tag}/predicted_vs_gt": predicted_vs_gt})

        # logging predicted vs ground truth point cloud (wta)
        viz_args["pred_action_viz"] = pred_action_wta_viz
        predicted_vs_gt_wta = viz_predicted_vs_gt(**viz_args)
        wandb.log({f"{tag}/predicted_vs_gt_wta": predicted_vs_gt_wta})

    def training_step(self, batch):
        """
        Training step for the module. Logs training metrics and visualizations to wandb.
        """
        self.train()
        t = torch.randint(
            0, self.diff_steps, (self.batch_size,), device=self.device
        ).long()
        _, loss = self(batch, t)
        #########################################################
        # logging training metrics
        #########################################################
        self.log_dict(
            {"train/loss": loss},
            add_dataloader_idx=False,
            prog_bar=True,
        )

        # determine if additional logging should be done
        do_additional_logging = (
            self.global_step % self.additional_train_logging_period == 0
        )

        # additional logging
        if do_additional_logging:
            # winner-take-all predictions
            pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)

            ####################################################
            # logging training wta metrics
            ####################################################
            self.log_dict(
                {
                    "train/rmse": pred_wta_dict["rmse"].mean(),
                    "train/rmse_wta": pred_wta_dict["rmse_wta"].mean(),
                },
                add_dataloader_idx=False,
                prog_bar=True,
            )

            ####################################################
            # logging visualizations
            ####################################################
            self.log_viz_to_wandb(batch, pred_wta_dict, "train")

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step for the module. Logs validation metrics and visualizations to wandb.
        """
        self.eval()
        with torch.no_grad():
            # winner-take-all predictions
            pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)
        
        ####################################################
        # logging validation wta metrics
        ####################################################
        self.log_dict(
            {
                f"val_rmse_{dataloader_idx}": pred_wta_dict["rmse"].mean(),
                f"val_rmse_wta_{dataloader_idx}": pred_wta_dict["rmse_wta"].mean(),
            },
            add_dataloader_idx=False,
            prog_bar=True,
        )

        ####################################################
        # logging visualizations
        ####################################################
        self.log_viz_to_wandb(batch, pred_wta_dict, f"val_{dataloader_idx}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for model evaluation. Computes winner-take-all metrics.
        """
        # winner-take-all predictions
        pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)
        return {
            "rmse": pred_wta_dict["rmse"],
            "rmse_wta": pred_wta_dict["rmse_wta"],
        }
    
    def predict_obs(self, obs, run_cfg):
        """
        Predict for a single observation. Note: the input run_cfg is very different from self.run_cfg...sigh...
        """
        # TODO: right now, this needs to take the configs, becuase some model-specific configs 
        # are actually specified in the dataset config - eventually, these should be moved over 
        
        # process input observation based on the dataset and model configs
        action_pc = obs["pc_action"]
        anchor_pc = obs["pc_anchor"]
        action_seg = obs["seg"]
        anchor_seg = obs["seg_anchor"]

        if run_cfg.dataset.scene:
            # scene-level processing
            scene_pc = torch.cat([action_pc, anchor_pc], dim=1)
            scene_seg = torch.cat([action_seg, anchor_seg], dim=1)

            # center the point cloud
            scene_center = scene_pc.mean(dim=1)
            scene_pc = scene_pc - scene_center
            T_goal2world = Translate(scene_center).get_matrix()

            item = {
                "pc_action": scene_pc,
                "seg": scene_seg,
                "T_goal2world": T_goal2world,
            }
        else:
            # object-centric processing
            if run_cfg.dataset.world_frame:
                action_center = torch.zeros(3, dtype=torch.float32, device=self.device).unsqueeze(0)
                anchor_center = torch.zeros(3, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                action_center = action_pc.mean(dim=1)
                anchor_center = anchor_pc.mean(dim=1)
            
            # check for scene anchor
            if run_cfg.dataset.scene_anchor:
                anchor_pc = torch.cat([action_pc, anchor_pc], dim=1)
                anchor_seg = torch.cat([action_seg, anchor_seg], dim=1)
                # if scene anchor center, center the anchor point cloud in the scene
                if run_cfg.dataset.scene_anchor_center:
                    anchor_center = anchor_pc.mean(dim=1)
            
            # center the point clouds
            action_pc = action_pc - action_center
            anchor_pc = anchor_pc - anchor_center
            T_action2world = Translate(action_center)
            T_goal2world = Translate(anchor_center)

            item = {
                "pc_action": action_pc,
                "pc_anchor": anchor_pc,
                "seg": action_seg,
                "seg_anchor": anchor_seg,
                "T_action2world": T_action2world.get_matrix(),
                "T_goal2world": T_goal2world.get_matrix(),
            }

            # if relative action-anchor pose, add the relative transform
            if run_cfg.dataset.rel_pose:
                rel_pose = T_action2world.compose(T_goal2world.inverse())
                # converting relative pose based on representation type
                if run_cfg.dataset.rel_pose_type == "quaternion":
                    translation = rel_pose.get_matrix()[:, 3, :3]
                    rotation = matrix_to_quaternion(rel_pose.get_matrix()[:, :3, :3])
                    rel_pose = torch.cat([translation, rotation], dim=1)
                elif run_cfg.dataset.rel_pose_type == "rotation_6d":
                    translation = rel_pose.get_matrix()[:, 3, :3]
                    rotation = matrix_to_rotation_6d(rel_pose.get_matrix()[:, :3, :3])
                    rel_pose = torch.cat([translation, rotation], dim=1)
                elif run_cfg.dataset.rel_pose_type == "logmap":
                    rel_pose = rel_pose.get_se3_log()
                item["rel_pose"] = rel_pose

        pred_dict = self.predict(item, run_cfg.inference.num_trials, progress=False)
        pred_action = pred_dict["point"]["pred_world"]
        results_world = pred_dict["results_world"]

        # masking out non-action points in scene-level processing
        if run_cfg.dataset.scene:
            pred_action = pred_action[:, scene_seg.squeeze().bool(), :]
            results_world = [res[:, scene_seg.squeeze(0).bool(), :] for res in results_world]

        return pred_action, results_world


class SceneDisplacementModule(DenseDisplacementDiffusionModule):
    """
    Scene-level DDD module. Applies self-attention to the entire scene.
    """
    def __init__(self, network, cfg) -> None:
        super().__init__(network, cfg)

    def get_model_kwargs(self, batch, num_samples=None):
        pc_action = batch["pc_action"].to(self.device)
        if num_samples is not None:
            # expand point clouds if num_samples is provided; used for WTA predictions
            pc_action = expand_pcd(pc_action, num_samples)

        pc_action = pc_action.permute(0, 2, 1) # channel first
        model_kwargs = dict(x0=pc_action)
        return model_kwargs
    
    def get_world_preds(self, batch, num_samples, pc_action, pred_dict):
        """
        Get world-frame predictions from the given batch and predictions.
        """
        T_goal2world = Transform3d(
            matrix=expand_pcd(batch["T_goal2world"].to(self.device), num_samples)
        )

        pred_point_world = T_goal2world.transform_points(pred_dict["point"]["pred"])
        pc_action_world = T_goal2world.transform_points(pc_action)
        pred_flow_world = pred_point_world - pc_action_world
        results_world = [
            T_goal2world.transform_points(res) for res in pred_dict["results"]
        ]
        return pred_flow_world, pred_point_world, results_world
    
    def get_viz_args(self, batch, viz_idx):
        """
        Get visualization arguments for wandb logging.
        """
        pc_pos_viz = batch["pc"][viz_idx, :, :3]
        pc_action_viz = batch["pc_action"][viz_idx, :, :3]
        viz_args = {
            "pc_pos_viz": pc_pos_viz,
            "pc_action_viz": pc_action_viz,
        }
        return viz_args

class CrossDisplacementModule(DenseDisplacementDiffusionModule):
    """
    Object-centric DDD module. Applies cross attention between action and anchor objects.
    """
    def __init__(self, network, cfg) -> None:
        super().__init__(network, cfg)

    def get_model_kwargs(self, batch, num_samples=None):
        pc_action = batch["pc_action"].to(self.device)
        pc_anchor = batch["pc_anchor"].to(self.device)
        if num_samples is not None:
            # expand point clouds if num_samples is provided; used for WTA predictions
            pc_action = expand_pcd(pc_action, num_samples)
            pc_anchor = expand_pcd(pc_anchor, num_samples)
        
        pc_action = pc_action.permute(0, 2, 1) # channel first
        pc_anchor = pc_anchor.permute(0, 2, 1) # channel first
        model_kwargs = dict(x0=pc_action, y=pc_anchor)

        # check for relative action-anchor pose
        if self.model_cfg.rel_pose:
            rel_pose = batch["rel_pose"].to(self.device)
            if num_samples is not None:
                rel_pose = expand_pcd(rel_pose, num_samples)
            model_kwargs["rel_pose"] = rel_pose
        return model_kwargs
    
    def get_world_preds(self, batch, num_samples, pc_action, pred_dict):
        """
        Get world-frame predictions from the given batch and predictions.
        """
        T_action2world = Transform3d(
            matrix=expand_pcd(batch["T_action2world"].to(self.device), num_samples)
        )
        T_goal2world = Transform3d(
            matrix=expand_pcd(batch["T_goal2world"].to(self.device), num_samples)
        )

        pred_point_world = T_goal2world.transform_points(pred_dict["point"]["pred"])
        pc_action_world = T_action2world.transform_points(pc_action)
        pred_flow_world = pred_point_world - pc_action_world
        results_world = [
            T_goal2world.transform_points(res) for res in pred_dict["results"]
        ]
        return pred_flow_world, pred_point_world, results_world

    def get_viz_args(self, batch, viz_idx):
        """
        Get visualization arguments for wandb logging.
        """
        pc_pos_viz = batch["pc"][viz_idx, :, :3]
        pc_action_viz = batch["pc_action"][viz_idx, :, :3]
        pc_anchor_viz = batch["pc_anchor"][viz_idx, :, :3]
        viz_args = {
            "pc_pos_viz": pc_pos_viz,
            "pc_action_viz": pc_action_viz,
            "pc_anchor_viz": pc_anchor_viz,
        }
        return viz_args