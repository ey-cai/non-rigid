import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from diffusers import get_cosine_schedule_with_warmup
from pytorch3d.transforms import Transform3d
from torch import nn, optim

from non_rigid.metrics.flow_metrics import flow_cos_sim, flow_rmse
from non_rigid.models.dit.models import DiT_PointCloud_Cross, LinearRegressionModel
from non_rigid.utils.logging_utils import viz_predicted_vs_gt
from non_rigid.utils.pointcloud_utils import expand_pcd


def LinearRegression_xS(**kwargs):
    return LinearRegressionModel(depth=5, hidden_size=132, num_heads=4, **kwargs)


def DiT_PointCloud_Cross_xS(**kwargs):
    return DiT_PointCloud_Cross(depth=5, hidden_size=132, num_heads=4, **kwargs)


models = {
    "linear_regression_xS": LinearRegression_xS,
    "regression_xS": DiT_PointCloud_Cross_xS,
}


def get_model(model_cfg):
    model_name = f"{model_cfg.name}_{model_cfg.size}"
    return models[model_name]


class LinearRegression(nn.Module):
    def __init__(
        self,
        model_cfg=None,
    ):
        super().__init__()
        self.model = get_model(model_cfg)(
            in_channels=model_cfg.in_channels,
            model_cfg=model_cfg,
        )

    def forward(self, x, y):
        return self.model(x=x, y=y)


class LinearRegressionTrainingModule(L.LightningModule):
    def __init__(self, network, training_cfg, model_cfg):
        super().__init__()
        self.network = network
        self.training_cfg = training_cfg
        self.model_cfg = model_cfg

        self.lr = training_cfg.lr
        self.weight_decay = training_cfg.weight_decay

        self.num_training_steps = training_cfg.num_training_steps
        self.lr_warmup_steps = training_cfg.lr_warmup_steps

        self.batch_size = training_cfg.batch_size
        self.val_batch_size = training_cfg.val_batch_size
        self.sample_size = training_cfg.sample_size
        self.sample_size_anchor = training_cfg.sample_size_anchor

    def forward(self, batch, mode="train"):
        # Extract point clouds from batch
        pos = batch["pc"].permute(0, 2, 1)  # (B, C, N)
        pc_action = batch["pc_action"].permute(0, 2, 1)  # (B, C, N)
        pc_anchor = batch["pc_anchor"].permute(0, 2, 1)  # (B, C, N)

        # Setup data required by the model
        model_kwargs = dict(x=pc_action, y=pc_anchor)
        pred_pos = self.network(**model_kwargs)

        loss = F.mse_loss(pred_pos, pos)
        return None, loss

    @torch.no_grad()
    def predict(
        self,
        batch,
    ):
        pos = batch["pc"].to(self.device)
        pc_action = batch["pc_action"].to(self.device)
        pc_anchor = batch["pc_anchor"].to(self.device)
        # reshaping
        pc_action = pc_action.permute(0, 2, 1)
        pc_anchor = pc_anchor.permute(0, 2, 1)

        pred_pos = self.network(x=pc_action, y=pc_anchor)
        pred_pos = pred_pos.permute(0, 2, 1)

        cos_sim = flow_cos_sim(pred_pos, pos, mask=False, seg=None)
        rmse = flow_rmse(pred_pos, pos, mask=False, seg=None)

        return {
            "pred_action": pred_pos,
            "cos_sim": cos_sim,
            "rmse": rmse,
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=self.num_training_steps,
            num_warmup_steps=self.lr_warmup_steps,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        self.train()
        _, loss = self(batch, "train")
        ##################################
        # Logging
        ##################################
        self.log_dict(
            {
                f"train/loss": loss,
            },
            add_dataloader_idx=False,
            prog_bar=True,
        )

        do_additional_logging = (
            self.global_step % self.training_cfg.additional_train_logging_period == 0
        )

        # Additional logging
        if do_additional_logging and False:
            pred_dict = self.predict(batch)
            pred_action = pred_dict["pred_action"]
            cos_sim = pred_dict["cos_sim"]
            rmse = pred_dict["rmse"]

            self.log_dict(
                {
                    "train/cos_sim": cos_sim.mean(),
                    "train/rmse": rmse.mean(),
                },
                add_dataloader_idx=False,
                prog_bar=True,
            )

            viz_idx = np.random.randint(0, batch["pc"].shape[0])
            pc_pos_viz = batch["pc"][viz_idx, :, :3]
            pc_action_viz = batch["pc_action"][viz_idx, :, :3]
            pc_anchor_viz = batch["pc_anchor"][viz_idx, :, :3]
            pred_action_viz = pred_action[viz_idx, :, :3]

            # Get predicted vs ground truth visualization
            predicted_vs_gt_viz = viz_predicted_vs_gt(
                pc_pos_viz=pc_pos_viz,
                pc_action_viz=pc_action_viz,
                pc_anchor_viz=pc_anchor_viz,
                pred_action_viz=pred_action_viz,
            )
            wandb.log({"train/predicted_vs_gt": predicted_vs_gt_viz})

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        with torch.no_grad():
            pred_dict = self.predict(batch)
            pred_action = pred_dict["pred_action"]
            cos_sim = pred_dict["cos_sim"]
            rmse = pred_dict["rmse"]

        # duplicatin wta metrics to prevent errors
        self.log_dict(
            {
                f"val_cos_sim_{dataloader_idx}": cos_sim.mean(),
                f"val_rmse_{dataloader_idx}": rmse.mean(),
                f"val_wta_cos_sim_{dataloader_idx}": cos_sim.mean(),
                f"val_wta_rmse_{dataloader_idx}": rmse.mean(),
            },
            add_dataloader_idx=False,
            prog_bar=True,
        )

        viz_idx = np.random.randint(0, batch["pc"].shape[0])
        pc_pos_viz = batch["pc"][viz_idx, :, :3]
        pc_action_viz = batch["pc_action"][viz_idx, :, :3]
        pc_anchor_viz = batch["pc_anchor"][viz_idx, :, :3]
        pred_action_viz = pred_action[viz_idx, :, :3]

        # Get predicted vs ground truth visualization
        predicted_vs_gt_viz = viz_predicted_vs_gt(
            pc_pos_viz=pc_pos_viz,
            pc_action_viz=pc_action_viz,
            pc_anchor_viz=pc_anchor_viz,
            pred_action_viz=pred_action_viz,
        )
        wandb.log({f"val_{dataloader_idx}/predicted_vs_gt": predicted_vs_gt_viz})

        return {
            "cos_sim": cos_sim,
            "loss": rmse,
        }

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError(
            "Predict step not implemented for LinearRegressionTrainingModule"
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError(
            "Test step not implemented for LinearRegressionTrainingModule"
        )


class LinearRegressionInferenceModule(L.LightningModule):
    def __init__(
        self,
        network,
        inference_cfg,
        model_cfg,
    ):
        super().__init__()
        self.network = network
        self.inference_cfg = inference_cfg
        self.model_cfg = model_cfg

        self.batch_size = inference_cfg.batch_size
        self.val_batch_size = inference_cfg.val_batch_size
        self.sample_size = inference_cfg.sample_size
        self.sample_size_anchor = inference_cfg.sample_size_anchor

    def forward(self, batch):
        raise NotImplementedError(
            "Inference module should not use forward method - use 'predict' instead."
        )

    @torch.no_grad()
    def predict(
        self,
        batch,
        num_samples,
        progress=False,
    ):
        # TODO: add in num_samples as a dummy parameter
        pc_action = batch["pc_action"].to(self.device)
        pc_anchor = batch["pc_anchor"].to(self.device)
        # reshaping
        pc_action = pc_action.permute(0, 2, 1)
        pc_anchor = pc_anchor.permute(0, 2, 1)

        pred_action = self.network(x=pc_action, y=pc_anchor)
        pred_action = pred_action.permute(0, 2, 1)

        # expanding point clouds
        pc_action = expand_pcd(pc_action, num_samples)
        pred_action = expand_pcd(pred_action, num_samples)

        T_action2world = Transform3d(
            matrix=expand_pcd(batch["T_action2world"].to(self.device), num_samples),
        )
        T_goal2world = Transform3d(
            matrix=expand_pcd(batch["T_goal2world"].to(self.device), num_samples),
        )
        # computing pred flow in world frame
        pc_action = pc_action.permute(0, 2, 1)
        pred_flow = T_goal2world.transform_points(
            pred_action
        ) - T_action2world.transform_points(pc_action)
        return {
            "pred_action": pred_action,
            "pred_world_flow": pred_flow,
            "pred_world_action": T_goal2world.transform_points(pred_action),
        }

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pos = batch["pc"].to(self.device)
        pred_dict = self.predict(batch, 1)
        pred_action = pred_dict["pred_action"]

        # computing errors
        cos_sim = flow_cos_sim(pred_action, pos, mask=False, seg=None)
        rmse = flow_rmse(pred_action, pos, mask=False, seg=None)
        # duplicating WTA metrics to prevent errors
        return {
            "cos_sim": cos_sim,
            "rmse": rmse,
            "cos_sim_wta": cos_sim,
            "rmse_wta": rmse,
        }


class RegressionNetwork(nn.Module):
    """
    Network built on the Diffusion Transformer architecture, but adapted for regressive inference. All
    architecture design is kept the same, but the input time-step is fixed to 0, and the reverse
    diffusion process is skipped.
    """

    def __init__(self, model_cfg=None):
        super().__init__()
        self.model = get_model(model_cfg)(
            in_channels=model_cfg.in_channels,
            learn_sigma=model_cfg.learn_sigma,
            model_cfg=model_cfg,
        )

    def forward(self, x, y):
        t = torch.zeros(x.shape[0]).to(x.device)
        return self.model(x=x, y=y, t=t)


# consolidating regression modules into one
class RegressionModule(L.LightningModule):
    def __init__(self, network, cfg) -> None:
        super().__init__()
        self.network = network
        self.model_cfg = cfg.model
        self.prediction_type = self.model_cfg.type  # flow or point
        self.mode = cfg.mode  # train or eval

        # prediction type-specific processing
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
            self.additional_train_logging_period = (
                self.run_cfg.additional_train_logging_period
            )
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

    def forward(self, batch):
        """
        Forward pass to compute training loss.
        """
        ground_truth = batch[self.label_key].permute(0, 2, 1)  # channel first
        pc_action = batch["pc_action"].permute(0, 2, 1)  # channel first
        pc_anchor = batch["pc_anchor"].permute(0, 2, 1)  # channel first

        model_kwargs = dict(x=pc_action, y=pc_anchor)
        pred = self.network(**model_kwargs)
        loss = F.mse_loss(pred, ground_truth)
        return None, loss

    @torch.no_grad()
    def predict(
        self, batch, num_samples, unflatten=False, progress=False, full_prediction=True
    ):
        pc_action = batch["pc_action"].to(self.device)
        pc_anchor = batch["pc_anchor"].to(self.device)
        # reshaping
        pc_action = pc_action.permute(0, 2, 1)
        pc_anchor = pc_anchor.permute(0, 2, 1)
        pred = self.network(x=pc_action, y=pc_anchor)

        pred = pred.permute(0, 2, 1)
        pred = expand_pcd(pred, num_samples)

        if not full_prediction:
            return {self.prediction_type: {"pred": pred}}
        else:
            # return full prediction (flow and point, goal and world frame)
            pc_action = pc_action.permute(0, 2, 1)
            pc_action = expand_pcd(pc_action, num_samples)

            # computing flow and point predictions
            if self.prediction_type == "flow":
                pred_flow = pred
                pred_point = pc_action + pred_flow
            elif self.prediction_type == "point":
                pred_point = pred
                pred_flow = pred_point - pc_action

            pred_dict = {
                "flow": {
                    "pred": pred_flow,
                },
                "point": {
                    "pred": pred_point,
                },
            }

            # compute world frame predictions
            T_action2world = Transform3d(
                matrix=expand_pcd(batch["T_action2world"].to(self.device), num_samples),
            )
            T_goal2world = Transform3d(
                matrix=expand_pcd(batch["T_goal2world"].to(self.device), num_samples),
            )

            pred_point_world = T_goal2world.transform_points(pred_point)
            pc_action_world = T_action2world.transform_points(pc_action)
            pred_flow_world = pred_point_world - pc_action_world

            pred_dict["flow"]["pred_world"] = pred_flow_world
            pred_dict["point"]["pred_world"] = pred_point_world
            # trivialy, results_world is just the point prediction in world frame
            pred_dict["results_world"] = pred_point_world
            return pred_dict

    def predict_wta(self, batch, num_samples):
        ground_truth = batch[self.label_key].to(self.device)
        seg = batch["seg"].to(self.device)

        # re-shaping and expanding for winner-take-all
        bs = ground_truth.shape[0]
        ground_truth = expand_pcd(ground_truth, num_samples)
        seg = expand_pcd(seg, num_samples)

        # generating predictions
        pred_dict = self.predict(
            batch, num_samples, unflatten=False, progress=True, full_prediction=False
        )
        pred = pred_dict[self.prediction_type]["pred"]

        # computing error metrics
        rmse = flow_rmse(pred, ground_truth, mask=True, seg=seg).reshape(
            bs, num_samples
        )
        pred = pred.reshape(bs, num_samples, -1, 3)

        # computing winner-take-all predictions
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
        Log visualizations to wandb. Since this is a regression module, no need to
        log WTA visualizations.

        Args:
            batch: the input batch
            pred_wta_dict: the prediction dictionary
            tag: the tag to use for logging
        """
        # pick a random sample in the batch to visualize
        viz_idx = np.random.randint(0, batch["pc"].shape[0])
        pred_viz = pred_wta_dict["pred"][viz_idx, 0, :, :3]
        # getting viz args
        pc_pos_viz = batch["pc"][viz_idx, :, :3]
        pc_action_viz = batch["pc_action"][viz_idx, :, :3]
        pc_anchor_viz = batch["pc_anchor"][viz_idx, :, :3]
        viz_args = {
            "pc_pos_viz": pc_pos_viz,
            "pc_action_viz": pc_action_viz,
            "pc_anchor_viz": pc_anchor_viz,
        }

        # getting predicted action point cloud
        if self.prediction_type == "flow":
            pred_action_viz = viz_args["pc_action_viz"] + pred_viz
        elif self.prediction_type == "point":
            pred_action_viz = pred_viz

        # logging predicted vs ground truth point cloud
        viz_args["pred_action_viz"] = pred_action_viz
        predicted_vs_gt = viz_predicted_vs_gt(**viz_args)
        wandb.log({f"{tag}/predicted_vs_gt": predicted_vs_gt})

    def training_step(self, batch):
        """
        Training step for the module. Logs training metrics and visualizations to wandb. Since this
        is a regression module, we sample only 1 prediction - no need to log WTA metrics.
        """
        self.train()
        _, loss = self(batch)
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
        if do_additional_logging and False:
            # winner-take-all predictions
            pred_wta_dict = self.predict_wta(batch, num_samples=1)

            ####################################################
            # logging training wta metrics
            ####################################################
            self.log_dict(
                {
                    "train/rmse": pred_wta_dict["rmse"].mean(),
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
        Validation step for the module. Logs validation metrics and visualizations to wandb. Since this
        is a regression module, we sample only 1 prediction - still logging WTA metrics for checkpoint
        callback.
        """
        self.eval()
        with torch.no_grad():
            # winner-take-all predictions
            pred_wta_dict = self.predict_wta(batch, 1)

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
        Prediction step for model evaluation. Computes winner-take-all metrics. Since this
        is a regression module, we sample only 1 prediction - still return WTA metrics
        for compatibility with evaluation scripts.
        """
        # winner-take-all predictions
        pred_wta_dict = self.predict_wta(batch, 1)
        return {
            "rmse": pred_wta_dict["rmse"],
            "rmse_wta": pred_wta_dict["rmse_wta"],
        }
