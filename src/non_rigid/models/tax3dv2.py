import lightning as L
import numpy as np
import torch
from torch import nn, optim
import wandb

from diffusers import get_cosine_schedule_with_warmup
from pytorch3d.transforms import Transform3d, Translate

from non_rigid.metrics.flow_metrics import flow_rmse
from non_rigid.models.dit.diffusion import create_diffusion
from non_rigid.models.dit.models import PointCloudDiT2
from non_rigid.utils.logging_utils import viz_predicted_vs_gt
from non_rigid.utils.pointcloud_utils import expand_pcd

def PointCloudDiT2_xS(**kwargs):
    return PointCloudDiT2(depth=5, hidden_size=128, num_heads=4, **kwargs)

class TAX3Dv2Network(nn.Module):
    """
    Network containing the TAX3Dv2 architecture.
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.dit = PointCloudDiT2_xS(
            in_channels=model_cfg.in_channels,
            learn_sigma=model_cfg.learn_sigma,
            model_cfg=model_cfg,
        )

    def forward(self, x, t, **kwargs):
        return self.dit(x, t, **kwargs)

class TAX3Dv2Module(L.LightningModule):
    """
    Lightning module that handles training, inference, evaluation, and visualization for TAX3Dv2.
    """
    def __init__(self, network, cfg):
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
        self.diff_steps = self.model_cfg.diff_train_steps # TODO: rename to diff_steps?
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

    def get_model_kwargs(self, batch, num_samples=None):
        """
        Extract model-specific kwargs from the given batch.

        Args:
            batch: the input batch
            num_samples: the number of samples to generate per batch element
        """
        pre_kwargs = {
            "query": batch["pc_action"].to(self.device),
            "context": batch["pc_anchor"].to(self.device),
        }
        bs = pre_kwargs["query"].shape[0]

        # expanding point clouds, if necessary
        if num_samples is not None:
            pre_kwargs = {k: expand_pcd(v, num_samples) for k, v in pre_kwargs.items()}
        
        # permuting dimensions - channel first, (B, 3, N)
        pre_kwargs = {k: v.permute(0, 2, 1) for k, v in pre_kwargs.items()}

        # populating model kwargs
        model_kwargs = dict(
            y=torch.cat([pre_kwargs["query"], pre_kwargs["context"]], dim=-1),
            q=pre_kwargs["query"].shape[-1],
        )
        
        # handling reference frame inpainting
        if "ref_frame" in batch:
            ref_frame = batch["ref_frame"].to(self.device)
            if num_samples is not None:
                # either frame is already in correct shape, or it needs to be expanded
                if ref_frame.shape[0] == bs:
                    ref_frame = expand_pcd(ref_frame, num_samples)
                elif ref_frame.shape[0] != bs * num_samples:
                    raise ValueError("Invalid reference frame shape.")
            model_kwargs["ref_frame"] = ref_frame.permute(0, 2, 1)

        return model_kwargs

    def get_world_preds(self, batch, num_samples, pc_query, pred_dict):
        """
        Get world-frame predictions from the given batch and predictions.
        """
        T_query2world = Transform3d(
            matrix=expand_pcd(batch["T_action2world"].to(self.device), num_samples)
        )
        T_context2world = Transform3d(
            matrix=expand_pcd(batch["T_goal2world"].to(self.device), num_samples)
        )

        pred_point = pred_dict["point"]["pred"]
        results = pred_dict["results"]
        # updating prediction reference frames
        pred_point = pred_point + pred_dict["ref_frame"]
        results = [res + pred_dict["ref_frame_results"][i] for i, res in enumerate(results)]

        pred_point_world = T_context2world.transform_points(pred_point)
        pc_query_world = T_query2world.transform_points(pc_query)
        pred_flow_world = pred_point_world - pc_query_world
        results_world = [
            T_context2world.transform_points(res) for res in results
        ]
        return pred_flow_world, pred_point_world, results_world

    def get_viz_args(self, batch, viz_idx):
        """
        Get visualization arguments for wandb logging.
        """
        pc_pos_viz = batch["pc"][viz_idx, :, :3] + batch["goal_origin"][viz_idx, :3].unsqueeze(0)
        pc_query_viz = batch["pc_action"][viz_idx, :, :3]
        pc_context_viz = batch["pc_anchor"][viz_idx, :, :3]
        viz_args = {
            "pc_pos_viz": pc_pos_viz,
            "pc_action_viz": pc_query_viz,
            "pc_anchor_viz": pc_context_viz,
        }
        return viz_args

    def forward(self, batch, t):
        """
        Forward pass to compute diffusion training loss.
        """
        ground_truth = batch[self.label_key].permute(0, 2, 1) # channel first
        # update ground truth with goal origin
        ground_truth = torch.cat([ground_truth, batch['goal_origin'].unsqueeze(-1)], dim=-1)
        model_kwargs = self.get_model_kwargs(batch)

        # run diffusion
        loss_dict = self.diffusion.training_losses(
            model=self.network,
            x_start=ground_truth,
            t=t,
            model_kwargs=model_kwargs,
        )
        loss = loss_dict["loss"].mean()
        return None, loss
    
    @torch.no_grad()
    def predict(self, batch, num_samples, unflatten=False, progress=True, full_prediction=True):
        """
        Sample prediction for a given batch.

        Args:
            batch: the input batch
            num_samples: the number of samples to generate per batch element
            progress: whether to show progress bar
            full_prediction: whether to return full prediction (flow and point, goal and world frame)
        """
        bs, sample_size = batch["pc_action"].shape[:2]
        sample_size += 1 # adding 1 for the goal origin prediction
        inpaint_ref_frame = "ref_frame" in batch
        model_kwargs = self.get_model_kwargs(batch, num_samples)

        # generating latents and running diffusion
        z = torch.randn(bs * num_samples, 3, sample_size, device=self.device)
        pred, results, extras = self.diffusion.p_sample_loop(
            self.network,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress,
            device=self.device,
        )
        pred = pred.permute(0, 2, 1)

        # splitting prediction into shape and reference frame diffusion
        pred, ref_frame = pred[:, :-1, :], pred[:, -1:, :]
        # extracting inpainted reference frame, if necessary
        if inpaint_ref_frame:
            ref_frame = model_kwargs["ref_frame"].permute(0, 2, 1)
        
        # handling prediction output
        if full_prediction:
            # return full prediction (flow and point, goal and world frame)
            q = model_kwargs["q"]
            pc_query = model_kwargs["y"][:, :, :q].permute(0, 2, 1)
            results = [res.permute(0, 2, 1) for res in results]

            # splitting results into shape and reference frame diffusion
            ref_frame_results = [res[:, -1:, :] for res in results]
            results = [res[:, :-1, :] for res in results]
            # extracting inpainted reference frame, if necessary
            if inpaint_ref_frame:
                ref_frame_results = [ref_frame] * len(results)

            # combining flow and point predictions
            if self.prediction_type == "flow":
                pred_flow = pred
                pred_point = pc_query + pred_flow
                # for flow predictions, convert results to point predictions
                results = [pc_query + res for res in results]
            elif self.prediction_type == "point":
                pred_point = pred
                pred_flow = pred_point - pc_query

            pred_dict = {
                "flow": {"pred": pred_flow},
                "point": {"pred": pred_point},
                "ref_frame": ref_frame,
                "results": results,
                "ref_frame_results": ref_frame_results,
                "extras": extras,
            }

            # computing world-frame predictions
            pred_flow_world, pred_point_world, results_world = self.get_world_preds(
                batch, num_samples, pc_query, pred_dict
            )
            pred_dict["flow"]["pred_world"] = pred_flow_world
            pred_dict["point"]["pred_world"] = pred_point_world
            pred_dict["results_world"] = results_world
        else:
            # only return the prediction type in the goal frame
            pred_dict = {
                self.prediction_type: {"pred": pred},
                "ref_frame": ref_frame,
            }
        return pred_dict

    def predict_wta(self, batch, num_samples):
        """
        Predict WTA (winner-take-all) samples, and compute WTA metrics. Unlike predict, this function 
        assumes the ground truth is available.

        Args:
            batch: the input batch
            num_samples: the number of samples to generate per batch element
        """
        # TODO: is there a way for this to incorporate some precision metric?
        ground_truth = batch[self.label_key].to(self.device)
        ground_truth_ref_frame = batch["goal_origin"].to(self.device).unsqueeze(-2)
        seg = batch["seg"].to(self.device)

        # re-shaping and expanding for winner-take-all
        bs = ground_truth.shape[0]
        ground_truth = expand_pcd(ground_truth, num_samples)
        ground_truth_ref_frame = expand_pcd(ground_truth_ref_frame, num_samples)
        seg = expand_pcd(seg, num_samples)

        # generating diffusion predictions
        pred_dict = self.predict(
            batch, num_samples, unflatten=False, progress=True, full_prediction=False
        )
        pred = pred_dict[self.prediction_type]["pred"]
        pred_ref_frame = pred_dict["ref_frame"]
        
        # updating prediction and ground truth with corresponding reference frames
        pred = pred + pred_ref_frame
        ground_truth = ground_truth + ground_truth_ref_frame

        # computing error metrics
        seg = seg == 0
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
            self.eval()
            with torch.no_grad():
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