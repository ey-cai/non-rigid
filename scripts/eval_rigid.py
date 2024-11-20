import hydra
import lightning as L
import json
import omegaconf
import torch
import torch.utils._pytree as pytree
import wandb

from functools import partial
from pathlib import Path
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from non_rigid.utils.logging_utils import viz_predicted_vs_gt
from non_rigid.datasets.rigid import NDFPointDataset, RigidDataModule
from non_rigid.models.df_base import (
    DiffusionFlowBase, 
    FlowPredictionInferenceModule, 
    PointPredictionInferenceModule
)
from non_rigid.models.regression import (
    LinearRegression,
    LinearRegressionInferenceModule
)
from non_rigid.utils.vis_utils import FlowNetAnimation
from non_rigid.utils.script_utils import (
    PROJECT_ROOT,
    LogPredictionSamplesCallback,
    create_model,
    create_datamodule,
    match_fn,
    flatten_outputs
)

from non_rigid.metrics.flow_metrics import flow_cos_sim, flow_rmse, pcd_rmse
from non_rigid.metrics.rigid_metrics import get_pred_pcd_rigid_errors
from non_rigid.models.dit.diffusion import create_diffusion
from non_rigid.utils.pointcloud_utils import expand_pcd
from tqdm import tqdm
import numpy as np

from pytorch3d.transforms import Transform3d, Rotate, Translate, euler_angles_to_matrix
import rpad.visualize_3d.plots as vpl



def visualize_batched_point_clouds(point_clouds):
    """
    Helper function to visualize a list of batched point clouds. This is meant to be used 
    when visualizing action/anchor/prediction point clouds, without having to add 

    point_clouds: list of point clouds, each of shape (B, N, 3)
    """
    pcs = [pc.cpu().flatten(0, 1) for pc in point_clouds]
    segs = []
    for i, pc in enumerate(pcs):
        segs.append(torch.ones(pc.shape[0]).int() * i)

    return vpl.segmentation_fig(
        torch.cat(pcs),
        torch.cat(segs),
    )


def eval_precision(stage, dataloader, model, device, num_trials, cfg):
    metrics_list = []
    # Function to accumulate errors in a list
    def accumulate_metrics(metrics_list, transformation_errors, rmse_errors, t_err_centroid):
        metrics_list.append({
            "t_err": transformation_errors["error_t_mean"],
            "r_err": transformation_errors["error_R_mean"],
            "rmse": rmse_errors,
            "t_err_centroid": t_err_centroid,
        })

    # Evaluating Training Precision
    for batch in tqdm(dataloader, desc="Evaluating {} precision".format(stage)):
        batch = {key: value.to(device) for key, value in batch.items()}
        pred_dict = model.predict(batch, num_trials, progress=False)
        pred = pred_dict[cfg.model.type]["pred"]

        # getting predicted action point cloud
        if cfg.model.type == "flow":
            pred_world = batch["pc_action"][:, :, :3] + pred
        elif cfg.model.type == "point":
            pred_world = pred
        goal_world = batch["pc"][:, :, :3]
        
        # fix scaling for diffusion
        pred_world_scaled = pred_world / cfg.dataset.pcd_scale_factor
        goal_world_scaled = goal_world / cfg.dataset.pcd_scale_factor
        #pred_world_scaled = pred_world
        #goal_world_scaled = goal_world

        mean_rmse = pcd_rmse(pred_world_scaled, goal_world_scaled).mean().cpu().item()

        gt_action_centroid = goal_world_scaled.mean(dim=1)
        pred_action_centroid = pred_world_scaled.mean(dim=1)

        t_err_centroid = torch.norm(gt_action_centroid - pred_action_centroid, dim=1).mean().cpu().item()
 
        transformation_errors = get_pred_pcd_rigid_errors(batch=batch, pred_xyz=pred_world, error_type= "demo", scale_factor=cfg.dataset.pcd_scale_factor)
        accumulate_metrics(metrics_list, transformation_errors, mean_rmse, t_err_centroid)
        
        if cfg.wandb.online:
            # pick a random sample in the batch to visualize
            viz_idx = np.random.randint(0, batch["pc"].shape[0])
            pred_viz = pred_dict[cfg.model.type]["pred"][viz_idx]
            viz_args = model.get_viz_args(batch, viz_idx)

            # getting predicted action point cloud
            if cfg.model.type == "flow":
                pred_action_viz = viz_args["pc_action_viz"] + pred_viz
            elif cfg.model.type == "point":
                pred_action_viz = pred_viz

            # logging predicted vs ground truth point cloud
            viz_args["pred_action_viz"] = pred_action_viz
            predicted_vs_gt = viz_predicted_vs_gt(**viz_args)
            wandb.log({f"{stage}/predicted_vs_gt": predicted_vs_gt})

    return metrics_list

@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
def main(cfg):
    print(
        json.dumps(
            omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            sort_keys=True,
            indent=4,
        )
    )

    run = None
    if cfg.wandb.online:
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            job_type=cfg.job_type,
            save_code=True,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
        )

        if cfg.wandb.name is not None: 
            wandb.run.name = cfg.wandb.name
            wandb.run.save() 

    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")

    # Global seed for reproducibility.
    L.seed_everything(42)

    device = f"cuda:{cfg.resources.gpus[0]}"

    ######################################################################
    # Create the datamodule.
    # Should be the same one as in training, but we're gonna use val+test
    # dataloaders.
    ######################################################################
    cfg, datamodule = create_datamodule(cfg)

    ######################################################################
    # Create the network(s) which will be evaluated (same as training).
    # You might want to put this into a "create_network" function
    # somewhere so train and eval can be the same.
    #
    # We'll also load the weights.
    ######################################################################

    # Model architecture is dataset-dependent, so we have a helper
    # function to create the model (while separating out relevant vals).
    network, model = create_model(cfg)


    # get checkpoint file (for now, this does not log a run)
    checkpoint_reference = cfg.checkpoint.reference
    if checkpoint_reference.startswith(cfg.wandb.entity):
        api = wandb.Api()
        artifact_dir = cfg.wandb.artifact_dir
        artifact = api.artifact(checkpoint_reference, type="model")
        ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    else:
        ckpt_file = checkpoint_reference
    # Load the network weights.
    ckpt = torch.load(ckpt_file, map_location=device)
    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    )
    # set model to eval mode
    network.eval()
    model.eval()


    ######################################################################
    # Create the trainer.
    # Bit of a misnomer here, we're not doing training. But we are gonna
    # use it to set up the model appropriately and do all the batching
    # etc.
    #
    # If this is a different kind of downstream eval, chuck this block.
    ######################################################################

    trainer = L.Trainer(
        accelerator="gpu",
        devices=cfg.resources.gpus,
        # precision="16-mixed",
        precision="32-true",
        logger=False,
    )


    ######################################################################
    # Run the model on the train/val/test sets.
    # This outputs a list of dictionaries, one for each batch. This
    # is annoying to work with, so later we'll flatten.
    #
    # If a downstream eval, you can swap it out with whatever the eval
    # function is.
    ######################################################################
    
    ### Running Metrics Eval ###
    model.to(device)
    # precision_dm = 2

    # creating precision-specific datamodule - needs specific batch size
    # if "multi_cloth" in cfg.dataset:
    #     bs = int(400 / cfg.dataset.multi_cloth.size)
    # else:
    #     raise NotImplementedError("Precision metrics only supported for multi-cloth datasets.")
    
    train_bs = 16
    val_bs = 16
    
    # We should not be using WTAs for performance evaluation, since ground truth should be assumed unavailable
    '''
    if cfg.wta:
        num_trials = cfg.inference.num_wta_trials   # if use winner-takes-all, we evaluate the best performance over multiple trials
    else:
        num_trials = 1
    '''
    num_trials = 1

    # cfg.dataset.sample_size_action = -1
    datamodule = RigidDataModule(
        # root=data_root,
        batch_size=train_bs,
        val_batch_size=val_bs,
        num_workers=cfg.resources.num_workers,
        dataset_cfg=cfg.dataset,
    )
    # just need the dataloaders
    datamodule.setup(stage="predict")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    train_metrics_list = eval_precision('train', train_loader, model, device, num_trials, cfg)
    val_metrics_list = eval_precision('val', val_loader, model, device, num_trials, cfg)


    train_metrics = {k: np.array([m[k] for m in train_metrics_list]) for k in train_metrics_list[0].keys()}
    train_mean_t_err = train_metrics["t_err"].mean()
    train_mean_t_err_centroid = train_metrics["t_err_centroid"].mean()
    train_mean_r_err = train_metrics["r_err"].mean()
    train_mean_rmse_err = train_metrics["rmse"].mean()
    train_std_t_err = train_metrics["t_err"].std()
    train_std_t_err_centroid = train_metrics["t_err_centroid"].std()
    train_std_r_err = train_metrics["r_err"].std()
    train_std_rmse_err = train_metrics["rmse"].std()


    val_metrics = {k: np.array([m[k] for m in val_metrics_list]) for k in val_metrics_list[0].keys()}
    val_mean_t_err = val_metrics["t_err"].mean()
    val_mean_t_err_centroid = val_metrics["t_err_centroid"].mean()
    val_mean_r_err = val_metrics["r_err"].mean()
    val_mean_rmse_err = val_metrics["rmse"].mean()
    val_std_t_err = val_metrics["t_err"].std()
    val_std_t_err_centroid = val_metrics["t_err_centroid"].std()
    val_std_r_err = val_metrics["r_err"].std()
    val_std_rmse_err = val_metrics["rmse"].std()

    print(f"Training Mean Translation Error: {train_mean_t_err}")
    print(f"Training Std Translation Error: {train_std_t_err}")
    print(f"Training Mean Translation Error Centroid: {train_mean_t_err_centroid}")
    print(f"Training Std Translation Error Centroid: {train_std_t_err_centroid}")
    print(f"Training Mean Rotation Error: {train_mean_r_err}")
    print(f"Training Std Rotation Error: {train_std_r_err}")
    print(f"Training Mean RMSE Error: {train_mean_rmse_err}")
    print(f"Training Std RMSE Error: {train_std_rmse_err}")

    print(f"Validation Mean Translation Error: {val_mean_t_err}")
    print(f"Validation Std Translation Error: {val_std_t_err}")
    print(f"Validation Mean Translation Error Centroid: {val_mean_t_err_centroid}")
    print(f"Validation Std Translation Error Centroid: {val_std_t_err_centroid}")
    print(f"Validation Mean Rotation Error: {val_mean_r_err}")
    print(f"Validation Std Rotation Error: {val_std_r_err}")
    print(f"Validation Mean RMSE Error: {val_mean_rmse_err}")
    print(f"Validation Std RMSE Error: {val_std_rmse_err}")


    if cfg.wandb.online:

        # Create a WandB table for metrics
        metrics_table = wandb.Table(
            columns=[
                "Dataset", "Mean Translation Error", "Std Translation Error", "Mean Translation Centroid Error", "Std Translation Centroid Error",
                "Mean Rotation Error", "Std Rotation Error", "Mean RMSE Error", "Std RMSE Error"
            ],
            data=[
                ["Training", train_mean_t_err, train_std_t_err, train_mean_t_err_centroid, train_std_t_err_centroid, train_mean_r_err, train_std_r_err, train_mean_rmse_err, train_std_rmse_err],
                ["Validation", val_mean_t_err, val_std_t_err, val_mean_t_err_centroid, val_std_t_err_centroid, val_mean_r_err, val_std_r_err, val_mean_rmse_err, val_std_rmse_err],
            ]
        )
        wandb.log({"Evaluation Metrics": metrics_table})




    '''
    if cfg.coverage:
            train_outputs, val_outputs = trainer.predict(
                model,
                dataloaders=[
                    datamodule.train_dataloader(),
                    datamodule.val_dataloader(),
                ]
                )
        

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
            fig_wta = make_subplots(rows=2, cols=1, shared_xaxes=True)
            color_dict = {
                "train": "blue",
                "val": "red",
            }

            for outputs_list, name in [
                (train_outputs, "train"),
                (val_outputs, "val"),
            ]:
                # Put everything on CPU, and flatten a list of dicts into one dict.
                out_cpu = [pytree.tree_map(lambda x: x.cpu(), o) for o in outputs_list]
                outputs = flatten_outputs(out_cpu)

                # Plot histogram
                fig.add_trace(go.Histogram(
                    x=outputs["rmse"].flatten(), 
                    nbinsx=100, 
                    name=f"{name} RMSE",
                    legendgroup=f"{name} RMSE",
                    marker=dict(
                        color=color_dict[name],
                    ),
                ), row=1, col=1)

                fig.add_trace(go.Box(
                    x=outputs["rmse"].flatten(),
                    marker_symbol='line-ns-open',
                    marker=dict(
                        color=color_dict[name],
                    ),
                    boxpoints='all',
                    pointpos=0,
                    hoveron='points',
                    name=f"{name} RMSE",
                    showlegend=False,
                    legendgroup=f"{name} RMSE",
                ), row=2, col=1)

                # Plot WTA histogram
                fig_wta.add_trace(go.Histogram(
                    x=outputs["rmse_wta"].flatten(), 
                    nbinsx=100, 
                    name=f"{name} RMSE WTA",
                    legendgroup=f"{name} RMSE WTA",
                    marker=dict(
                        color=color_dict[name],
                    ),
                ), row=1, col=1)

                fig_wta.add_trace(go.Box(
                    x=outputs["rmse_wta"].flatten(),
                    marker_symbol='line-ns-open',
                    marker=dict(
                        color=color_dict[name],
                    ),
                    boxpoints='all',
                    pointpos=0,
                    hoveron='points',
                    name=f"{name} RMSE WTA",
                    showlegend=False,
                    legendgroup=f"{name} RMSE WTA",
                ), row=2, col=1)

                # Compute the metrics
                rmse = torch.mean(outputs["rmse"])
                rmse_wta = torch.mean(outputs["rmse_wta"])
                print(f"{name} rmse: {rmse}, rmse wta: {rmse_wta}")

            fig.show()
            fig_wta.show()
    '''

if __name__ == "__main__":
    main()