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

from non_rigid.datasets.rigid import NDFDataset, RigidDataModule
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

from non_rigid.metrics.flow_metrics import flow_cos_sim, flow_rmse
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



    VISUALIZE_DEMOS = False
    VISUALIZE_PREDS = True
    VISUALIZE_SINGLE = False
    VISUALIZE_PULL = False



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

    if cfg.precision:
        model.to(device)
        # precision_dm = 2

        # creating precision-specific datamodule - needs specific batch size
        # if "multi_cloth" in cfg.dataset:
        #     bs = int(400 / cfg.dataset.multi_cloth.size)
        # else:
        #     raise NotImplementedError("Precision metrics only supported for multi-cloth datasets.")
        
        train_bs = 16
        val_bs = 16
        num_wta_trials = 1 # assuming we don't need to do multiple trials for precision

        train_metrics_list = []
        val_metrics_list = []

        # Function to accumulate errors in a list
        def accumulate_metrics(metrics_list, errors):
            metrics_list.append({
                "t_err": errors["error_t_mean"],
                "r_err": errors["error_R_mean"]
            })
        
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

        for batch in tqdm(train_loader, desc="Evaluating Training Precision"):
            batch = {key: value.to(device) for key, value in batch.items()}
            pred_dict = model.predict(batch, num_wta_trials, progress=False)
            pred = pred_dict["point"]["pred_world"]
            errors = get_pred_pcd_rigid_errors(batch=batch, pred_xyz=pred, error_type= "demo")
            accumulate_metrics(train_metrics_list, errors)

        for batch in tqdm(val_loader, desc="Evaluating Training Precision"):
            batch = {key: value.to(device) for key, value in batch.items()}
            pred_dict = model.predict(batch, num_wta_trials, progress=False)
            pred = pred_dict["point"]["pred_world"]
            errors = get_pred_pcd_rigid_errors(batch=batch, pred_xyz=pred, error_type= "demo")
            accumulate_metrics(val_metrics_list, errors)

        train_metrics = {k: np.array([m[k] for m in train_metrics_list]) for k in train_metrics_list[0].keys()}
        train_mean_t_err = train_metrics["t_err"].mean()
        train_mean_r_err = train_metrics["r_err"].mean()

        val_metrics = {k: np.array([m[k] for m in val_metrics_list]) for k in val_metrics_list[0].keys()}
        val_mean_t_err = val_metrics["t_err"].mean()
        val_mean_r_err = val_metrics["r_err"].mean()

        print(f"Training Mean Translation Error: {train_mean_t_err}")
        print(f"Training Mean Rotation Error: {train_mean_r_err}")
        print(f"Validation Mean Translation Error: {val_mean_t_err}")
        print(f"Validation Mean Rotation Error: {val_mean_r_err}")

    if cfg.viz:
        if VISUALIZE_DEMOS:
            model.to(device)
            bs = 12
            train_dataloader = torch.utils.data.DataLoader(
                datamodule.train_dataset, batch_size=400, shuffle=True
            )
            val_dataloader = torch.utils.data.DataLoader(
                datamodule.val_dataset, batch_size=40, shuffle=True
            )
            val_ood_loader = torch.utils.data.DataLoader(
                datamodule.val_ood_dataset, batch_size=40, shuffle=True
            )

            train_batch = next(iter(train_dataloader))
            val_batch = next(iter(val_dataloader))
            val_ood_batch = next(iter(val_ood_loader))


            # train_dict = model.predict_wta(train_batch, 'train')
            # val_dict = model.predict_wta(val_batch, 'val')
            # val_ood_dict = model.predict_wta(val_ood_batch, 'val_ood')

            # val_errors = val_dict['rmse']
            # val_ood_errors = val_ood_dict['rmse']


            cdw_errs = np.load('/home/eycai/datasets/nrp/cd-w.npz')
            cd_errs = np.load('/home/eycai/datasets/nrp/tax3dcd.npz')

            vem_cdw = cdw_errs['vem']
            voem_cdw = cdw_errs['voem']
            vem_cd = cd_errs['vem']
            voem_cd = cd_errs['voem']



            val_errors = np.random.rand(40)
            val_ood_errors = np.random.rand(40) * 4

            train_pc = train_batch["pc_anchor"]
            val_pc = val_batch["pc_anchor"]
            val_ood_pc = val_ood_batch["pc_anchor"]


            train_locs = torch.mean(train_pc, dim=1)
            val_locs = torch.mean(val_pc, dim=1)
            val_ood_locs = torch.mean(val_ood_pc, dim=1)
            # plotly go to scatter plot locs
            fig = go.Figure()
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                row_heights=[0.5, 0.5],
                                subplot_titles=("CD-W", "TAX3D-CP (Ours)"))

            # ----------- PLOTTING X VS Y ----------------
            fig.add_trace(go.Scatter(
                x=train_locs[:, 0].cpu(),
                y=train_locs[:, 1].cpu(),
                mode='markers',
                marker_symbol='x-thin',
                marker=dict(
                    size=20,
                    color='rgb(38,133,249)',
                    line=dict(
                        width=4,
                        color='rgb(38,133,249)'
                    ),
                ),
                name='Train',
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=val_locs[:, 0].cpu(),
                y=val_locs[:, 1].cpu(),
                mode='markers',
                marker_symbol='square',
                marker=dict(
                    size=20,
                    color=vem_cdw,
                    coloraxis='coloraxis',
                    line=dict(
                        width=2,
                        color='Black'
                    ),
                ),
                name='Unseen',
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=val_ood_locs[:, 0].cpu(),
                y=val_ood_locs[:, 1].cpu(),
                mode='markers',
                marker_symbol='diamond',
                marker=dict(
                    size=20,
                    color=voem_cdw,
                    coloraxis='coloraxis',
                    line=dict(
                        width=2,
                        color='Black'
                    ),
                ),
                name='Unseen (OOD)',
            ), row=1, col=1)


            # ----------- PLOTTING X VS Z ----------------
            fig.add_trace(go.Scatter(
                x=train_locs[:, 0].cpu(),
                y=train_locs[:, 1].cpu(),
                mode='markers',
                marker_symbol='x-thin',
                marker=dict(
                    size=20,
                    color='rgb(38,133,249)',
                    line=dict(
                        width=4,
                        color='rgb(38,133,249)'
                    ),
                ),
                name='Train',
                showlegend=False,
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=val_locs[:, 0].cpu(),
                y=val_locs[:, 1].cpu(),
                mode='markers',
                marker_symbol='square',
                marker=dict(
                    size=20,
                    color=vem_cd,
                    coloraxis='coloraxis',
                    line=dict(
                        width=2,
                        color='Black'
                    ),
                ),
                name='Unseen',
                showlegend=False,
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=val_ood_locs[:, 0].cpu(),
                y=val_ood_locs[:, 1].cpu(),
                mode='markers',
                marker_symbol='diamond',
                marker=dict(
                    size=20,
                    color=voem_cd,
                    coloraxis='coloraxis',
                    line=dict(
                        width=2,
                        color='Black'
                    ),
                ),
                name='Unseen (OOD)',
                showlegend=False,
            ), row=2, col=1)


            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                'font': dict(
                    family="Arial",
                    size=52,
                    color="Black"
                ),
                })
            fig.update_annotations(font_size=72)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
            fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
            fig.update_yaxes(title_text="Y", row=1, col=1)
            fig.update_yaxes(title_text="Y", row=2, col=1)
            fig.update_xaxes(title_text="X", row=2, col=1)

            fig.update_layout(legend=dict(
                # yanchor="top",
                # y=0.65,
                xanchor="left",
                x=0.68,
                orientation="h",
            ))
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title="RMSE",
                ),
                coloraxis=dict(
                    colorscale=['green', 'red'],
                    #cmin=0,
                    #cmax=4,
                )
            )

            fig.show()
            quit()
            visualize_batched_point_clouds([train_pc, val_pc, val_ood_pc]).show()

            pass

        if VISUALIZE_PREDS:
            model.to(device)
            dataloader = torch.utils.data.DataLoader(
                datamodule.val_dataset, batch_size=1, shuffle=False
            )
            iterator = iter(dataloader)
            for _ in range(32):
                batch = next(iterator)
            pred_dict = model.predict(batch, 50)
            # extracting anchor point cloud depending on model type
            if cfg.model.type == "flow":
                scene_pc = batch["pc"].flatten(0, 1).cpu().numpy()
                seg = batch["seg"].flatten(0, 1).cpu().numpy()
                anchor_pc = scene_pc[~seg.astype(bool)]
            else:
                anchor_pc = batch["pc_anchor"].flatten(0, 1).cpu().numpy()

            # pred_action = pred_dict["pred_action"][[8]] # 0,8
            pred_action = pred_dict["pred_action"]
            pred_action_size = pred_action.shape[1]
            pred_action = pred_action.flatten(0, 1).cpu().numpy()
            # color-coded segmentations
            anchor_seg = np.zeros(anchor_pc.shape[0], dtype=np.int64)
            # if cfg.model.type == "flow":
            #     pred_action_size = cfg.dataset.sample_size_action + cfg.dataset.sample_size_anchor
            # else:
            #     pred_action_size = cfg.dataset.sample_size_action
            pred_action_seg = np.array([np.arange(1, 11)] * pred_action_size).T.flatten()
            # visualize
            fig = vpl.segmentation_fig(
                np.concatenate((anchor_pc, pred_action)),
                np.concatenate((anchor_seg, pred_action_seg)),
            )
            fig.show()

        if VISUALIZE_PULL:
            model.to(device)
            dataloader = torch.utils.data.DataLoader(
                datamodule.val_dataset, batch_size=1, shuffle=False
            )
            iterator = iter(dataloader)
            for _ in range(11):
                batch = next(iterator)
            pred_dict = model.predict(batch, 1)
            results = pred_dict["results"]
            action_pc = batch["pc_action"].flatten(0, 1).cpu()
            # pred_action = .cpu()
            if cfg.model.type == "flow":
                # pcd = batch["pc_action"].flatten(0, 1).cpu()
                pcd = torch.cat([
                    batch["pc_action"].flatten(0, 1),
                    pred_dict["pred_action"].flatten(0, 1).cpu(),
                ]).cpu()
            elif cfg.model.type == "flow_cross":
                pcd = torch.cat([
                    batch["pc_anchor"].flatten(0, 1),
                    batch["pc_action"].flatten(0, 1),
                    # pred_dict['pred_action'].flatten(0, 1).cpu(),
                ], dim=0).cpu()
            elif cfg.model.type == "point_cross":
                pcd = torch.cat([
                    batch["pc_anchor"].flatten(0, 1),
                    pred_dict["pred_action"].flatten(0, 1).cpu()
                ], dim=0).cpu()    
            
            # visualize
            animation = FlowNetAnimation()
            for noise_step in tqdm(results):
                pred_step = noise_step[0].permute(1, 0).cpu()
                if cfg.model.type == "point_cross":
                    flows = torch.zeros_like(pred_step)
                    animation.add_trace(
                        pcd,
                        [flows],
                        [pred_step],
                        "red",
                    )
                else:
                    animation.add_trace(
                        pcd,
                        [action_pc],# if cfg.model.type == "flow_cross" else pcd],
                        [pred_step],
                        "red",
                    )
            fig = animation.animate()
            fig.show()


    if VISUALIZE_SINGLE:
        model.to(device)
        dataloader = torch.utils.data.DataLoader(
            datamodule.val_dataset, batch_size=1, shuffle=False
        )
        batch = next(iter(dataloader))
        pred_dict = model.predict(batch, 1)

        results = pred_dict["results"]
        action_pc = batch["pc_action"].flatten(0, 1).cpu()
        # pred_action = .cpu()
        if cfg.model.type == "flow":
            # pcd = batch["pc_action"].flatten(0, 1).cpu()
            pcd = torch.cat([
                batch["pc_action"].flatten(0, 1),
                pred_dict["pred_action"].flatten(0, 1).cpu(),
            ]).cpu()
        elif cfg.model.type == "flow_cross":
            pcd = torch.cat([
                batch["pc_anchor"].flatten(0, 1),
                batch["pc_action"].flatten(0, 1),
                # pred_dict['pred_action'].flatten(0, 1).cpu(),
            ], dim=0).cpu()
        elif cfg.model.type == "point_cross":
            pcd = torch.cat([
                batch["pc_anchor"].flatten(0, 1),
                pred_dict["pred_action"].flatten(0, 1).cpu()
            ], dim=0).cpu()    
        
        # visualize
        animation = FlowNetAnimation()
        for noise_step in tqdm(results):
            pred_step = noise_step[0].permute(1, 0).cpu()
            if cfg.model.type == "point_cross":
                flows = torch.zeros_like(pred_step)
                animation.add_trace(
                    pcd,
                    [flows],
                    [pred_step],
                    "red",
                )
            else:
                animation.add_trace(
                    pcd,
                    [action_pc],# if cfg.model.type == "flow_cross" else pcd],
                    [pred_step],
                    "red",
                )
        fig = animation.animate()
        fig.show()

if __name__ == "__main__":
    main()