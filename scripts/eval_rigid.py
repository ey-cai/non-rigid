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

from rpad.visualize_3d.plots import flow_fig, _flow_traces, pointcloud, _3d_scene
from non_rigid.datasets.proc_cloth_flow import ProcClothFlowDataset, ProcClothFlowDataModule
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
        train_outputs, val_outputs, val_ood_outputs = trainer.predict(
            model,
            dataloaders=[
                datamodule.train_dataloader(),
                *datamodule.val_dataloader(),
            ]
            )
    

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig_wta = make_subplots(rows=2, cols=1, shared_xaxes=True)
        color_dict = {
            "train": "blue",
            "val": "red",
            "val_ood": "green",
        }
        for outputs_list, name in [
            (train_outputs, "train"),
            (val_outputs, "val"),
            (val_ood_outputs, "val_ood")
        ]:
            # Put everything on CPU, and flatten a list of dicts into one dict.
            out_cpu = [pytree.tree_map(lambda x: x.cpu(), o) for o in outputs_list]
            outputs = flatten_outputs(out_cpu)
            # plot histogram
            fig.add_trace(go.Histogram(
                x=outputs["rmse"].flatten(), 
                nbinsx=100, 
                name=f"{name} RMSE",
                legendgroup=f"{name} RMSE",
                marker=dict(
                    color=color_dict[name],
                ),
                # color=name,
                ), row=1, col=1,
            )
            fig.add_trace(go.Box(
                x=outputs["rmse"].flatten(),
                marker_symbol='line-ns-open',
                marker=dict(
                    color=color_dict[name],
                ),
                boxpoints='all',
                #fillcolor='rgba(0,0,0,0)',
                #line_color='rgba(0,0,0,0)',
                pointpos=0,
                hoveron='points',
                name=f"{name} RMSE",
                showlegend=False,
                legendgroup=f"{name} RMSE",           
                ), row=2, col=1
            )
            # plot wta histogram
            fig_wta.add_trace(go.Histogram(
                x=outputs["rmse_wta"].flatten(), 
                nbinsx=100, 
                name=f"{name} RMSE WTA",
                legendgroup=f"{name} RMSE WTA",
                marker=dict(
                    color=color_dict[name],
                ),
                # color=name,
                ), row=1, col=1,
            )
            fig_wta.add_trace(go.Box(
                x=outputs["rmse_wta"].flatten(),
                marker_symbol='line-ns-open',
                marker=dict(
                    color=color_dict[name],
                ),
                boxpoints='all',
                #fillcolor='rgba(0,0,0,0)',
                #line_color='rgba(0,0,0,0)',
                pointpos=0,
                hoveron='points',
                name=f"{name} RMSE WTA",
                showlegend=False,
                legendgroup=f"{name} RMSE WTA",           
                ), row=2, col=1
            )

            # Compute the metrics.
            # cos_sim = torch.mean(outputs["cos_sim"])
            rmse = torch.mean(outputs["rmse"])
            # cos_sim_wta = torch.mean(outputs["cos_sim_wta"])
            rmse_wta = torch.mean(outputs["rmse_wta"])
            # print(f"{name} cos sim: {cos_sim}, rmse: {rmse}")
            # print(f"{name} cos sim wta: {cos_sim_wta}, rmse wta: {rmse_wta}")
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
        
        if cfg.dataset.cloth_geometry == "single":
            num_samples = 1
            train_bs = 400
            val_bs = 40
        else:
            num_samples = 20
            train_bs = 4
            val_bs = 4


        # cfg.dataset.sample_size_action = -1
        datamodule = ProcClothFlowDataModule(
            # root=data_root,
            batch_size=train_bs,
            val_batch_size=val_bs,
            num_workers=cfg.resources.num_workers,
            dataset_cfg=cfg.dataset,
        )
        # just need the dataloaders
        datamodule.setup(stage="predict")
        train_loader = datamodule.train_dataloader()
        val_loader, val_ood_loader = datamodule.val_dataloader()

        # TODO: PRED FLOW IS NOT IN THE SAME FRAME AS GT FLOW...
        # ALSO WHY DO THEY GET SO MUCH WORSE FOR VAL AND VAL_OOD


        def precision_eval(dataloader, model, bs):
            precision_rmses = []
            for batch in tqdm(dataloader):
                # generate predictions
                seg = batch["seg"].to(device)
                pred_dict = model.predict(batch, num_samples, progress=False)
                pred_action = pred_dict["pred_action"]#.cpu() # in goal/anchor frame

                rot = batch['rot'].to(device)
                trans = batch['trans'].to(device)
                pc = batch["pc"].to(device)
                # reverting transforms
                T_world2origin = Translate(trans).inverse().compose(
                    Rotate(euler_angles_to_matrix(rot, 'XYZ'))
                )
                T_goal2world = Transform3d(
                    matrix=batch["T_goal2world"].to(device)
                )
                # goal to world, then world to origin
                pc = T_goal2world.transform_points(pc)
                pc = T_world2origin.transform_points(pc)

                # expanding transform to handle pred_action
                T_goal2world = Transform3d(
                    matrix=expand_pcd(T_goal2world.get_matrix(), num_samples)
                )
                T_world2origin = Transform3d(
                    matrix=expand_pcd(T_world2origin.get_matrix(), num_samples)
                )
                pred_action = T_goal2world.transform_points(pred_action)
                pred_action = T_world2origin.transform_points(pred_action)
            
                #fig = visualize_batched_point_clouds([pc_anchor, pc, pred_action])
                #fig.show()

                batch_rmses = []

                for p in pred_action:
                    p = expand_pcd(p.unsqueeze(0), bs)
                    rmse = flow_rmse(p, pc, mask=True, seg=seg)
                    rmse_min = torch.min(rmse)
                    batch_rmses.append(rmse_min)
                precision_rmses.append(torch.tensor(batch_rmses).mean())
            return torch.stack(precision_rmses).mean()

        train_precision_rmse = precision_eval(train_loader, model, train_bs)
        val_precision_rmse = precision_eval(val_loader, model, val_bs)
        val_ood_precision_rmse = precision_eval(val_ood_loader, model, val_bs)
        print("Train Precision RMSE: ", train_precision_rmse)
        print("Val Precision RMSE: ", val_precision_rmse)
        print("Val OOD Precision RMSE: ", val_ood_precision_rmse)


    MMD_METRICS = False
    PRECISION_METRICS = True
    VISUALIZE_ALL = False
    VISUALIZE_SINGLE = False
    VISUALIZE_EVAL_SINGLE = True
    VISUALIZE_SINGLE_IDX = 4
    
    SHOW_FIG = True
    SAVE_FIG = False


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

    
    if MMD_METRICS:
        print(f'Calculating MMD metrics')
        train_outputs, val_outputs = trainer.predict(
            model,
            dataloaders=[
                datamodule.train_dataloader(),
                datamodule.val_dataloader(),
            ]
        )

        for outputs_list, name in [
            (train_outputs, "train"),
            (val_outputs, "val"),
        ]:
            # Put everything on CPU, and flatten a list of dicts into one dict.
            out_cpu = [pytree.tree_map(lambda x: x.cpu(), o) for o in outputs_list]
            outputs = flatten_outputs(out_cpu)
            # plot histogram
            fig = px.histogram(outputs["rmse"], nbins=100, title=f"{name} MMD RMSE")
            if SHOW_FIG:
                fig.show()
            if SAVE_FIG:
                fig.write_html(f"{name}_mmd_histogram.html")
            # Compute the metrics.
            cos_sim = torch.mean(outputs["cos_sim"])
            rmse = torch.mean(outputs["rmse"])
            print(f"{name} cos sim: {cos_sim}, rmse: {rmse}")
            # TODO: THIS SHOULD ALSO LOG HISTOGRAMS FROM BEFORE MEANS


    # TODO: for now, all action inputs are the same, so just grab the first one
    if PRECISION_METRICS:
        print(f'Calculating precision metrics')
        device = "cuda"
        data_root = Path(os.path.expanduser(cfg.dataset.data_dir))
        num_samples = cfg.inference.num_wta_trials
        # generate predictions
        model.to(device)

        if cfg.model.type == "point":
            bs = 1
            data = datamodule.val_dataset[VISUALIZE_SINGLE_IDX]
            pos = data["pc"].unsqueeze(0).to(device)
            pc_action = data["pc_action"].unsqueeze(0).to(device)
            pc_anchor = data["pc_anchor"].unsqueeze(0).to(device)
            
            pc_action = (
                pc_action.transpose(-1, -2)
                .unsqueeze(1)
                .expand(-1, num_samples, -1, -1)
                .reshape(bs * num_samples, -1, cfg.dataset.sample_size)
            )
            pc_anchor = (
                pc_anchor.transpose(-1, -2)
                .unsqueeze(1)
                .expand(-1, num_samples, -1, -1)
                .reshape(bs * num_samples, -1, cfg.dataset.sample_size)
            )
            
            model_kwargs = dict(
                y=pc_anchor,
                x0=pc_action
            )
            model_kwargs, pred_actions, results = model.predict(bs, model_kwargs, num_samples, False)
            preds = pred_actions.cpu().numpy().reshape(-1, 3)

            # top - down heat map
            fig = px.density_heatmap(x=preds[:, 0], y=preds[:, 1], 
                                        nbinsx=100, nbinsy=100,
                                        title="Predicted Flows XY Heatmap")
            if SHOW_FIG:
                fig.show()
            if SAVE_FIG:
                fig.write_html("heatmap.html")

            # load val dataset at once
            val_action = []
            for i in range(len(datamodule.val_dataset)):
                val_action.append(datamodule.val_dataset[i]["pc"])
            val_action = torch.stack(val_action).to(device)

            viz_batch_idx=0
            fig = go.Figure()
            fig.add_trace(pointcloud(pc_anchor[0 + viz_batch_idx*num_samples].permute(1, 0).detach().cpu(), downsample=1, scene="scene1", name="Anchor PCD", colors=['gray'] * pc_anchor.shape[1]))
            for i in range(num_samples):
                fig.add_trace(pointcloud(pred_actions[i + viz_batch_idx*num_samples].detach().cpu(), downsample=1, scene="scene1", name=f"Predicted PCD {i}"))
            # for i in range(val_action.shape[0]):
            #     fig.add_trace(pointcloud(val_action[i].detach().cpu(), downsample=1, scene="scene1", name=f"Goal Action PCD {i}"))
            fig.update_layout(
                scene1=_3d_scene(
                    torch.cat(
                        [pred_actions[0 + viz_batch_idx*num_samples].detach().cpu(), pc_anchor[0 + viz_batch_idx*num_samples].permute(1, 0).detach().cpu()],
                        dim=0
                    ),
                    domain_scale=2
                )
            )
            if SHOW_FIG:
                fig.show()
            if SAVE_FIG:
                fig.write_html("multi_pred_pcd.html")

            precision_rmses = []
            for i in tqdm(range(pred_actions.shape[0])):
                pa = pred_actions[[i]].expand(val_action.shape[0], -1, -1)
                rmse = flow_rmse(pa, val_action, mask=False, seg=None)
                rmse_match = torch.min(rmse)
                precision_rmses.append(rmse_match)
            precision_rmses = torch.stack(precision_rmses)
            fig = px.histogram(precision_rmses.cpu().numpy(), 
                                nbins=20, title="Precision RMSE")
            if SHOW_FIG:
                fig.show()
            if SAVE_FIG:
                fig.write_html("histogram.html")
            print(precision_rmses.mean())
        elif cfg.model.type == "flow":
            data = datamodule.val_dataset[0]
            pos = data["pc"].unsqueeze(0).to(device)
            pc_anchor = data["pc_anchor"].unsqueeze(0).to(device)
            
            bs = 1
            pos = (
                pos.transpose(-1, -2)
                .unsqueeze(1)
                .expand(-1, num_samples, -1, -1)
                .reshape(bs * num_samples, -1, cfg.dataset.sample_size)
            )
            pc_anchor = (
                pc_anchor.transpose(-1, -2)
                .unsqueeze(1)
                .expand(-1, num_samples, -1, -1)
                .reshape(bs * num_samples, -1, cfg.dataset.sample_size)
            )
            
            model_kwargs = dict(
                y=pc_anchor,
                x0=pos
            )
            model_kwargs, pred_flows, results = model.predict(bs, model_kwargs, num_samples, False)
            preds = (model_kwargs["x0"] + pred_flows).cpu().numpy().reshape(-1, 3)
            
            # top - down heat map
            fig = px.density_heatmap(x=preds[:, 0], y=preds[:, 1], 
                                        nbinsx=100, nbinsy=100,
                                        title="Predicted Flows XY Heatmap")
            if SHOW_FIG:
                fig.show()
            if SAVE_FIG:
                fig.write_html("heatmap.html")
            
            # load val dataset at once
            val_flows = []
            for i in range(len(datamodule.val_dataset)):
                val_flows.append(datamodule.val_dataset[i]["flow"])
            val_flows = torch.stack(val_flows).to(device)
            
            precision_rmses = []
            for i in tqdm(range(pred_flows.shape[0])):
                pf = pred_flows[[i]].expand(val_flows.shape[0], -1, -1)
                rmse = flow_rmse(pf, val_flows, mask=False, seg=None)
                rmse_match = torch.min(rmse)
                precision_rmses.append(rmse_match)
            precision_rmses = torch.stack(precision_rmses)
            fig = px.histogram(precision_rmses.cpu().numpy(), 
                                nbins=20, title="Precision RMSE")
            if SHOW_FIG:
                fig.show()
            if SAVE_FIG:
                fig.write_html("histogram.html")
            print(precision_rmses.mean())
        else:
            raise ValueError(f"Model type {cfg.model.type} not recognized.")


    if VISUALIZE_ALL:
        import rpad.visualize_3d.plots as vpl
        pred_pcs_t = []
        gt_pcs_t = []
        model.to(device)
        
        # visualizing predictions
        for batch in tqdm(datamodule.val_dataloader()):
            pred_actions_wta, pred_actions, _, _ = model.predict_wta(batch, "val")
            
            if cfg.model.type == "point_cross":
                pred_pc = pred_actions_wta.detach().cpu()
                gt_pc = batch["pc"]
            elif cfg.model.type == "flow_cross":
                pred_pc = batch["pc"] + pred_actions_wta.detach().cpu()
                gt_pc = batch["pc_action"]

            gt_pc_t = gt_pc.flatten(end_dim=-2).cpu().numpy()
            pred_pc_t = pred_pc.flatten(end_dim=-2).cpu().numpy()
            gt_pcs_t.append(gt_pc_t)
            pred_pcs_t.append(pred_pc_t)
        
        pred_pcs_t = np.concatenate(pred_pcs_t)
        pred_seg = np.array([np.arange(3, 19)] * cfg.dataset.sample_size).T.flatten()
        
        # Get other pcds from single example. TODO: These change across examples, change this to something better
        data = datamodule.val_dataset[0]
        
        anchor_pc = data["pc_anchor"]
        anchor_seg = np.zeros(anchor_pc.shape[0], dtype=np.int64)*1

        pos = data["pc"]
        pos_seg = np.ones(pos.shape[0], dtype=np.int64)*1
        
        action_pc = data["pc_action"]
        action_seg = np.full(action_pc.shape[0], 2, dtype=np.int64)

        fig = vpl.segmentation_fig(
            # np.concatenate((pred_pcs_t, gt_pcs_t, anchor_pc, action_pc)), 
            # np.concatenate((pred_seg, gt_seg, anchor_seg, action_seg)),
            np.concatenate((pred_pcs_t, anchor_pc, action_pc, pos)), 
            np.concatenate((pred_seg, anchor_seg, action_seg, pos_seg)),
        )
        if SHOW_FIG:
            fig.show()
        if SAVE_FIG:
            fig.write_html("viz_all.html")


    # plot single diffusion chain
    if VISUALIZE_SINGLE:
        model.to(device)
        animation = FlowNetAnimation()
        
        data = datamodule.val_dataset[VISUALIZE_SINGLE_IDX]
        if cfg.model.type == "point_cross":
            pos = data["pc"]
            pc_action = data["pc_action"]
            pc_anchor = data["pc_anchor"]
            
            pc_action = pc_action.unsqueeze(0).permute(0, 2, 1).to(device)
            pc_anchor = pc_anchor.unsqueeze(0).permute(0, 2, 1).to(device)
            
            model_kwargs = dict(
                y=pc_anchor,
                x0=pc_action, 
            )
            model_kwargs, pred_pos, results = model.predict(1, model_kwargs, 1, False)
            
            pred_pos = pred_pos[0].cpu()
            pos = pos.cpu()
            pcd = pc_action[0].permute(1, 0).cpu()
            anchor_pc = pc_anchor[0].permute(1, 0).cpu()

            fig = go.Figure()
            fig.add_trace(pointcloud(pos, downsample=1, scene="scene1", name="Goal Action PCD"))
            fig.add_trace(pointcloud(anchor_pc, downsample=1, scene="scene1", name="Anchor PCD"))
            fig.add_trace(pointcloud(pcd, downsample=1, scene="scene1", name="Context Action PCD"))
            fig.add_trace(pointcloud(pred_pos, downsample=1, scene="scene1", name="Predicted PCD"))
            fig.update_layout(scene1=_3d_scene(torch.cat([pred_pos, pos, anchor_pc], dim=0).detach().cpu()))
            if SHOW_FIG:
                fig.show()
            if SAVE_FIG:
                fig.write_html("pcd.html")
        elif cfg.model.type == "flow_cross":
            pos = data["pc"]
            pc_anchor = data["pc_anchor"]
            pc_action = data["pc_action"]
            
            pos = pos.unsqueeze(0).permute(0, 2, 1).to(device)
            pc_anchor = pc_anchor.unsqueeze(0).permute(0, 2, 1).to(device)
            pc_action = pc_action.unsqueeze(0).permute(0, 2, 1).to(device)
            
            model_kwargs = dict(
                y=pc_anchor,
                x0=pos, 
            )
            model_kwargs, pred_flow, results = model.predict(1, model_kwargs, 1, False)
            
            pred_flow = pred_flow[0].permute(1, 0).cpu()
            pcd = pos[0].permute(1, 0).cpu()
            print(pcd.shape, pred_flow.shape)
            pred_pos = pcd + pred_flow.permute(1, 0)
            print(pred_pos.shape)
            pc_action = pc_action[0].permute(1, 0).cpu()
            pc_anchor = pc_anchor[0].permute(1, 0).cpu()
            
            combined_pcd = torch.cat([pc_anchor, pcd], dim=0)

            for noise_step in tqdm(results[0:]):
                pred_flow_step = noise_step[0].permute(1, 0).cpu()
                animation.add_trace(
                    combined_pcd,
                    [pcd],
                    [pred_flow_step],#combined_flow,
                    "red",
                )

            fig = animation.animate()
            if SHOW_FIG:
                fig.show()
            if SAVE_FIG:
                fig.write_html("animation.html")
            
            fig = go.Figure()
            fig.add_trace(pointcloud(pc_action, downsample=1, scene="scene1", name="Goal Action PCD"))
            fig.add_trace(pointcloud(pc_anchor, downsample=1, scene="scene1", name="Anchor PCD"))
            fig.add_trace(pointcloud(pcd, downsample=1, scene="scene1", name="Starting Action PCD"))
            fig.add_trace(pointcloud(pred_pos, downsample=1, scene="scene1", name="Final Predicted PCD"))
            fig.update_layout(scene1=_3d_scene(torch.cat([pred_pos.cpu(), pcd.cpu(), pc_anchor.cpu()], dim=0).detach().cpu()))
            if SHOW_FIG:
                fig.show()
            if SAVE_FIG:
                fig.write_html("pcd.html")


    if VISUALIZE_EVAL_SINGLE:
        model.to(device)
        animation = FlowNetAnimation()
        
        data = datamodule.val_dataset[VISUALIZE_SINGLE_IDX]
        if cfg.model.type == "point_cross":
            pos = data["pc"]
            pc_action = data["pc_action"]
            pc_anchor = data["pc_anchor"]
            
            pc_action = pc_action.unsqueeze(0).permute(0, 2, 1).to(device)
            pc_anchor = pc_anchor.unsqueeze(0).permute(0, 2, 1).to(device)
            
            print(f'PC Action: {pc_action.shape}')
            print(f'PC Anchor: {pc_anchor.shape}')
            
            model_kwargs = dict(
                y=pc_anchor,
                x0=pc_action, 
            )
            model_kwargs, pred_pos, results = model.predict(1, model_kwargs, 1, False)
            
            pred_pos = pred_pos[0].cpu()
            pos = pos.cpu()
            pcd = pc_action[0].permute(1, 0).cpu()
            anchor_pc = pc_anchor[0].permute(1, 0).cpu()
            
            start_to_pred_flows = pred_pos - pcd
            start_to_pred_tf = flow_to_tf(pcd.unsqueeze(0), start_to_pred_flows.unsqueeze(0))
            
            eval_pred_pos = start_to_pred_tf.transform_points(pcd.unsqueeze(0)).squeeze(0)

            pred_vs_eval_rmse = pcd_rmse(pred_pos, eval_pred_pos)
            print(f'Pred vs Eval RMSE: {pred_vs_eval_rmse}')

            fig = go.Figure()
            fig.add_trace(pointcloud(pos, downsample=1, scene="scene1", name="Goal Action PCD", colors=['green'] * pos.shape[0]))
            fig.add_trace(pointcloud(anchor_pc, downsample=1, scene="scene1", name="Anchor PCD", colors=['gray'] * anchor_pc.shape[0]))
            fig.add_trace(pointcloud(pcd, downsample=1, scene="scene1", name="Context Action PCD"))
            fig.add_trace(pointcloud(pred_pos, downsample=1, scene="scene1", name="Predicted PCD", colors=['blue'] * pred_pos.shape[0]))
            fig.add_trace(pointcloud(eval_pred_pos, downsample=1, scene="scene1", name="Eval Predicted PCD", colors=['red'] * eval_pred_pos.shape[0]))
            ft = _flow_traces(pcd, start_to_pred_flows, scene="scene1", flowcolor="red", flowscale=1, name="Start to Pred Flow")
            fig.add_trace(ft[0])
            fig.add_trace(ft[1])
            fig.update_layout(scene1=_3d_scene(torch.cat([pred_pos, pos, anchor_pc], dim=0).detach().cpu()))
            if SHOW_FIG:
                fig.show()
            if SAVE_FIG:
                fig.write_html("eval_pcd.html")
        elif cfg.model.type == "flow_cross":
            pos = data["pc"]
            pc_anchor = data["pc_anchor"]
            pc_action = data["pc_action"]
            
            pos = pos.unsqueeze(0).permute(0, 2, 1).to(device)
            pc_anchor = pc_anchor.unsqueeze(0).permute(0, 2, 1).to(device)
            pc_action = pc_action.unsqueeze(0).permute(0, 2, 1).to(device)
            
            model_kwargs = dict(
                y=pc_anchor,
                x0=pos, 
            )
            model_kwargs, pred_flow, results = model.predict(1, model_kwargs, 1, False)
            
            pred_flow = pred_flow[0].permute(1, 0).cpu()
            pcd = pos[0].permute(1, 0).cpu()
            print(pcd.shape, pred_flow.shape)
            pred_pos = pcd + pred_flow.permute(1, 0)
            print(pred_pos.shape)
            pc_action = pc_action[0].permute(1, 0).cpu()
            pc_anchor = pc_anchor[0].permute(1, 0).cpu()
            
            combined_pcd = torch.cat([pc_anchor, pcd], dim=0)

            for noise_step in tqdm(results[0:]):
                pred_flow_step = noise_step[0].permute(1, 0).cpu()
                animation.add_trace(
                    combined_pcd,
                    [pcd],
                    [pred_flow_step],#combined_flow,
                    "red",
                )

            fig = animation.animate()
            if SHOW_FIG:
                fig.show()
            if SAVE_FIG:
                fig.write_html("eval_animation.html")
            
            fig = go.Figure()
            fig.add_trace(pointcloud(pc_action, downsample=1, scene="scene1", name="Goal Action PCD"))
            fig.add_trace(pointcloud(pc_anchor, downsample=1, scene="scene1", name="Anchor PCD"))
            fig.add_trace(pointcloud(pcd, downsample=1, scene="scene1", name="Starting Action PCD"))
            fig.add_trace(pointcloud(pred_pos, downsample=1, scene="scene1", name="Final Predicted PCD"))
            fig.update_layout(scene1=_3d_scene(torch.cat([pred_pos.cpu(), pcd.cpu(), pc_anchor.cpu()], dim=0).detach().cpu()))
            if SHOW_FIG:
                fig.show()
            if SAVE_FIG:
                fig.write_html("eval_pcd.html")

if __name__ == "__main__":
    main()