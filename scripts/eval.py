import hydra
from hydra.core.hydra_config import HydraConfig
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

from non_rigid.datasets.proc_cloth_flow import ProcClothFlowDataModule
from non_rigid.utils.vis_utils import FlowNetAnimation
from non_rigid.utils.script_utils import (
    create_model,
    create_datamodule,
    flatten_outputs,
    load_checkpoint_config_from_wandb,
)

from non_rigid.metrics.flow_metrics import flow_rmse
from non_rigid.utils.pointcloud_utils import expand_pcd
from tqdm import tqdm
import numpy as np

from pytorch3d.transforms import Transform3d
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
    task_overrides = HydraConfig.get().overrides.task
    cfg = load_checkpoint_config_from_wandb(
        cfg, 
        task_overrides, 
        cfg.wandb.entity, 
        cfg.wandb.project, 
        cfg.checkpoint.run_id
    )
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
        train_outputs, val_outputs, val_ood_outputs = trainer.predict(
            model,
            dataloaders=[
                datamodule.train_dataloader(),
                *datamodule.val_dataloader(),
            ]
            )

        for outputs_list, name in [
            (train_outputs, "train"),
            (val_outputs, "val"),
            (val_ood_outputs, "val_ood")
        ]:
            # Put everything on CPU, and flatten a list of dicts into one dict.
            out_cpu = [pytree.tree_map(lambda x: x.cpu(), o) for o in outputs_list]
            outputs = flatten_outputs(out_cpu)
            
            # Compute the metrics.
            rmse = torch.mean(outputs["rmse"])
            rmse_wta = torch.mean(outputs["rmse_wta"])
            print(f"{name} rmse: {rmse}, rmse wta: {rmse_wta}")


    if cfg.precision:
        model.to(device)
        # precision_dm = 2

        # TODO: this code chunk below currently breaks because all the datasets were 
        # trained as "real", even though they were "dedo"...sad. can find a way to fix later
        # # precision metric only for DEDO tasks
        # if cfg.dataset.source != "dedo":
        #     raise ValueError("Precision metric only implemented for DEDO tasks.")
        
        # determining number of samples and batch size based on cloth type
        num_samples = 20
        # NOTE: in theory, I would compare against all the cloths in the single set
        # However, this kind of implementation would not make extend well once I start 
        # conditioning on anchor pose - in other words, using the whole train set is not analogous 
        # to bucketizing the multi-cloth experiments, because the anchor pose is fixed in those 
        # buckets - so it's better to bucketize here as well, and have higher precision rmses, 
        # since we won't report them anyways
        if cfg.dataset.hole == "single":
            bs = 1
        elif cfg.dataset.hole == "double":
            bs = 2
        else:
            raise ValueError(f"Unknown hole type: {cfg.dataset.hole}.")
        
        # NOTE: this is very brittle - this will if a batch contains more than one cloth geometry
        # consider setting this back after precision is computed so the whole script can be run together
        cfg.dataset.sample_size_action = -1

        # set up a new datamodule for the dataloaders
        datamodule = ProcClothFlowDataModule(
            batch_size=bs,
            val_batch_size=bs,
            num_workers=cfg.resources.num_workers,
            dataset_cfg=cfg.dataset,
        )
        datamodule.setup(stage="predict")
        train_loader = datamodule.train_dataloader()
        val_loader, val_ood_loader = datamodule.val_dataloader()

        # helper function to run precision evaluation
        def precision_eval(dataloader, model):
            precision_rmses = []
            for batch in tqdm(dataloader):
                # generate predictions
                pred_dict = model.predict(batch, num_samples, progress=False)
                pred_pc = pred_dict["point"]["pred"]
                pc = batch["pc"].to(device)
                seg = batch["seg"].to(device)

                for pred in pred_pc:
                    # expand each prediction to batch size, and save minimum RMSE
                    pred = expand_pcd(pred.unsqueeze(0), bs)
                    precision_rmses.append(torch.min(flow_rmse(pred, pc, mask=True, seg=seg)))
                    

                # VISUALIZATION CODE - KEEP FOR LATER
                # pred_pc = pred_dict["point"]["pred_world"].cpu().numpy()
                # # convert batch back to world frame
                # T_action2world = Transform3d(
                #     matrix=batch["T_action2world"]
                # )
                # T_goal2world = Transform3d(
                #     matrix=batch["T_goal2world"]
                # )
                # pc_action_world = T_action2world.transform_points(batch["pc_action"]).numpy().reshape(-1, 3)
                # pc_anchor_world = T_goal2world.transform_points(batch["pc_anchor"]).numpy().reshape(-1, 3)


                # pc_action_seg = np.ones((pc_action_world.shape[0])) * 0
                # pc_anchor_seg = np.ones((pc_anchor_world.shape[0])) * 1
                # pred_pc_seg = np.ones((pred_pc.shape[0], pred_pc.shape[1])) * np.arange(pred_pc.shape[0]).reshape(-1, 1) + 2
                
                # vpl.segmentation_fig(
                #     np.concatenate([
                #         pc_action_world,
                #         pc_anchor_world,
                #         pred_pc.reshape(-1, 3),
                #     ]),
                #     np.concatenate([
                #         pc_action_seg,
                #         pc_anchor_seg,
                #         pred_pc_seg.flatten(),
                #     ]).astype(int),
            
                # ).show()
            return torch.stack(precision_rmses).mean().item()

        train_precision_rmse = precision_eval(train_loader, model)
        val_precision_rmse = precision_eval(val_loader, model)
        val_ood_precision_rmse = precision_eval(val_ood_loader, model)
        print("Train Precision RMSE: ", train_precision_rmse)
        print("Val Precision RMSE: ", val_precision_rmse)
        print("Val OOD Precision RMSE: ", val_ood_precision_rmse)

if __name__ == "__main__":
    main()