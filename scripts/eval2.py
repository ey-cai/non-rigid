import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
import json
import omegaconf
import torch
import wandb

from non_rigid.utils.script_utils import (
    create_model,
    create_datamodule,
    load_checkpoint_config_from_wandb,
)

from non_rigid.metrics.flow_metrics import flow_rmse
from non_rigid.utils.pointcloud_utils import expand_pcd
from tqdm import tqdm
import numpy as np

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
    # Manually setting eval-specific configs.
    ######################################################################
    # Using a custom cloth-specific batch size, to allow for simultaneous evaluation 
    # of RMSE, coverage, and precision.
    if cfg.dataset.hole == "single":
        bs = 1
    elif cfg.dataset.hole == "double":
        bs = 2
    else:
        raise ValueError(f"Unknown hole type: {cfg.dataset.hole}.")
    cfg.inference.batch_size = bs
    cfg.inference.val_batch_size = bs
    # also, turn off all downsampling - need custom downsampling for precision
    #true_sample_size_action = cfg.dataset.sample_size_action
    #true_sample_size_anchor = cfg.dataset.sample_size_anchor
    #cfg.dataset.sample_size_action = -1
    #cfg.dataset.sample_size_anchor = -1


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
    # Helper function to run evals for a given dataset.
    ######################################################################
    def run_eval(dataset, model):
        num_samples = cfg.inference.num_wta_trials // bs
        num_batches = len(dataset) // bs
        eval_keys = ["pc_action", "pc_anchor", "pc", "flow", "seg", "seg_anchor", "T_action2world", "T_goal2world"]

        rmse = []
        coverage = []
        precision = []


        for i in tqdm(range(num_batches)):
            batch_list = []

            # get first item in batch, and keep downsampling indices
            item = dataset.__getitem__(i * bs, return_indices=True)
            downsample_indices = {
                "action_pc_indices": item["action_pc_indices"],
                "anchor_pc_indices": item["anchor_pc_indices"],
            }
            batch_list.append({key: item[key] for key in eval_keys})

            # get the rest of the batch
            for j in range(1, bs):
                item = dataset.__getitem__(i * bs + j, downsample_indices=downsample_indices)
                batch_list.append({key: item[key] for key in eval_keys})

            # convert to batch
            batch = {key: torch.stack([item[key] for item in batch_list]) for key in eval_keys}

            # generate predictions
            pred_dict = model.predict(batch, num_samples, progress=False)
            pred_pc = pred_dict["point"]["pred"]
            # pc = batch["pc"].to(device)
            # seg = batch["seg"].to(device)

            batch_rmse = torch.zeros(bs, cfg.inference.num_wta_trials)

            for j in range(bs):
                # expand ground truth pc to compute RMSE for cloth-specific sample
                gt_pc = batch["pc"][j].unsqueeze(0).to(device)
                seg = batch["seg"][j].unsqueeze(0).to(device)
                gt_pc = expand_pcd(gt_pc, num_samples)
                seg = expand_pcd(seg, num_samples)
                batch_rmse[j] = flow_rmse(pred_pc, gt_pc, mask=True, seg=seg)

            # computing precision and coverage
            batch_precision = torch.min(batch_rmse, dim=0).values
            batch_coverage = torch.min(batch_rmse, dim=1).values

            # update dataset-wide metrics
            rmse.append(batch_rmse.mean().item())
            coverage.append(batch_coverage.mean().item())
            precision.append(batch_precision.mean().item())
            
        rmse = np.mean(rmse)
        coverage = np.mean(coverage)
        precision = np.mean(precision)
        return rmse, coverage, precision


    ######################################################################
    # Run the model on the train/val/test sets.
    ######################################################################
    model.to(device)
    train_rmse, train_coverage, train_precision = run_eval(datamodule.train_dataset, model)
    val_rmse, val_coverage, val_precision = run_eval(datamodule.val_dataset, model)
    val_ood_rmse, val_ood_coverage, val_ood_precision = run_eval(datamodule.val_ood_dataset, model)

    print(f"Train RMSE: {train_rmse}, Coverage: {train_coverage}, Precision: {train_precision}")
    print(f"Val RMSE: {val_rmse}, Coverage: {val_coverage}, Precision: {val_precision}")
    print(f"Val OOD RMSE: {val_ood_rmse}, Coverage: {val_ood_coverage}, Precision: {val_ood_precision}")

if __name__ == "__main__":
    main()