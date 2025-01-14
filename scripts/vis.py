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
from plotly import graph_objects as go

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
    bs *= cfg.dataset.num_anchors

    cfg.inference.batch_size = bs
    cfg.inference.val_batch_size = bs
    cfg.dataset.sample_size_action = -1
    # cfg.dataset.sample_size_anchor = -1

    ######################################################################
    # Create the datamodule. This is just to initialize the datasets - we are
    # not going to use the dataloaders, because we need to manually downsample 
    # and batch.
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
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items() if k.startswith("network.")}
    )
    # TODO: hacky bugfix for load weights for ref frame predictor; probably need module-specific load function
    if cfg.model.predict_ref_frame:
        model.ref_frame_predictor.load_state_dict(
            {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items() if k.startswith("ref_frame_predictor.")}
        )
    # set model to eval mode
    network.eval()
    model.eval()

    ######################################################################
    # Helper function to run evals for a given dataset.
    ######################################################################
    def run_vis(dataset, model, indices):
        num_samples = cfg.inference.num_wta_trials
        eval_keys = ["pc_action", "pc_anchor", "pc", "flow", "seg", "seg_anchor", "T_action2world", "T_goal2world"]
        if cfg.model.rel_pose:
            eval_keys.append("rel_pose")
        predict_ref_frame = cfg.model.predict_ref_frame

        for i in tqdm(indices):
            # index item, and batchify
            item = dataset[i]
            batch = [{key: item[key] for key in eval_keys}]
            batch = {key: torch.stack([item[key] for item in batch]) for key in eval_keys}

            # predict
            pred_dict = model.predict(batch, num_samples, progress=False)
            # TODO: PUT EVERYTHING BACK IN THE WORLD FRAME?

            # get point clouds
            pred_pc = pred_dict["point"]["pred"].cpu().numpy()
            anchor_pc = batch["pc_anchor"].squeeze().cpu().numpy()
            action_pc = batch["pc_action"].squeeze().cpu().numpy()
            gt_pc = batch["pc"].squeeze().cpu().numpy()
            if predict_ref_frame:
                pred_ref_frame = pred_dict["ref_frame"].cpu().numpy()

            # get segmentations
            pred_seg = np.arange(3, 3 + num_samples).reshape(-1, 1).repeat(pred_pc.shape[1], axis=-1)
            anchor_seg = np.zeros(anchor_pc.shape[0])
            action_seg = np.ones(action_pc.shape[0])
            gt_seg = np.ones(gt_pc.shape[0]) * 2

            # visualize point cloud predictions
            pred_pc = np.concatenate(pred_pc, axis=0)
            pred_seg = np.concatenate(pred_seg, axis=0)
            fig = vpl.segmentation_fig(
                np.concatenate([
                    pred_pc,
                    anchor_pc,
                    action_pc,
                    gt_pc,
                ]),
                np.concatenate([
                    pred_seg,
                    anchor_seg,
                    action_seg,
                    gt_seg,
                ]).astype(int),
            )
            # add in reference frame predictions, if necessary
            if predict_ref_frame:
                ref_frame_trace = go.Scatter3d(
                    x=pred_ref_frame[:, 0],
                    y=pred_ref_frame[:, 1],
                    z=pred_ref_frame[:, 2],
                    mode="markers",
                    marker={"size": 30, "color": np.arange(3, 3 + num_samples)},
                    scene="scene",
                    showlegend=True,
                )
                fig.add_trace(ref_frame_trace)
            fig.show()


            # visualize per-point weights and residuals, if necessary
            if predict_ref_frame:
                # need anchor point cloud, and also output from ref frame predictor

                # grab logits and residuals (only need the first sample, since all have the same anchor)
                logit_residuals = pred_dict["logit_residuals"][0]
                logits = logit_residuals[:, 0]
                residuals = logit_residuals[:, 1:].cpu().numpy()
                probs = torch.nn.functional.softmax(logits, dim=0).detach().cpu().numpy()

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter3d(
                        mode="markers",
                        marker={
                            "size": 5,
                            "color": probs,
                            "colorscale": "Inferno",
                            "colorbar": dict(title="Logits"),
                        },
                        x=anchor_pc[:, 0],
                        y=anchor_pc[:, 1],
                        z=anchor_pc[:, 2],
                    ),
                )

                traces = vpl._flow_traces(
                    start=anchor_pc,
                    flows=residuals,
                    flowscale=1.0,
                    flowcolor=probs,
                )
                # just add the lines trace
                fig.add_trace(traces[0])

                fig.show()
                breakpoint()
                pass

    ######################################################################
    # Run the model on the train/val/test sets.
    ######################################################################
    train_indices = [0, 1, 2, 3, 4]
    val_indices = []
    val_ood_indices = []
    model.to(device)
    run_vis(datamodule.train_dataset, model, train_indices)
    run_vis(datamodule.val_dataset, model, val_indices)
    run_vis(datamodule.val_ood_dataset, model, val_ood_indices)


if __name__ == "__main__":
    main()