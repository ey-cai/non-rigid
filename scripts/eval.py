import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
import json
import omegaconf
import torch
import wandb
from pytorch3d.transforms import Transform3d, Translate
import torch.utils._pytree as pytree

from non_rigid.utils.script_utils import (
    create_model,
    create_datamodule,
    load_checkpoint_config_from_wandb,
    flatten_outputs,
)

from non_rigid.nets.dgcnn import DGCNN
from non_rigid.metrics.flow_metrics import flow_rmse
from non_rigid.utils.pointcloud_utils import expand_pcd
from tqdm import tqdm
import numpy as np

import rpad.visualize_3d.plots as vpl
from non_rigid.models.frame_predict import (
    FramePredictorSimple,
    FramePredictorDGCNN,
    FramePredictorMLPTransformer,
)

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
    # cfg.dataset.sample_size_action = -1
    # cfg.dataset.sample_size_anchor = -1

    ######################################################################
    # Load the reference frame predictor, if necessary.
    ######################################################################
    
    if cfg.use_gmm:
        # gmm can only be used with oracle models
        if not cfg.model.oracle and not cfg.model.tax3dv2:
            raise ValueError("GMM can only be used with oracle models or TAX3Dv2 models.")
        
        # cannot predict and diffuse reference frame together
        if cfg.model.diffuse_ref_frame and not cfg.model.tax3dv2:
            raise ValueError("Can only predict and diffuse reference frame together for TAX3Dv2.")

        import torch_geometric.data as tgd
        import os

        ref_frame_predictor = FramePredictorMLPTransformer()
        checkpoint_dir = os.path.expanduser("~/non-rigid-robot/notebooks/mlp_transformer_epochs=5000_var=1.0_uniform_loss=0.1/checkpoints/")
        gmm_ckpt = torch.load(checkpoint_dir + "model_5000.pt", map_location=device)

        ref_frame_predictor.load_state_dict(gmm_ckpt)
        ref_frame_predictor.eval()
        ref_frame_predictor.to(device)

        # update the dataset_cfg if needed
        cfg.dataset.oracle = False
        cfg.model.oracle = False

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

        if cfg.checkpoint.alias == "v0":
            model_name = "model.ckpt"
        elif cfg.checkpoint.alias == "monitor":
            # getting artifact names, and sanity checking monitor name
            artifact_file_names = [f.name for f in artifact.files()]
            monitor_name = cfg.checkpoint.monitor_name
            if not isinstance(monitor_name, str):
                raise ValueError(f"Invalid monitor name: {monitor_name}. Must be a string.")
            
            # searching for checkpoints with exact monitor name - should only be one for now.
            valid_artifact_file_names = [f for f in artifact_file_names if 
                                         f.split("-")[2].split("=")[0] == monitor_name]
            if len(valid_artifact_file_names) == 0:
                raise ValueError(f"Could not find any files with monitor name: {monitor_name}.")
            elif len(valid_artifact_file_names) > 1:
                raise ValueError(f"Found multiple files with monitor name: {monitor_name}.")
            else:
                model_name = valid_artifact_file_names[0]
        else:
            raise ValueError(f"Invalid checkpoint alias: {cfg.checkpoint.alias}.")
        ckpt_file = artifact.get_path(model_name).download(root=artifact_dir)
    else:
        ckpt_file = checkpoint_reference
    # Load the network weights.
    ckpt = torch.load(ckpt_file, map_location=device)
    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items() if k.startswith("network.")}
    )
    # set model to eval mode
    network.eval()
    model.eval()

    ######################################################################
    # Helper function to run evals for a given dataset.
    ######################################################################
    def run_eval(dataset, model):
        num_samples = cfg.inference.num_wta_trials # // bs
        num_batches = len(dataset) // bs
        eval_keys = ["pc_action", "pc_anchor", "pc", "flow", "seg", "seg_anchor", "T_action2world", "T_goal2world"]
        if cfg.model.diffuse_ref_frame:
            eval_keys.append("goal_origin")
        if cfg.model.rel_pose:
            eval_keys.append("rel_pose")
            
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
                item = dataset.__getitem__(i * bs + j, use_indices=downsample_indices)
                batch_list.append({key: item[key] for key in eval_keys})

            # convert to batch
            batch = {key: torch.stack([item[key] for item in batch_list]) for key in eval_keys}
            # generate predictions
            if cfg.use_gmm:
                # expand action and anchor point clouds
                gmm_action = expand_pcd(batch["pc_action"], num_samples)
                gmm_anchor = expand_pcd(batch["pc_anchor"], num_samples)

                if "rel_pose" in batch:
                    gmm_rel_pose = expand_pcd(batch["rel_pose"], num_samples)
                    gmm_batch = tgd.Batch.from_data_list([
                        tgd.Data(
                            x=gmm_anchor[i],
                            pos=gmm_anchor[i],
                            action=gmm_action[i],
                            rel_pose=gmm_rel_pose[i],
                        ) for i in range(bs * num_samples)
                    ]).to(device)
                else:
                    gmm_batch = tgd.Batch.from_data_list([
                        tgd.Data(
                            x=gmm_anchor[i],
                            pos=gmm_anchor[i],
                            action=gmm_action[i],
                        ) for i in range(bs * num_samples)
                    ]).to(device)

                # sample reference frames for WTA
                gmm_pred = ref_frame_predictor(gmm_batch)
                gmm_probs, gmm_means = gmm_pred["probs"], gmm_pred["means"]
                idxs = torch.multinomial(gmm_probs.squeeze(-1), 1).squeeze()
                sampled_ref_frames = gmm_means[torch.arange(bs * num_samples), idxs].unsqueeze(-2)
                sampled_ref_frames = sampled_ref_frames.cpu()
                batch["ref_frame"] = sampled_ref_frames

            pred_dict = model.predict(batch, num_samples, progress=False, full_prediction=True)
            pred_point_world = pred_dict["point"]["pred_world"]

            # if diffusing reference frame, update prediction
            if cfg.model.diffuse_ref_frame:
                gt_ref_frame = batch["goal_origin"].unsqueeze(-2).to(device)
                # pred_ref_frame = pred_dict["ref_frame"]
                # pred_pc = pred_pc + pred_ref_frame

            batch_rmse = torch.zeros(bs, cfg.inference.num_wta_trials * bs)

            for j in range(bs):
                # expand ground truth pc to compute RMSE for cloth-specific sample
                gt_pc = batch["pc"][j].unsqueeze(0).to(device)
                seg = batch["seg"][j].unsqueeze(0).to(device)
                gt_pc = expand_pcd(gt_pc, num_samples * bs)
                seg = expand_pcd(seg, num_samples * bs)

                # if predicting reference frame, update ground truth
                if cfg.use_gmm and not cfg.model.tax3dv2:
                    gt_pc = gt_pc - sampled_ref_frames.to(device)

                # if diffusing reference frame, update ground truth
                if cfg.model.diffuse_ref_frame:
                    gt_pc = gt_pc + expand_pcd(gt_ref_frame[j].unsqueeze(0), num_samples * bs)
                
                # put ground truth in world frame
                T_goal2world = Transform3d(
                    matrix=expand_pcd(batch["T_goal2world"][j].unsqueeze(0).to(device), num_samples * bs) 
                )
                gt_pc_world = T_goal2world.transform_points(gt_pc)

                seg = seg == 0
                batch_rmse[j] = flow_rmse(pred_point_world, gt_pc_world, mask=True, seg=seg)

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



    def simple_eval(datamodule, model):
        trainer = L.Trainer(
            accelerator="gpu",
            devices=cfg.resources.gpus,
            precision="32-true",
            logger=False,
        )

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
            (val_ood_outputs, "val_ood"),
        ]:
            out_cpu = [pytree.tree_map(lambda x: x.cpu(), o) for o in outputs_list]
            outputs = flatten_outputs(out_cpu)

            rmse = torch.mean(outputs["rmse"])
            rmse_wta = torch.mean(outputs["rmse_wta"])
            print(f"{name} RMSE: {rmse}, WTA RMSE: {rmse_wta}")

    ######################################################################
    # Run the model on the train/val/test sets.
    ######################################################################
    model.to(device)

    # simple_eval(datamodule, model)
    # quit()

    train_rmse, train_coverage, train_precision = run_eval(datamodule.train_dataset, model)
    val_rmse, val_coverage, val_precision = run_eval(datamodule.val_dataset, model)
    val_ood_rmse, val_ood_coverage, val_ood_precision = run_eval(datamodule.val_ood_dataset, model)

    print(f"Train RMSE: {train_rmse}, Coverage: {train_coverage}, Precision: {train_precision}")
    print(f"Val RMSE: {val_rmse}, Coverage: {val_coverage}, Precision: {val_precision}")
    print(f"Val OOD RMSE: {val_ood_rmse}, Coverage: {val_ood_coverage}, Precision: {val_ood_precision}")

if __name__ == "__main__":
    main()