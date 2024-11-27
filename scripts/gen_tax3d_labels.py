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
import zarr
import rpad.visualize_3d.plots as vpl


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
    # For now, use a batch size of 1; this can probably be optimized later.
    cfg.inference.batch_size = 1
    cfg.inference.val_batch_size = 1
    cfg.dataset.sample_size_action = -1


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
    model.to(device)

    ######################################################################
    # Generate predictions, and update the zarr files.
    ######################################################################
    # Helper function to update the zarr file for a dataset split.
    def update_zarr(zarr_path, dataloader, model):

        data_group = zarr.open(zarr_path, mode="a")['data']
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        tax3d_pred_chunk_size = (100, 10, 625, 3)

        preds = []
        for batch in tqdm(dataloader):
            pred_dict = model.predict(batch, 10, progress=False)
            # preds.append(pred_dict["point"]["pred_world"].cpu().numpy())

            pred_world = pred_dict["point"]["pred_world"].cpu().numpy()
            pred_world = np.tile(pred_world, (301, 1, 1, 1))
            if pred_world.shape[2] < 625:
                pad = np.zeros((pred_world.shape[0], pred_world.shape[1], 625 - pred_world.shape[2], 3))
                pred_world = np.concatenate([pred_world, pad], axis=2)
            preds.append(pred_world)

        preds = np.concatenate(preds, axis=0)
        data_group.create_dataset(
            "tax3d_pred", 
            data=preds, 
            chunks=tax3d_pred_chunk_size,
            dtype='float32',
            overwrite=True,
            compressor=compressor)
        del preds

    train_dataloader = datamodule.train_dataloader()
    val_dataloader, val_ood_dataloader = datamodule.val_dataloader()
    train_zarr_path = datamodule.root / "train.zarr"
    val_zarr_path = datamodule.root / "val.zarr"
    val_ood_zarr_path = datamodule.root / "val_ood.zarr"


    update_zarr(train_zarr_path, train_dataloader, model)
    update_zarr(val_zarr_path, val_dataloader, model)
    update_zarr(val_ood_zarr_path, val_ood_dataloader, model)


if __name__ == "__main__":
    main()