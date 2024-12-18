# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

from omegaconf import OmegaConf

# %%
from non_rigid.datasets.proc_cloth_flow import DeformablePlacementDataset

dataset = DeformablePlacementDataset(
    root=Path(
        "/home/beisner/datasets/tax3d_data/proccloth/cloth=single-fixed anchor=single-random hole=single"
    ),
    dataset_cfg=OmegaConf.create(
        {
            "train_size": 1,
            "scene": True,
            "sample_size_action": 1024,
            "sample_size_anchor": 1024,
            # "sample_size_action": 0,
            # "sample_size_anchor": 0,
            "world_frame": False,
            "source": "dedo",
            "anchor_occlusion": False,
            "rotation_variance": 0.0,
            "translation_variance": 0.0,
            "action_transform_type": "identity",
            "anchor_transform_type": "identity",
            "center_type": "anchor_center",
            "action_context_center_type": "center",
            "downsample_type": "fps",
        }
    ),
    split="train_tax3d",
)
print(dataset)

# %%
import itertools

# %%
data = dataset[0]
for key, value in data.items():
    print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")

# %%
import torch


def get_scale(data):
    return data.max(dim=0).values - data.min(dim=0).values


def get_mean(data):
    return data.mean(dim=0)


scale = get_scale(data["pc_action"])
mean = get_mean(data["pc_action"])
print(scale)
print(mean)


class RescaleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, scale, mean):
        self.dataset = dataset
        self.scale = scale
        self.mean = mean

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data["pc_action"] = (data["pc_action"] - self.mean) / self.scale
        data["flow"] = data["flow"] / self.scale
        return data


import torch

# %%
from rpad.visualize_3d.plots import segmentation_fig

dataset2 = RescaleDataset(dataset, scale, mean)
data2 = dataset2[0]

fig = segmentation_fig(
    data=torch.cat(
        [
            data2["pc_action"],
            # data2["pc_anchor"],
            data2["pc_action"] + data2["flow"],
        ]
    ),
    labels=torch.cat(
        [
            torch.zeros(data2["pc_action"].shape[0]),
            # torch.ones(data2["pc_anchor"].shape[0]),
            torch.ones(data2["pc_action"].shape[0]) * 2,
        ]
    ).int(),
    labelmap={
        0: "action",
        1: "anchor",
        2: "action + flow",
    },
)
fig

# %%
import lightning as L
from PointTransformerV3.model import Point, PointTransformerV3

from non_rigid.models.dit.models import PTv3_xs

tformer = PointTransformerV3(enable_flash=False, in_channels=3)
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class PTv3RegressionModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        # self.model = PointTransformerV3(
        #     enable_flash=False,
        #     in_channels=3,
        #     # enc_num_head=(2, 4, 8, 16, 32),
        #     # dec_num_head=(4, 4, 8, 16),
        #     # enc_num_head=(1, 2, 4, 8, 16),
        #     # dec_num_head=(2, 2, 4, 8),
        #     # enc_patch_size=(256, 256, 256, 256, 256),
        #     # dec_patch_size=(256, 256, 256, 256),
        #     # enable_rpe=True,
        #     stride=(2, 2, 2),
        #     enc_depths=(2, 2, 2, 6),
        #     enc_channels=(32, 64, 128, 256),
        #     enc_num_head=(2, 4, 8, 16),
        #     enc_patch_size=(1024, 1024, 1024, 1024),
        #     dec_depths=(2, 2, 2),
        #     dec_channels=(64, 64, 128),
        #     dec_num_head=(4, 4, 8),
        #     dec_patch_size=(1024, 1024, 1024),
        #     )
        self.model = PTv3_xs()
        # self.final_linear = nn.Linear(64, 3)
        # 3 layer MLP
        self.final_linear = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, batch):
        # B, _, C = batch["pc_action"].shape
        # # full_pc = torch.cat([batch["pc_action"], batch["pc_anchor"]], dim=1)
        # full_pc = batch["pc_action"]

        # # Reshape to (BxN, C), and create a batch vector with the indices.
        # # Right now assuming that the batch has same number of points for each example.
        # full_pc_squashed = full_pc.view(-1, C)
        # full_pc_batch = torch.repeat_interleave(
        #     torch.arange(B, device=full_pc.device), full_pc.shape[1]
        # )

        # # TO device.
        # full_pc_squashed = full_pc_squashed.to(self.device)
        # full_pc_batch = full_pc_batch.to(self.device)

        # data = Point(
        #     coord=full_pc_squashed,
        #     feat=full_pc_squashed,
        #     batch=full_pc_batch,
        #     grid_size=0.001,
        # )
        # pred = self.model(data)

        # # Only need the action points.
        # # Reshape back to (B, N, C)
        # feats = pred.feat.view(B, -1, 64)
        # action_feats = feats[:, :batch["pc_action"].shape[1]]
        action_feats = self.model(batch["pc_action"].to(self.device))

        return self.final_linear(action_feats)

    def training_step(self, batch, batch_idx):
        data = batch
        output = self(data)
        target = data["flow"]
        loss = F.mse_loss(output, target)
        self.log_dict(
            {"train/loss": loss},
            add_dataloader_idx=False,
            prog_bar=True,
        )
        return loss

    def predict(self, batch, num_samples=10):
        # Predict multiple times and average.
        preds = []
        for _ in range(num_samples):
            preds.append(self(batch))
        return torch.stack(preds, dim=0)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)


# %%
# from torch.utils.data import DataLoader
# dataloader = DataLoader(itertools.islice(itertools.cycle(dataset), 1000), batch_size=16, shuffle=True)
# batch = next(iter(dataloader))

# device = "cuda:1"

# mod = PTv3RegressionModule().cuda()
# res = mod(batch)

# %%
from non_rigid.models.regression import RegressionModule, RegressionNetwork
from non_rigid.models.tax3d import CrossDisplacementModule, DiffusionTransformerNetwork

NUM_TRAINING_STEPS = 100000
REGRESSION = False

if REGRESSION:
    model_cfg = OmegaConf.create(
        {
            "name": "regression",
            "type": "flow",
            "size": "xS",
            "rotary": False,
            "center_noise": False,
            "in_channels": 3,
            "learn_sigma": False,
            "x_encoder": "mlp",
            "y_encoder": "mlp",
            "x0_encoder": None,
            "diff_train_steps": 100,
        }
    )
    network = RegressionNetwork(model_cfg=model_cfg)
    model = RegressionModule(
        network,
        cfg=OmegaConf.create(
            {
                "mode": "train",
                "prediction_type": "flow",
                "model": model_cfg,
                "training": {
                    "lr": 1e-3,
                    "weight_decay": 1e-5,
                    "num_training_steps": NUM_TRAINING_STEPS,
                    "lr_warmup_steps": 100,
                    "additional_train_logging_period": 1000,
                    "batch_size": 1,
                    "val_batch_size": 1,
                    "sample_size": None,
                    "sample_size_anchor": None,
                    "num_wta_trials": 10,
                },
            }
        ),
    )
else:
    model_cfg = OmegaConf.create(
        {
            "name": "df_cross",
            "type": "flow",
            "size": "xS",
            "rotary": False,
            "center_noise": False,
            "in_channels": 3,
            "learn_sigma": True,
            "x_encoder": "mlp",
            # "y_encoder": "mlp",
            # "x0_encoder": "mlp",
            "y_encoder": "ptv3_standalone",
            "x0_encoder": "ptv3_standalone",
            "diff_train_steps": 100,
            "diff_inference_steps": 100,
            "diff_noise_scale": 1,
            "diff_noise_schedule": "linear",
            "diff_type": "gaussian",
        }
    )
    network = DiffusionTransformerNetwork(model_cfg=model_cfg)
    model = CrossDisplacementModule(
        network,
        cfg=OmegaConf.create(
            {
                "mode": "train",
                "prediction_type": "flow",
                "model": model_cfg,
                "training": {
                    "lr": 1e-4,
                    "weight_decay": 1e-5,
                    "num_training_steps": NUM_TRAINING_STEPS,
                    "lr_warmup_steps": 100,
                    "additional_train_logging_period": 1000,
                    "batch_size": 1,
                    "val_batch_size": 1,
                    "sample_size": None,
                    "sample_size_anchor": None,
                    "num_wta_trials": 10,
                },
            }
        ),
    )


# model = PTv3RegressionModule()


# %%
import itertools

# %%
from non_rigid.models.diptv3 import DiPTv3, DiPTv3Adapter

# %%
# Import default_collate from torch


class IterableDatasetWrapper(torch.utils.data.IterableDataset):
    def __init__(self, iterable, length):
        self.iterable = iterable
        self.length = length

    def __iter__(self):
        return iter(self.iterable)

    def __len__(self):
        return self.length


# Example usage
# repeat_count = 1000
# batch_size = 6
repeat_count = 1000
batch_size = 8

# Create an iterable dataset with a fixed length
wrapped_dataset = IterableDatasetWrapper(
    itertools.islice(itertools.cycle(dataset), repeat_count), repeat_count
)


class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, repeat_count):
        self.dataset = dataset
        self.repeat_count = repeat_count

    def __getitem__(self, index):
        item = self.dataset[index % len(self.dataset)]

        # print(item["pc_action"][0, 0])
        return item

    def __len__(self):
        return len(self.dataset) * self.repeat_count


# Create a DataLoader on top of the wrapped dataset

# Dataloader on top of dataset[0]
from torch.utils.data import DataLoader

dataloader = DataLoader(
    RescaleDataset(RepeatDataset(dataset, repeat_count), scale, mean),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)
# dataloader = DataLoader(itertools.islice(itertools.cycle(dataset), 1000), batch_size=16, shuffle=True)
# dataloader = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=False)
batch = next(iter(dataloader))

# device = "cuda"
# model = model.cuda()
# batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# optimizers, schedulers = model.configure_optimizers()


# %%
batch["flow"].shape

# %%
from non_rigid.models.tax3d import SceneDisplacementModule

adapter = DiPTv3Adapter(
    DiPTv3(
        enable_flash=False,
    )
).cuda()


preds = adapter(
    x=batch["flow"].permute(0, 2, 1).cuda(),
    t=torch.rand(
        batch["flow"].shape[0],
    )
    .float()
    .cuda(),
    x0=batch["pc_action"].permute(0, 2, 1).cuda(),
)


model_cfg = OmegaConf.create(
    {
        "name": "df_diptv3",
        "type": "flow",
        # "size": "xS",
        # "rotary": False,
        # "center_noise": False,
        # "in_channels": 3,
        "learn_sigma": False,
        # "x_encoder": "mlp",
        # # "y_encoder": "mlp",
        # # "x0_encoder": "mlp",
        # "y_encoder": "ptv3_standalone",
        # "x0_encoder": "ptv3_standalone",
        "diff_train_steps": 100,
        "diff_inference_steps": 100,
        "diff_noise_scale": 1,
        "diff_noise_schedule": "linear",
        "diff_type": "gaussian",
    }
)
network = adapter
model = SceneDisplacementModule(
    network,
    cfg=OmegaConf.create(
        {
            "mode": "train",
            "prediction_type": "flow",
            "model": model_cfg,
            "training": {
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "num_training_steps": NUM_TRAINING_STEPS,
                "lr_warmup_steps": 0,
                "additional_train_logging_period": 1000,
                "batch_size": batch_size,
                "val_batch_size": batch_size,
                "sample_size": None,
                "sample_size_anchor": None,
                "num_wta_trials": 10,
            },
        }
    ),
)

# %%
preds.shape

# %%
del preds

# %%
# Garbage collect
import gc

gc.collect()

# GC with torch.cuda.empty_cache()
torch.cuda.empty_cache()

# %%
# Using the trainer...

import time

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

time_now_str = time.strftime("%Y-%m-%d_%H-%M-%S")
torch.set_float32_matmul_precision("high")
trainer = L.Trainer(
    accelerator="gpu",
    devices=[0],
    # precision="16-mixed",
    precision="32-true",
    max_epochs=NUM_TRAINING_STEPS,
    # max_epochs=1000,
    logger=False,
    check_val_every_n_epoch=0,
    # log_every_n_steps=2, # TODO: MOVE THIS TO TRAINING CFG
    log_every_n_steps=1,
    gradient_clip_val=1.0,
    callbacks=[
        ModelCheckpoint(dirpath=f"./ckpts/{time_now_str}", every_n_train_steps=100),
    ],
)

trainer.fit(model, dataloader)


# %%


# %%
model

# %%
from tqdm import tqdm

losses = []
with tqdm(range(NUM_TRAINING_STEPS)) as pbar:
    for i in pbar:
        loss = model.training_step(batch)
        optimizers[0].zero_grad()
        loss.backward()

        losses.append(loss.item())

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizers[0].step()

        # Scheduler step
        schedulers[0].step()

        if i % 10 == 0:
            pbar.set_description(f"Step {i}, loss: {loss}")


# %%
# with_clipping_losses = losses
without_clipping_losses = losses

# %%
import matplotlib.pyplot as plt

plt.plot(without_clipping_losses, label="Without clipping")
plt.plot(with_clipping_losses, label="With clipping")
plt.legend()

plt.show()

# %%
data["pc_action"].shape

# %%
pred_point[0] - (data["pc_action"] + pred_flow[0])

# %%
og_action - batch["pc_action"]

# %%
prediction["point"]["pred"].cpu().shape

# %%
for key, value in batch.items():
    print(
        f"{key}: {value.shape}"
        if isinstance(value, torch.Tensor)
        else f"{key}: {value}"
    )

# %%
import tree

batch = tree.map_structure(
    lambda x: x[0:1] if isinstance(x, torch.Tensor) else x, batch
)


# %%
batch["flow"].shape

# %%
pred_point

# %%
# Make a prediction
model.cuda()
model.eval()

import tree

# batch = tree.map_structure(lambda x: x[0] if isinstance(x, torch.Tensor) else x, batch)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataloader = DataLoader(
    RescaleDataset(dataset, scale, mean), batch_size=1, shuffle=False
)
batch = next(iter(dataloader))

with torch.no_grad():
    prediction = model.predict(batch, num_samples=1)
    # prediction = model.predict(batch, num_samples=10)[:, 0]

# pred_flow = prediction["flow"]["pred"].cpu()
pred_point = prediction["point"]["pred"].cpu()
# og_action = prediction["pc_action_og"].cpu()

# pred_flow = prediction.cpu()

fig = segmentation_fig(
    data=torch.cat(
        [
            batch["pc_action"][0],
            # batch["pc_anchor"][0],
            batch["pc_action"][0] + batch["flow"][0],
            # *[batch["pc_action"][0] + pred_flow[i] for i in range(pred_flow.shape[0])],
            *[pred_point[i] for i in range(pred_point.shape[0])],
        ]
    ),
    labels=torch.cat(
        [
            torch.zeros(data["pc_action"].shape[0]),
            # torch.ones(data["pc_anchor"].shape[0]),
            torch.ones(data["pc_action"].shape[0]) * 2,
            *[
                torch.ones(data["pc_action"].shape[0]) * (3 + i)
                for i in range(pred_point.shape[0])
            ],
        ]
    ).int(),
    labelmap={
        0: "action",
        1: "anchor",
        2: "action + gt flow",
        **{3 + i: f"action + pred flow {i}" for i in range(pred_point.shape[0])},
    },
)
fig

# %%
prediction["results"][:1][0].shape

# %%


# %%
import plotly.graph_objects as go

# Create a figure
fig = go.Figure()

# Add traces for each frame
for i, pts in enumerate(prediction["results"][:1]):
    fig.add_trace(
        go.Scatter3d(
            x=pts[0][:, 0].cpu().numpy(),
            y=pts[0][:, 1].cpu().numpy(),
            z=pts[0][:, 2].cpu().numpy(),
            mode="markers",
            marker=dict(size=2),
            name=f"Frame {i}",
        )
    )

# Create frames for the animation
frames = [
    go.Frame(
        data=[
            go.Scatter3d(
                x=pts[0][:, 0].cpu(),
                y=pts[0][:, 1].cpu(),
                z=pts[0][:, 2].cpu(),
                mode="markers",
                marker=dict(size=2),
            )
        ]
    )
    for pts in prediction["results"]
]

# Add frames to the figure
fig.frames = frames

# Add animation settings
fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True,
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]
)

# Set the layout for the 3D scatter plot
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-100, 100], autorange=False),
        yaxis=dict(range=[-100, 100], autorange=False),
        zaxis=dict(range=[-100, 100], autorange=False),
    ),
    title="Point Cloud Animation",
    width=800,
    height=800,
)

# Show the figure
fig.show()

# %%
preds = model(batch)[0].cpu()
gt = batch["flow"]

print(f"gt - preds: {(((gt - preds) ** 2).sum(dim=-1).sqrt()).mean()}")

# %%
gt

# %%
((preds - gt) ** 2).sum(dim=-1).sqrt().max()

# %%
# Get Average RMSE of the predictions
prediction = model.predict_wta(batch, num_samples=10)
print(f"RMSE: {prediction['rmse'].cpu()}")
print(f"RMSE wta: {prediction['rmse_wta'].item()}")

# %%


# %%
import wandb

from non_rigid.utils.script_utils import create_model, load_checkpoint_config_from_wandb

# MOdel ID to verify against
model_id = "kr93ivph"

cfg = load_checkpoint_config_from_wandb(
    OmegaConf.create(
        {
            "mode": "train",
            "wandb": {
                "entity": "r-pad",
                "project": "non-rigid",
                "artifact_dir": "/home/beisner/artifacts",
            },
            "checkpoint": {
                "reference": "r-pad/non_rigid/model-kr93ivph:v0",
            },
            "model": model_cfg,
            "training": {
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "num_training_steps": NUM_TRAINING_STEPS,
                "lr_warmup_steps": 100,
                "additional_train_logging_period": 1000,
                "batch_size": 1,
                "val_batch_size": 1,
                "sample_size": None,
                "sample_size_anchor": None,
                "num_wta_trials": 10,
            },
            "dataset": {
                "data_dir": "/home/beisner/datasets/tax3d_data/proccloth/cloth=single-fixed anchor=single-random hole=single",
                "train_size": 1,
                "scene": False,
                "sample_size_action": 512,
                "sample_size_anchor": 512,
                "world_frame": False,
                "source": "dedo",
                "anchor_occlusion": False,
                "rotation_variance": 0.0,
                "translation_variance": 0.0,
                "action_transform_type": "identity",
                "anchor_transform_type": "identity",
                "center_type": "anchor_center",
                "action_context_center_type": "center",
                "downsample_type": "fps",
            },
        }
    ),
    {},
    "r-pad",
    "non_rigid",
    model_id,
)


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

# %%
model = model.cuda()
ckpt_model_preds = model.predict_wta(batch, num_samples=10)
print(f"RMSE: {ckpt_model_preds['rmse'].cpu()}")
print(f"RMSE wta: {ckpt_model_preds['rmse_wta'].item()}")

# %%
device = "cuda:0"

# %%
from PointTransformerV3.model import Point, PointTransformerV3

tformer = PointTransformerV3(enable_flash=False, in_channels=3)

# %%
tformer.cuda()

# %%
data["pc_action"].shape

# %%
import torch

point = Point(
    coord=data["pc_action"].cuda(),
    grid_size=0.01,
    batch=torch.zeros(data["pc_action"].shape[0]).long().cuda(),
)
point["feat"] = point["coord"]

# %%
pred = tformer(point)

# %%
pred.feat.shape

# %%
pts_t[1].shape

# %%
pts_t[1].max(dim=0).values


# %%
def noised_points(points, t, alpha_start=1e-5, alpha_end=(1 - 0.02), noise_scale=1):
    noise = torch.randn_like(points) * noise_scale
    alpha = alpha_start + (alpha_end - alpha_start) * t
    print(alpha)
    return (1 - alpha) * points + (alpha) * noise


pts_t = []

errs = []

scene_volume = []

for i in range(100):
    pts_t.append(noised_points(data["pc_action"], i / 100))
    errs.append(((pts_t[i] - data["pc_action"]) ** 2).sum(dim=-1).sqrt().mean())

    # Get the volume of the scene, by computing the bounding box.
    scene_volume.append(
        (pts_t[i].max(dim=0).values - pts_t[i].min(dim=0).values).prod()
    )


# %%
scene_volume


# %%
import plotly.graph_objects as go

# Create a figure
fig = go.Figure()

# Add traces for each frame
for i, pts in enumerate(pts_t[:1]):
    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=2),
            name=f"Frame {i}",
        )
    )

# Create frames for the animation
frames = [
    go.Frame(
        data=[
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=2),
            )
        ]
    )
    for pts in pts_t
]

# Add frames to the figure
fig.frames = frames

# Add animation settings
fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True,
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]
)

# Set the layout for the 3D scatter plot
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-10, 10], autorange=False),
        yaxis=dict(range=[-10, 10], autorange=False),
        zaxis=dict(range=[-10, 10], autorange=False),
    ),
    title="Point Cloud Animation",
    width=800,
    height=800,
)

# Show the figure
fig.show()

# %%
import matplotlib.pyplot as plt

plt.plot(errs)

# %%
scene_volume

# %%
# Plot the scene volume
plt.plot(scene_volume)

# %%
pcd = data["pc_action"]

# Scale data to [-1, 1] in every dimension
pcd = (pcd - pcd.min()) / (pcd.max() - pcd.min()) * 2 - 1

# Do the same noising
pts_t = []
errs = []
scene_volume = []

for i in range(100):
    pts_t.append(noised_points(pcd, i / 100))
    errs.append(((pts_t[i] - pcd) ** 2).sum(dim=-1).sqrt().mean())

    # Get the volume of the scene, by computing the bounding box.
    scene_volume.append(
        (pts_t[i].max(dim=0).values - pts_t[i].min(dim=0).values).prod()
    )

plt.plot(errs)

# %%
import plotly.graph_objects as go

# Create a figure
fig = go.Figure()

# Add traces for each frame
for i, pts in enumerate(pts_t[:1]):
    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=2),
            name=f"Frame {i}",
        )
    )

# Create frames for the animation
frames = [
    go.Frame(
        data=[
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=2),
            )
        ]
    )
    for pts in pts_t
]

# Add frames to the figure
fig.frames = frames

# Add animation settings
fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True,
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]
)

# Set the layout for the 3D scatter plot
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-1, 1], autorange=False),
        yaxis=dict(range=[-1, 1], autorange=False),
        zaxis=dict(range=[-1, 1], autorange=False),
    ),
    title="Point Cloud Animation",
    width=800,
    height=800,
)

# Show the figure
fig.show()

# %%
# Plot the scene volume
plt.plot(scene_volume)

# %%


# %%
(pts_t[-1].max(dim=0).values - pts_t[-1].min(dim=0).values).prod()

# %%
