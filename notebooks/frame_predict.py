from non_rigid.nets.pn2 import PN2Dense
from non_rigid.nets.dgcnn import DGCNN
from non_rigid.datasets.dedo import DedoDataset, DedoDataModule
from non_rigid.models.dit.models import DiT_PointCloud_Cross
import numpy as np
import torch
from omegaconf import OmegaConf
import json
import os
from pathlib import Path

import torch_geometric.data as tgd
import torch_geometric.loader as tgl

import rpad.visualize_3d.plots as vpl
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from tqdm import tqdm

from argparse import ArgumentParser

# ignore TypedStorage warnings
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)

# torch settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Since most of us are training on 3090s+, we can use mixed precision.
torch.set_float32_matmul_precision("medium")
torch.manual_seed(42)

# set dataset config and create datamodule
dataset_cfg = OmegaConf.load("../configs/dataset/dedo.yaml")

overrides = {
    "train_size": 400, #400,
    "scene": False,
    "world_frame": False,
    "scene_anchor": False,
    "rel_pose": True,
    "rel_pose_type": "translation",
    "center_type": "anchor_center",
    "predict_ref_frame": True,
    "action_context_center_type": "center",
    "cloth_geometry": "multi",
    "cloth_pose": "random",
    # "scene_transform_type": "random_flat_upright",
    "sample_size_anchor": 512,
    "sample_size_action": 512,
}

dataset_cfg = OmegaConf.merge(dataset_cfg, overrides)
if dataset_cfg.sample_size_action != dataset_cfg.sample_size_anchor:
    raise ValueError("Sample sizes must be equal for now.")

# print(
#     json.dumps(
#         OmegaConf.to_container(dataset_cfg, resolve=True, throw_on_missing=False),
#         sort_keys=True,
#         indent=4,
#     )
# )

datamodule = DedoDataModule(
    batch_size=16, # 16,
    val_batch_size=4,
    num_workers=1,
    dataset_cfg=dataset_cfg,
)
datamodule.setup("fit")



class PygDataset(tgd.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def get(self, index):
        """
        Mini-wrapper for DedoDataset to return torch_geometric.data.Data. 
        "pos" is just the anchor point cloud, and "y" is the mean of the goal action point cloud in the anchor frame.
        """
        item = self.dataset[index]
        num_anchor_points = item["pc_anchor"].shape[0]

        # adding small random noise to anchor point cloud
        # item["pc_anchor"] += 5e-1 * torch.randn_like(item["pc_anchor"])

        data = tgd.Data(
            x=item["pc_anchor"],
            pos=item["pc_anchor"],
            action=item["pc_action"],
            rel_pose=item["rel_pose"],
            y=item["pc"].mean(dim=0, keepdim=True).repeat(num_anchor_points, 1),

        )
        return data
    
    def __getitem__(self, index):
        return self.get(index)


train_dataset = PygDataset(datamodule.train_dataset)
val_dataset = PygDataset(datamodule.val_dataset)

train_batch_size = 16
val_batch_size = 4

train_loader = tgl.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16)
val_loader = tgl.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=16)

# creating model

class FramePredictorSimple(torch.nn.Module):
    def __init__(self, out_channels):
        super(FramePredictorSimple, self).__init__()
        self.out_channels = out_channels
        self.in_channels = 3
        self.pn2 = PN2Dense(in_channels=3, out_channels=out_channels)

    def forward(self, batch):
        # make sure all of the point clouds in the batch have the same number of points
        ptr_diffs = torch.unique(batch.ptr[1:] - batch.ptr[:-1])
        if len(ptr_diffs) > 1:
            raise ValueError("All point clouds in the batch must have the same number of points.")
        else:
            num_points = ptr_diffs.item()

        output = self.pn2(batch)
        # reshape output to get (B, N, C) shape
        output = output.reshape(-1, num_points, self.out_channels)

        logits = output[..., [0]]
        residuals = output[..., 1:4]
        vars = output[..., 4:]

        # run vars through softplus to ensure positive values
        vars = torch.nn.functional.softplus(vars)

        # add mean residuals to points to get mean predictions
        means = residuals + batch.pos.reshape(-1, num_points, 3)

        # converting logits to probabilities
        probs = torch.softmax(logits, dim=1)

        return {
            "probs": probs,
            "means": means,
            "vars": vars,
        }
    
class FramePredictorDGCNN(torch.nn.Module):
    def __init__(self, out_channels):
        super(FramePredictorDGCNN, self).__init__()
        self.out_channels = out_channels
        self.dgcnn = DGCNN(emb_dims=512)
        self.final = torch.nn.Conv1d(
            in_channels=512,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, batch):
        # make sure all of the point clouds in the batch have the same number of points
        ptr_diffs = torch.unique(batch.ptr[1:] - batch.ptr[:-1])
        if len(ptr_diffs) > 1:
            raise ValueError("All point clouds in the batch must have the same number of points.")
        else:
            num_points = ptr_diffs.item()
        
        input = batch.pos.reshape(-1, num_points, 3).permute(0, 2, 1)
        output = self.dgcnn(input)
        output = self.final(output)
        # reshape output to get (B, N, C) shape
        output = output.permute(0, 2, 1)

        logits = output[..., [0]]
        residuals = output[..., 1:4]
        vars = output[..., 4:]

        # run vars through softplus to ensure positive values
        vars = torch.nn.functional.softplus(vars)

        # add mean residuals to points to get mean predictions
        means = residuals + batch.pos.reshape(-1, num_points, 3)

        # converting logits to probabilities
        probs = torch.softmax(logits, dim=1)

        return {
            "probs": probs,
            "means": means,
            "vars": vars,
        }

class FramePredictorMLPTransformer(torch.nn.Module):
    def __init__(self):
        super(FramePredictorMLPTransformer, self).__init__()
        model_cfg = {
            "rotary": False,
            "x_encoder": "mlp",
            "x0_encoder": None,
            "y_encoder": "mlp",
            "extra_features": False,
            "rel_pose": True,
            "rel_pose_type": "translation",
            "center_noise": False,
        }
        # omegaconf from dict
        model_cfg = OmegaConf.create(model_cfg)
        self.model = DiT_PointCloud_Cross(
            depth=5, hidden_size=128, num_heads=4, in_channels=3, learn_sigma=True, model_cfg=model_cfg
        )
    
    def forward(self, batch):
        # make sure all of the point clouds in the batch have the same number of points
        ptr_diffs = torch.unique(batch.ptr[1:] - batch.ptr[:-1])
        if len(ptr_diffs) > 1:
            raise ValueError("All point clouds in the batch must have the same number of points.")
        else:
            num_points = ptr_diffs.item()
        
        # assume action and anchor are the same size
        pc_anchor = batch.pos.reshape(-1, num_points, 3).permute(0, 2, 1)
        pc_action = batch.action.reshape(-1, num_points, 3).permute(0, 2, 1)
        bs = pc_anchor.shape[0]
        rel_pose = batch.rel_pose.reshape(bs, -1)
        model_kwargs = {
            "x": pc_anchor, # anchor
            "y": pc_action, # action
            "t": torch.zeros(bs).to(pc_anchor.device), # zeros
            "rel_pose": rel_pose,# relpose
        }
        output = self.model(**model_kwargs).permute(0, 2, 1)

        logits = output[..., [0]]
        residuals = output[..., 1:4]
        vars = output[..., 4:5] # ignore last output

        # run vars through softplus to ensure positive values
        vars = torch.nn.functional.softplus(vars)

        # add mean residuals to points to get mean predictions
        means = residuals + batch.pos.reshape(-1, num_points, 3)

        # converting logits to probabilities
        probs = torch.softmax(logits, dim=1)

        return {
            "probs": probs,
            "means": means,
            "vars": vars,
        }

# defining GMM loss function
class GMMLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        """
        eps: value used to clamp var, for stability.
        """
        super(GMMLoss, self).__init__()
        self.eps = eps
    
    def forward(self, batch, pred, vars=0.00001, uniform_loss=0.0):
        """
        batch: torch_geometric.data.Batch object. batch.y is (B x N, 3) tensor of target means.
        pred: dict with keys "probs", "means", "vars", which are shape (B, N, 1 or 3)
        """
        # make sure all of the point clouds in the batch have the same number of points
        ptr_diffs = torch.unique(batch.ptr[1:] - batch.ptr[:-1])
        if len(ptr_diffs) > 1:
            raise ValueError("All point clouds in the batch must have the same number of points.")
        else:
            num_points = ptr_diffs.item()

        targets = batch.y.reshape(-1, num_points, 3)
        probs = pred["probs"]
        means = pred["means"]
        # vars = pred["vars"]

        # clamp vars for stability
        # vars = torch.clamp(vars, min=self.eps)
        # vars = vars

        diff = targets - means
        point_likelihood_exps = -0.5 * torch.sum((diff ** 2) / vars, dim=-1, keepdim=True)
        maxlog = point_likelihood_exps.max(dim=-2, keepdim=True).values
        point_likelihoods = torch.exp(point_likelihood_exps - maxlog)
        likelihoods = torch.sum(probs * point_likelihoods, dim=-2, keepdim=True)
        log_likelihoods = torch.log(likelihoods) + maxlog

        loss = -torch.sum(log_likelihoods)

        # uniform loss
        if uniform_loss > 0.0:
            uniform_nll = -torch.sum(
                torch.log(torch.sum(point_likelihoods, dim=-2, keepdim=True)) + maxlog
            )
            loss += uniform_loss * uniform_nll
        return loss
    
# defining visualization code
def viz_model(model, device):
    model.eval()
    model.to(device)
    fig = make_subplots(rows=2, cols=4,
                        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}], 
                            [{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
                        subplot_titles=["Plot 1", "Plot2", "Plot3", "Plot4", "Plot5", "Plot6", "Plot7", "Plot8"],
    )
    
    # visualize training dataset
    for i in range(2):
        batch = tgd.Batch.from_data_list([train_dataset[i]]).to(device)
        with torch.no_grad():
            pred = model(batch)
        probs = pred["probs"][0].cpu().numpy().reshape((-1,))
        target = batch.y.cpu().numpy()[0]
        pc_anchor = batch.pos.cpu().numpy()
        means = pred["means"][0].cpu().numpy()

        # prob statistics
        prob_min, prob_max, prob_med = probs.min(), probs.max(), np.median(probs)
        sorted_indices = np.argsort(probs)[::-1]

        # left plot
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=2, color="blue"),
                x=pc_anchor[:, 0],
                y=pc_anchor[:, 1],
                z=pc_anchor[:, 2],
            ), row=i+1, col=1
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=4, color=probs, colorscale="Viridis", cmin=0),
                x=means[:, 0],
                y=means[:, 1],
                z=means[:, 2],
            ), row=i+1, col=1
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=8, color="red"),
                x=[target[0]],
                y=[target[1]],
                z=[target[2]],
            ), row=i + 1, col=1
        )

        # middle plot (top 99)
        top_99_indices = sorted_indices[np.cumsum(probs[sorted_indices]) < 0.99]
        means99 = means[top_99_indices]
        probs99 = probs[top_99_indices]
        num_probs99 = len(probs99)
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=2, color="blue"),
                x=pc_anchor[:, 0],
                y=pc_anchor[:, 1],
                z=pc_anchor[:, 2],
            ), row=i+1, col=2
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=4, color=probs99, colorscale="Viridis", cmin=0),
                x=means99[:, 0],
                y=means99[:, 1],
                z=means99[:, 2],
            ), row=i+1, col=2
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=8, color="red"),
                x=[target[0]],
                y=[target[1]],
                z=[target[2]],
            ), row=i + 1, col=2
        )

        # right plot
        top_90_indices = sorted_indices[np.cumsum(probs[sorted_indices]) < 0.90]
        means90 = means[top_90_indices]
        probs90 = probs[top_90_indices]
        num_probs90 = len(probs90)
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=2, color="blue"),
                x=pc_anchor[:, 0],
                y=pc_anchor[:, 1],
                z=pc_anchor[:, 2],
            ), row=i+1, col=3
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=4, color=probs90, colorscale="Viridis", cmin=0),
                x=means90[:, 0],
                y=means90[:, 1],
                z=means90[:, 2],
            ), row=i+1, col=3
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=8, color="red"),
                x=[target[0]],
                y=[target[1]],
                z=[target[2]],
            ), row=i + 1, col=3
        )

        # top50 plot
        top_50_indices = sorted_indices[np.cumsum(probs[sorted_indices]) < 0.50]
        means50 = means[top_50_indices]
        probs50 = probs[top_50_indices]
        num_probs50 = len(probs50)
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=2, color="blue"),
                x=pc_anchor[:, 0],
                y=pc_anchor[:, 1],
                z=pc_anchor[:, 2],
            ), row=i+1, col=4
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=4, color=probs50, colorscale="Viridis", cmin=0),
                x=means50[:, 0],
                y=means50[:, 1],
                z=means50[:, 2],
            ), row=i+1, col=4
        )
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker=dict(size=8, color="red"),
                x=[target[0]],
                y=[target[1]],
                z=[target[2]],
            ), row=i + 1, col=4
        )

        fig.layout.annotations[4 * i].update(text=f"Median: {prob_med:.6f} Min: {prob_min:.6f}, Max: {prob_max:.6f}")
        fig.layout.annotations[4 * i + 1].update(text=f"Top-0.99 Num Points: {num_probs99}")
        fig.layout.annotations[4 * i + 2].update(text=f"Top-0.90 Num Points: {num_probs90}")
        fig.layout.annotations[4 * i + 3].update(text=f"Top-0.50 Num Points: {num_probs50}")
    
    return fig, num_probs99, num_probs90, num_probs50



# TODO: PARSE ARGS FOR MODEL SELECTION
argparser = ArgumentParser()
argparser.add_argument("--model", type=str, default="simple")
argparser.add_argument("--epochs", type=int, default=5000)
argparser.add_argument("--var", type=float, default=0.001)
argparser.add_argument("--mode", type=str, default="train")
argparser.add_argument("--gpu", type=int, default=0)
argparser.add_argument("--multiscale_var", action="store_true")
argparser.add_argument("--uniform_loss", type=float, default=0.0)
args = argparser.parse_args()


# create model
if args.model == "simple":
    model = FramePredictorSimple(5)
elif args.model == "dgcnn":
    model = FramePredictorDGCNN(5)
elif args.model == "mlp_transformer":
    model = FramePredictorMLPTransformer()

if args.multiscale_var:
    exp_name = f"{args.model}_epochs={args.epochs}_multiscale_var"
else:
    exp_name = f"{args.model}_epochs={args.epochs}_var={args.var}"

if args.uniform_loss > 0.0:
    exp_name += f"_uniform_loss={args.uniform_loss}"
device = f"cuda:{args.gpu}"

if args.mode == "train":
    print(f"Training: {exp_name}")
    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # 5e-3 originally

    # initialize loss
    loss_fn = GMMLoss()

    # training params
    num_epochs = args.epochs
    val_every = args.epochs // 10

    model.to(device)

    # training statistics
    total_losses = []
    total_val_losses = []
    total_probs99 = []
    total_probs90 = []
    total_probs50 = []

    os.makedirs(exp_name, exist_ok=True)
    os.makedirs(f"{exp_name}/logs", exist_ok=True)
    os.makedirs(f"{exp_name}/checkpoints", exist_ok=True)


    min_val_loss = float("inf")


    fig, num_probs99, num_probs90, num_probs50 = viz_model(model, device)
    fig.update_layout(title_text="Epoch 0")
    # fig.show(renderer="browser")
    fig.write_html(f"{exp_name}/logs/epoch_0.html")
    total_probs99.append(num_probs99)
    total_probs90.append(num_probs90)
    total_probs50.append(num_probs50)


    # basic training loop
    for epoch in range(num_epochs):
        # train step
        model.train()
        epoch_loss = []
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)

            if args.multiscale_var:
                loss = sum(
                    [loss_fn(batch, pred, vars=var, uniform_loss=args.uniform_loss) 
                     for var in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]]
                )
            else:
                loss = loss_fn(batch, pred, vars=args.var, uniform_loss=args.uniform_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss.append(loss.item())
        epoch_loss = np.mean(epoch_loss)
        total_losses.append(epoch_loss)

        # val step
        if (epoch + 1) % val_every == 0:
            model.eval()
            val_loss = []
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    batch = batch.to(device)
                    pred = model(batch)

                    if args.multiscale_var:
                        loss = sum(
                            [loss_fn(batch, pred, vars=var, uniform_loss=args.uniform_loss) 
                             for var in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]]
                        )
                    else:
                        loss = loss_fn(batch, pred, vars=args.var, uniform_loss=args.uniform_loss)
                    val_loss.append(loss.item())
            val_loss = np.mean(val_loss)
            total_val_losses.append(val_loss)

            # save model checkpoint
            torch.save(model.state_dict(), f"{exp_name}/checkpoints/model_{epoch + 1}.pt")

        # logging.permute(1, 0).reshape(6, -1).permute(1, 0)
        if (epoch + 1) % val_every == 0:
            print(f"Epoch {epoch}, Train Loss: {epoch_loss}, Val Loss: {val_loss}")        
            val_fig, num_probs99, num_probs90, num_probs50 = viz_model(model, device)

            # visualize predictions
            val_fig.update_layout(title_text=f"Epoch {epoch + 1}, Train Loss: {epoch_loss} Val Loss: {val_loss}")
            # val_fig.show(renderer="browser")
            val_fig.write_html(f"{exp_name}/logs/epoch_{epoch + 1}.html")
            total_probs99.append(num_probs99)
            total_probs90.append(num_probs90)
            total_probs50.append(num_probs50)
        else:
            print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss}")


    # After training, plot the losses
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, len(total_losses) + 1), y=total_losses, name="Train Loss"))
    fig.add_trace(go.Scatter(x=np.arange(val_every, (len(total_val_losses) + 1) * val_every, val_every), y=total_val_losses, name="Val Loss"))
    fig.update_layout(title="Losses", xaxis_title="Epoch", yaxis_title="Loss")
    fig.show()
    fig.write_html(f"{exp_name}/logs/losses.html")

    # Also plot the number of points in the top 0.99 and 0.90 probabilities
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(val_every, (len(total_val_losses) + 1) * val_every, val_every), y=total_probs99, name="Top-0.99"))
    fig.add_trace(go.Scatter(x=np.arange(val_every, (len(total_val_losses) + 1) * val_every, val_every), y=total_probs90, name="Top-0.90"))
    fig.add_trace(go.Scatter(x=np.arange(val_every, (len(total_val_losses) + 1) * val_every, val_every), y=total_probs50, name="Top-0.50"))
    fig.update_layout(title="Top-K Probabilities", xaxis_title="Epoch", yaxis_title="Number of Points")
    fig.show()
    fig.write_html(f"{exp_name}/logs/top_probs.html")
elif args.mode == "eval":
    print(f"Evaluating: {exp_name}")
    os.makedirs(f"{exp_name}/viz", exist_ok=True)
    os.makedirs(f"{exp_name}/viz/train", exist_ok=True)
    os.makedirs(f"{exp_name}/viz/val", exist_ok=True)

    eval_model_path = f"{exp_name}/checkpoints/model_{args.epochs}.pt"


    model.load_state_dict(torch.load(eval_model_path))

    eval_viz_path = f"{exp_name}/viz/"

    model.to(device)
    num_samples = 50

    if dataset_cfg.hole == "single":
        bs = 1
    elif dataset_cfg.hole == "double":
        bs = 2
    else:
        raise ValueError("Invalid hole type.")
    bs *= dataset_cfg.num_anchors


    @torch.no_grad()
    def eval_dataset(dataset, model, split):
        rmses = []
        coverage_rmses = []
        precision_rmses = []
        num_batches = len(dataset) // bs

        for i in tqdm(range(num_batches)):
            gt_means = []
            sampled_means = []

            for j in range(bs):
                data_item = dataset[i * bs + j]
                gt_means.append(data_item.y[[0], :])

                pred = model(tgd.Batch.from_data_list([data_item]).to(device))
                probs, means = pred["probs"], pred["means"]
                idxs = torch.multinomial(probs.squeeze(-1), num_samples // bs, replacement=True).squeeze()
                sampled_means.append(means[:, idxs].squeeze())


            gt_means = torch.cat(gt_means, dim=0).to(device)
            sampled_means = torch.cat(sampled_means, dim=0)

            # plot point clouds
            fig = go.Figure()
            anchor_pc = data_item.pos.cpu().numpy()
            gt_means_pc = gt_means.cpu().numpy()
            sampled_means_pc = sampled_means.cpu().numpy()
            fig.add_trace(
                go.Scatter3d(
                    mode="markers",
                    marker=dict(size=2, color="blue"),
                    x=anchor_pc[:, 0],
                    y=anchor_pc[:, 1],
                    z=anchor_pc[:, 2],
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    mode="markers",
                    marker=dict(size=4, color="red"),
                    x=gt_means_pc[:, 0],
                    y=gt_means_pc[:, 1],
                    z=gt_means_pc[:, 2],
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    mode="markers",
                    marker=dict(size=4, color="green"),
                    x=sampled_means_pc[:, 0],
                    y=sampled_means_pc[:, 1],
                    z=sampled_means_pc[:, 2],
                )
            )
            fig.write_html(f"{eval_viz_path}/{split}/{i}.html")

            # compute pairwise distances
            dists = torch.cdist(gt_means, sampled_means, p=2)

            # compute rmse, coverage rmse, and precision rmse
            # rmses.append(torch.mean(dists).item())
            # coverage_rmses.append(torch.mean(torch.min(dists, dim=1).values).item())
            # precision_rmses.append(torch.mean(torch.min(dists, dim=0).values).item())
            rmses.append(dists.flatten().cpu().numpy())
            coverage_rmses.append(torch.min(dists, dim=1).values.cpu().numpy())
            precision_rmses.append(torch.min(dists, dim=0).values.cpu().numpy())
        
        rmses = np.concatenate(rmses)
        coverage_rmses = np.concatenate(coverage_rmses)
        precision_rmses = np.concatenate(precision_rmses)
        
        hist = go.Figure()
        hist.add_trace(
            go.Histogram(
                x=rmses,
                name="RMSE",
            )
        )
        hist.add_trace(
            go.Histogram(
                x=precision_rmses,
                name="Precision RMSE",
            )
        )
        hist.add_trace(
            go.Histogram(
                x=coverage_rmses,
                name="Coverage RMSE",
            )
        )
        hist.update_layout(barmode="overlay")
        hist.update_traces(opacity=0.75)
        hist.update_layout(title_text=f"{split} Histograms")
        hist.write_html(f"{eval_viz_path}/{split}/hist.html")

        rmses = np.mean(rmses)
        coverage_rmses = np.mean(coverage_rmses)
        precision_rmses = np.mean(precision_rmses)
        print(f"RMSE: {rmses}, Coverage RMSE: {coverage_rmses}, Precision RMSE: {precision_rmses}")

    
    # setting dataset size to num. demos
    train_dataset.dataset.size = train_dataset.dataset.num_demos
    val_dataset.dataset.size = val_dataset.dataset.num_demos
    eval_dataset(train_dataset, model, "train")
    eval_dataset(val_dataset, model, "val")
else:
    raise ValueError("Invalid mode.")