import hydra
import lightning as L
import omegaconf
import torch
import torch.utils._pytree as pytree
import wandb

import numpy as np
import os
from pathlib import Path

import multiprocessing as mp
from multiprocessing import Process, Queue

from non_rigid.datasets.articulated import ArticulatedDemoDataModule
from non_rigid.datasets.articulated_pc import ArticulatedPCDataModule
from non_rigid.models.mlp_policy import MLPPolicy
from non_rigid.models.pc_policy import PCPolicy
from non_rigid.utils.script_utils import (
    PROJECT_ROOT,
    create_model,
    flatten_outputs,
    match_fn,
)

from rpad.pybullet_envs.pm_suction import PMSuctionDemoEnv
from scipy.spatial.transform import Rotation as R

from tqdm import tqdm
import time

import pybullet as p
import pkgutil

MAX_STEPS = 1500
NUM_WORKERS = 32
NUM_TRAJECTORIES = 100
ACTION_REPEAT = 1
USE_EGL = True
GUI = False

TEST_1 = False
TEST_2 = False
TEST_3 = False
TEST_4 = True


def eval_policy(process_num, cfg, mean, std, success_queues, dist_queues):
    # process specific seed initialization
    L.seed_everything(42 + process_num)

    ##### LOADING NETWORK #####
    # load network based on config
    if cfg.model.name == "mlp":
        network = MLPPolicy(
            input_dim=cfg.model.input_dim,
            hidden_dims=cfg.model.hidden_dims,
        )
    elif cfg.model.name == "pn2":
        network = PCPolicy(
            embedding_dim=cfg.model.embedding_dim,
            lin1_dim=cfg.model.lin1_dim,
            lin2_dim=cfg.model.lin2_dim,
            in_dim=cfg.model.in_dim,
        )

    # get checkpoint file (for now, this does not log a run)
    checkpoint_reference = cfg.checkpoint.reference
    if checkpoint_reference.startswith(cfg.wandb.entity):
        api = wandb.Api()
        artifact_dir = cfg.wandb.artifact_dir
        artifact = api.artifact(checkpoint_reference, type="model")
        ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    else:
        ckpt_file = checkpoint_reference
    # load checkpoint
    ckpt = torch.load(ckpt_file)
    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    )
    # set model to eval mode
    network.eval()
    # move network to gpu for evaluation
    if torch.cuda.is_available():
        network.cuda() 

    ##### LOADING ENVIRONMENT #####
    PM_DIR = Path(os.path.expanduser("~/datasets/partnet-mobility/dataset"))
    # env = PMSuctionDemoEnv(str(cfg.dataset.obj_id), PM_DIR, True if process_num == 0 else False, use_egl=USE_EGL)
    env = PMSuctionDemoEnv(str(cfg.dataset.obj_id), PM_DIR, GUI if process_num == 0 else False, use_egl=USE_EGL)

    ##### RUNNING EVALUATION EXPERIMENTS #####
    # --- Test Environment 1: Same initial condition ---
    if TEST_1:
        if process_num == 0:
            print("Running test environment 1")
            pbar = tqdm(total=NUM_TRAJECTORIES)
        while True:
            # updating progress bar
            if process_num == 0:
                pbar.update(success_queues[0].qsize() - pbar.n)
            if success_queues[0].qsize() < NUM_TRAJECTORIES:
                # resetting environment
                pos_init = np.array([-1.0, 0.6, 0.8])
                ori_init = R.from_euler('xyz', [0, -np.pi / 2, 0]).as_quat()
                obs_init = env.reset(pos_init, ori_init)
                _, success, normalized_dist = generate_trajectory(network, env, obs_init, mean, std)
                # updating queues
                success_queues[0].put(success)
                dist_queues[0].put(normalized_dist)
            else:
                if process_num == 0:
                    pbar.close()
                break

    # --- Test Environment 2: Random gripper position ---
    if TEST_2:
        if process_num == 0:
            print("Running test environment 2")
            pbar = tqdm(total=NUM_TRAJECTORIES)
        while True:
            # updating progress bar
            if process_num == 0:
                pbar.update(success_queues[1].qsize() - pbar.n)
            if success_queues[1].qsize() < NUM_TRAJECTORIES:
                # resetting environment
                pos_init = np.array([-1.0, 0.6, 0.8]) + (np.random.rand(3) - 0.5) * 0.3
                ori_init = R.from_euler('xyz', [0, -np.pi / 2, 0]).as_quat()
                obs_init = env.reset(pos_init, ori_init)
                _, success, normalized_dist = generate_trajectory(network, env, obs_init, mean, std)
                # updating queues
                success_queues[1].put(success)
                dist_queues[1].put(normalized_dist)
            else:
                if process_num == 0:
                    pbar.close()
                break

    # --- Test Environment 3: Random gripper position and orientation ---
    if TEST_3:
        if process_num == 0:
            print("Running test environment 3")
            pbar = tqdm(total=NUM_TRAJECTORIES)
        while True:
            # updating progress bar
            if process_num == 0:
                pbar.update(success_queues[2].qsize() - pbar.n)
            if success_queues[2].qsize() < NUM_TRAJECTORIES:
                # resetting environment
                pos_init = np.array([-1.0, 0.6, 0.8]) + (np.random.rand(3) - 0.5) * 0.3
                # ori_init = R.from_euler('xyz', [0, -np.pi / 2, 0]).as_quat()
                ori_init = R.random().as_quat()
                # print('Position:', pos_init, 'Orientation:', R.from_quat(ori_init).as_euler('xyz', degrees=True))
                obs_init = env.reset(pos_init, ori_init)
                actions, success, normalized_dist = generate_trajectory(network, env, obs_init, mean, std)
                # save initial position and orientation to file
                # if process_num == 0:
                #     np.save(f"/home/eycai/Documents/non-rigid/scripts/viz/mlp/initial_pos_ori_{success_queues[2].qsize()}.npy", np.concatenate((pos_init, ori_init)))
                #     np.save(f"/home/eycai/Documents/non-rigid/scripts/viz/mlp/actions_{success_queues[2].qsize()}.npy", np.array(actions))
                # updating queues
                success_queues[2].put(success)
                dist_queues[2].put(normalized_dist)
            else:
                if process_num == 0:
                    pbar.close()
                break

    # --- Test Environment 4: Same initial gripper, random microwave position ---
    if TEST_4:
        if process_num == 0:
            print("Running test environment 4")
            pbar = tqdm(total=NUM_TRAJECTORIES)
        while True:
            # updating progress bar
            if process_num == 0:
                pbar.update(success_queues[3].qsize() - pbar.n)
            if success_queues[3].qsize() < NUM_TRAJECTORIES:
                # resetting environment
                pos_init = np.array([-1.0, 0.6, 0.8])
                ori_init = R.from_euler('xyz', [0, -np.pi / 2, 0]).as_quat()
                # sample microwave position
                pos_obj = np.array([0.3, 0, 0.513]) + (np.random.rand(3) - 0.5) * 0.3
                pos_ori = np.array([0, 0, 0, 1.0])
                obs_init = env.reset(pos_init, ori_init, pos_obj, pos_ori)
                _, success, normalized_dist = generate_trajectory(network, env, obs_init, mean, std)
                success_queues[3].put(success)
                dist_queues[3].put(normalized_dist)
            else:
                if process_num == 0:
                    pbar.close()
                break



@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
def main(cfg):
    if GUI and USE_EGL:
        quit("Cannot use GUI and EGL at the same time. Please choose one.")
    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")

    # Global seed for reproducibility.
    # L.seed_everything(42)

    # temporary - making sure the network works
    ######################################################################
    # Create the datamodule.
    # Should be the same one as in training, but we're gonna use val+test
    # dataloaders.
    ######################################################################

    # TODO: load datamodule based on config file


    # checking dataset name
    if cfg.dataset.name == "articulated10":
        datamodule = ArticulatedDemoDataModule(
            root=cfg.dataset.data_dir,
            batch_size=cfg.inference.batch_size,
            num_workers=cfg.resources.num_workers,
            filename=cfg.dataset.filename,
        )
    elif cfg.dataset.name == "articulated_pc":
        datamodule = ArticulatedPCDataModule(
            root=cfg.dataset.data_dir,
            batch_size=cfg.inference.batch_size,
            num_workers=cfg.resources.num_workers,
        )
    


    # datamodule = ArticulatedDemoDataModule(
    #     root=cfg.dataset.data_dir,
    #     batch_size=cfg.inference.batch_size,
    #     num_workers=cfg.resources.num_workers,
    #     filename=cfg.dataset.filename,
    # )
    # GET NORMALIZATION STATS FROM DATAMODULE
    datamodule.setup("predict")
    mean = datamodule.mean
    std = datamodule.std

    # create queues for multiprocessing
    success_queues = []
    dist_queues = []
    for i in range(4):
        success_queues.append(Queue())
        dist_queues.append(Queue())
    
    # create processes
    procs = []
    for i in range(NUM_WORKERS):
        proc = Process(target=eval_policy, args=(i, cfg, mean, std, success_queues, dist_queues))
        procs.append(proc)
        proc.start()

    # wait for processes to finish
    for proc in procs:
        proc.join()

    # convert each queue to a list
    success_lists = []
    dist_lists = []
    for i in range(4):
        success_lists.append([])
        dist_lists.append([])
        while not success_queues[i].empty():
            success_lists[i].append(success_queues[i].get())
            dist_lists[i].append(dist_queues[i].get())
    # evaluation metrics from last NUM_TRAJECTORIES
    for i in range(4):
        print("Test Environment ", i + 1, ":")
        # convert to numpy array
        success_lists[i] = np.array(success_lists[i])
        dist_lists[i] = np.array(dist_lists[i])
        # print shape of each list
        print(success_lists[i].shape, dist_lists[i].shape)
        # get last NUM_TRAJECTORIES from each list
        successes = success_lists[i][-NUM_TRAJECTORIES:]
        dists = dist_lists[i][-NUM_TRAJECTORIES:]
        print("Success Rate: ", np.sum(successes), "/", len(successes))
        print("Mean Norm. Dist.: ", np.mean(dists))
    return




def pred_to_action(pred, mean, std):
    # print(pred)
    action = pred.cpu()
    action[:, 2:] = action[:, 2:] * std + mean
    action[:, 0:2] = torch.round(action[:, 0:2])
    return np.squeeze(np.array(action))
    

def create_signal_plot():
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 3)

    # Label the plots:
    # Activate, Move/Pull, X, Y, Z, Roll, Pitch, Yaw
    axs[0, 0].set_title("Activate")
    axs[0, 1].set_title("Move/Pull")
    axs[0, 2].set_title("X")
    axs[1, 0].set_title("Y")
    axs[1, 1].set_title("Z")
    axs[1, 2].set_title("Roll")
    axs[2, 0].set_title("Pitch")
    axs[2, 1].set_title("Yaw")
    return fig, axs

def update_signal_plot(fig, axs, np_actions):
    axs[0, 0].plot(np_actions[:, 0])
    axs[0, 1].plot(np_actions[:, 1])
    axs[0, 2].plot(np_actions[:, 2])
    axs[1, 0].plot(np_actions[:, 3])
    axs[1, 1].plot(np_actions[:, 4])
    axs[1, 2].plot(np_actions[:, 5])
    axs[2, 0].plot(np_actions[:, 6])
    axs[2, 1].plot(np_actions[:, 7])

    fig.canvas.draw()
    fig.canvas.flush_events()

def generate_trajectory(network, env, obs, mean, std):
    actions = []
    success, _ = env.goal_check()
    normalized_dist = None
    steps = 0
    # env should have function to check for success
    while not success and steps < MAX_STEPS:
        # action = network(torch.tensor(obs).unsqueeze(0))
        action = network.predict_obs(obs)
        # convert to action
        action = pred_to_action(action, mean, std)
        actions.append(action)
        obs = env.step(action)
        success, normalized_dist = env.goal_check()
        steps += 1
    # should also return normalized distance
    return actions, success, normalized_dist


if __name__ == "__main__": 
    main()
