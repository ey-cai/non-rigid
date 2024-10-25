import torch
import numpy as np
from pytorch3d.transforms import Transform3d, Translate

from non_rigid.models.df_base import DiffusionFlowBase, FlowPredictionInferenceModule, PointPredictionInferenceModule
from non_rigid.utils.script_utils import create_model

from diffusion_policy_3d.policy.base_policy import BasePolicy

import rpad.visualize_3d.plots as vpl


class TAX3D(BasePolicy):
    """
    This is simple TAX3D wrapper exclusively for policy rollouts.
    """
    def __init__(
            self,
            ckpt_file,
            device,
            eval_cfg,
            run_cfg,
    ):
        super().__init__()
        self.run_cfg = run_cfg
        self.eval_cfg = eval_cfg

        # switch mode to eval
        self.run_cfg.mode = "eval"
        self.run_cfg.inference = self.eval_cfg.inference

        network, model = create_model(self.run_cfg)
        self.network = network
        self.model = model

        # load network weights from checkpoint
        checkpoint = torch.load(ckpt_file, map_location=device)
        self.network.load_state_dict(
            {k.partition(".")[2]: v for k, v, in checkpoint["state_dict"].items()}
        )
        self.network.eval()
        self.model.eval()
        self.model.to(device)
        self.to(device)

        # Initializing current goal position. This is set during policy reset.
        self.goal_position = None
        self.results_world = None



    def reset(self):
        """
        Since this is open loop, this function will set the goal position to None.
        """
        self.goal_position = None

    def predict_action(self, obs_dict, deform_params):
        """
        Predict the action.
        """
        # if goal_position is unset (after policy reset), predict the goal position.
        if self.goal_position == None:
            pred_action, results_world = self.model.predict_obs(obs_dict, self.run_cfg)

            if self.eval_cfg.task.env_runner.task_name == "proccloth":
                # TODO: this is is missing segmentation logic for SD models
                goal1 = pred_action[:, deform_params['node_density'] - 1, :]# + torch.tensor([0, -0.5, 1.0], device=self.device)
                goal2 = pred_action[:, 0, :]# + torch.tensor([0, -0.5, 1.0], device=self.device)
            elif self.eval_cfg.task.env_runner.task_name == "hangbag":
                goal1 = pred_action[:, 209, :] + torch.tensor([0, -0.5, 1.0], device=self.device)
                goal2 = pred_action[:, 297, :] + torch.tensor([0, -0.5, 1.0], device=self.device)

                # adding hard-coded offset
                # flow1 = pred_flow[:, 209, :]
                # flow2 = pred_flow[:, 297, :]
                # flow1 = flow1 / torch.norm(flow1, dim=1, keepdim=True) * 1.18
                # flow2 = flow2 / torch.norm(flow2, dim=1, keepdim=True) * 1.18

                # goal1 = goal1 + flow1
                # goal2 = goal2 + flow2

            else:
                raise ValueError(f"Unknown task name: {self.eval_cfg.env_runner.task_name}")
            self.goal_position = torch.cat([goal1, goal2], dim=1).unsqueeze(0)
            # self.results_world = [res.squeeze().cpu().numpy() for res in pred_dict["results_world"]]
            self.results_world = [res.squeeze().cpu().numpy() for res in results_world]

        action_dict = {
            'action': self.goal_position,
        }
        return action_dict