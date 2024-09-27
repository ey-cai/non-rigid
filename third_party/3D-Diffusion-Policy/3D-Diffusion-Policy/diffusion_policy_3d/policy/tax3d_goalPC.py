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

            pred_action = model_predict(obs_dict)

            # pred_action = pred_dict["point"]["pred_world"]

            if self.eval_cfg.task.env_runner.task_name == "proccloth":
                # TODO: this is is missing segmentation logic for SD models
                goal1 = pred_action[:, deform_params['node_density'] - 1, :]# + torch.tensor([0, -0.5, 1.0], device=self.device)
                goal2 = pred_action[:, 0, :]# + torch.tensor([0, -0.5, 1.0], device=self.device)
            elif self.eval_cfg.task.env_runner.task_name == "hangbag":
                goal1 = pred_action[:, 209, :] + torch.tensor([0, -0.5, 1.0], device=self.device)
                goal2 = pred_action[:, 297, :] + torch.tensor([0, -0.5, 1.0], device=self.device)

            else:
                raise ValueError(f"Unknown task name: {self.eval_cfg.env_runner.task_name}")
            self.goal_position = torch.cat([goal1, goal2], dim=1).unsqueeze(0)
            # self.results_world = [res.squeeze().cpu().numpy() for res in pred_dict["results_world"]]
            self.results_world = [res.squeeze().cpu().numpy() for res in results_world]

        action_dict = {
            'action': self.goal_position,
        }
        return action_dict
    
    def model_predict(self, obs_dict):
        """
        TAX3D inference.
        """
        action_pc = obs_dict["pc_action"]
        anchor_pc = obs_dict["pc_anchor"]
        action_seg = obs_dict["seg"]
        anchor_seg = obs_dict["seg_anchor"]

        if self.run_cfg.dataset.scene:
            # scene-level processing
            scene_pc = torch.cat([action_pc, anchor_pc], dim=1)
            scene_seg = torch.cat([action_seg, anchor_seg], dim=1)

            # center the point cloud
            scene_center = scene_pc.mean(dim=1)
            scene_pc = scene_pc - scene_center
            T_goal2world = Translate(scene_center).get_matrix()

            item = {
                "pc_action": scene_pc,
                "seg": scene_seg,
                "T_goal2world": T_goal2world,
            }
        else:
            # object-centric processing
            if self.run_cfg.dataset.world_frame:
                action_center = torch.zeros(3, dtype=torch.float32, device=self.device).unsqueeze(0)
                anchor_center = torch.zeros(3, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                action_center = action_pc.mean(dim=1)
                anchor_center = anchor_pc.mean(dim=1)

            # center the point clouds
            action_pc = action_pc - action_center
            anchor_pc = anchor_pc - anchor_center
            T_action2world = Translate(action_center).get_matrix()
            T_goal2world = Translate(anchor_center).get_matrix()

            item = {
                "pc_action": action_pc,
                "pc_anchor": anchor_pc,
                "seg": action_seg,
                "seg_anchor": anchor_seg,
                "T_action2world": T_action2world,
                "T_goal2world": T_goal2world,
            }

        pred_dict = self.model.predict(item, self.eval_cfg.inference.num_trials, progress=False)
        pred_action = pred_dict["point"]["pred_world"] # point cloud i want
        
        return pred_action