import os
import time

from matplotlib import cm  # for colors
import numpy as np
import gym
import pybullet
import pybullet_utils.bullet_client as bclient

from ..utils.init_utils import (
    load_deform_object, load_rigid_object, apply_rigid_params, get_preset_properties
)
from ..utils.mesh_utils import get_mesh_data
from ..utils.procedural_utils import gen_procedural_hang_cloth
from ..utils.args import preset_override_util
from ..utils.bullet_manipulator import BulletManipulator, theta_to_sin_cos
from ..utils.task_info import ROBOT_INFO

from scipy.spatial.transform import Rotation as R

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import copy

# import "constants" from Tax3dEnv
from .tax3d_env import DEFORM_INFO, SCENE_INFO
from .tax3d_proccloth_env import Tax3dProcClothEnv

from mplib import Planner

class Tax3dProcClothRobotEnv(Tax3dProcClothEnv):
    """
    Tax3d + robot environment for HangProcCloth task.
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # Setting default task-specific parameters.
        self.scene_name = 'hangcloth'
        self.args.node_density = 25
        # self.args.num_holes = 1
    
    def load_objects(self, args):
        res = super().load_objects(args)
        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        self.sim.setAdditionalSearchPath(data_path)
        robot_info = ROBOT_INFO.get(f'franka{self.num_anchors:d}', None)
        assert(robot_info is not None) # make sure robot_info is ok
        robot_path = os.path.join(data_path, 'robots', robot_info['file_name'])

        self.robot = BulletManipulator(
            self.sim, robot_path, control_mode='velocity',
            ee_joint_name=robot_info['ee_joint_name'],
            ee_link_name=robot_info['ee_link_name'],
            base_pos=robot_info['base_pos'],
            base_quat=pybullet.getQuaternionFromEuler([0, 0, np.pi]),
            global_scaling=robot_info['global_scaling'],
            use_fixed_base=robot_info['use_fixed_base'],
            rest_arm_qpos=robot_info['rest_arm_qpos'],
            left_ee_joint_name=robot_info.get('left_ee_joint_name', None),
            left_ee_link_name=robot_info.get('left_ee_link_name', None),
            left_fing_link_prefix='panda_hand_l_', left_joint_suffix='_l',
            left_rest_arm_qpos=robot_info.get('left_rest_arm_qpos', None),
            debug=args.debug)
        

        breakpoint()
        # TODO: maybe try to create the path planner here?
        self.planner = Planner(
            urdf=robot_path,
            move_group=robot_info['ee_link_name']
        )
        self.left_planner = Planner(
            urdf=robot_path,
            move_group=robot_info['left_ee_link_name']
        )

        # -5, 5, 6

        # roughly planning workflow: plan one arm, then plan the other one, treating
        # the waypoints as obstacles?

        breakpoint()

        return res
    
    def make_anchors(self):
        preset_dynamic_anchor_vertices = get_preset_properties(
            DEFORM_INFO, self.deform_obj, 'deform_anchor_vertices')
        _, mesh = get_mesh_data(self.sim, self.deform_id)
        assert (preset_dynamic_anchor_vertices is not None)

        anchor_positions = []
        for i in range(self.num_anchors):  # make anchors
            anchor_pos = np.array(mesh[preset_dynamic_anchor_vertices[i][0]])
            anchor_positions.append(anchor_pos)
            if not np.isfinite(anchor_pos).all():
                print('anchor_pos not sane:', anchor_pos)
                input('Press enter to exit')
                exit(1)
            link_id = self.robot.info.ee_link_id if i == 0 else \
                self.robot.info.left_ee_link_id
            # self.sim.createSoftBodyAnchor(
            #     self.deform_id, preset_dynamic_anchor_vertices[i][0],
            #     self.robot.info.robot_id, link_id)
        
        # setting robot ee positions to anchor positions (hardcoded to line up with fingers)
        qpos = self.robot.ee_pos_to_qpos(
            ee_pos=anchor_positions[0] + np.array([1.4, 0, 1.4]),
            ee_ori=theta_to_sin_cos([0, -3*np.pi / 4, 0]),
            fing_dist=0.0,
            left_ee_pos=anchor_positions[1] + np.array([-1.4, 0, 1.4]) if self.num_anchors > 1 else None,
            left_ee_ori=theta_to_sin_cos([0, 3*np.pi / 4, 0]),
        )
        self.robot.reset_to_qpos(qpos)

        # attaching anchors between robot and cloth
        for i in range(self.num_anchors):
            link_id = self.robot.info.ee_link_id if i == 0 else \
                self.robot.info.left_ee_link_id
            self.sim.createSoftBodyAnchor(
                self.deform_id, preset_dynamic_anchor_vertices[i][0],
                self.robot.info.robot_id, link_id)

    def get_grip_obs(self):
        grip_obs = []
        ee_pos, _, ee_linvel, _ = self.robot.get_ee_pos_ori_vel()
        grip_obs.extend(ee_pos)
        grip_obs.extend((np.array(ee_linvel) / Tax3dProcClothEnv.MAX_OBS_VEL))
        if self.num_anchors > 1:  # EE pos, vel of left arm
            left_ee_pos, _, left_ee_linvel, _ = \
                self.robot.get_ee_pos_ori_vel(left=True)
            grip_obs.extend(left_ee_pos)
            grip_obs.extend((np.array(left_ee_linvel) / Tax3dProcClothEnv.MAX_OBS_VEL))

        return grip_obs
    
    def do_action_ee_position(self, action):
        action = action.reshape(self.num_anchors, -1)
        ee_pos, ee_ori, _, _ = self.robot.get_ee_pos_ori_vel()
        tgt_ee_pos = action[0, :3]
        tgt_ee_ori = ee_ori if action.shape[-1] == 3 else action[0, 3:]
        tgt_kwargs = {
            'ee_pos': tgt_ee_pos,
            'ee_ori': tgt_ee_ori,
            'fing_dist': 0.0,
        }
        if self.num_anchors > 1: # dual-arm
            left_ee_pos, left_ee_ori, _, _ = self.robot.get_ee_pos_ori_vel(left=True)
            left_tgt_ee_pos = action[1, :3]
            left_tgt_ee_ori = left_ee_ori if action.shape[-1] == 3 else action[1, 3:]
            tgt_kwargs.update({
                'left_ee_pos': left_tgt_ee_pos,
                'left_ee_ori': left_tgt_ee_ori,
                'left_fing_dist': 0.0,
            })
        tgt_qpos = self.robot.ee_pos_to_qpos(**tgt_kwargs)
        n_slack = 1 # use > 1 if robot has trouble reaching the pose
        
        # for now, don't worry about n_slack
        self.robot.move_to_qpos(tgt_qpos, mode=pybullet.POSITION_CONTROL, kp=0.1, kd=1.0)

    def pseudo_expert_action(self, hole_id):
        ee_pos = self.robot.get_ee_pos() + np.array([0, -0.1, 0.0])
        left_ee_pos = self.robot.get_ee_pos(left=True) + np.array([0, -0.1, 0.0])
        return np.concatenate([ee_pos, left_ee_pos])