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
from ..utils.bullet_manipulator import BulletManipulator, theta_to_sin_cos, sin_cos_to_quat
from ..utils.task_info import ROBOT_INFO

from scipy.spatial.transform import Rotation as R

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import copy

# import "constants" from Tax3dEnv
from .tax3d_env import DEFORM_INFO, SCENE_INFO
from .tax3d_proccloth_env import Tax3dProcClothEnv


def clip_vec_mag(vec, mag):
    #print(vec)
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        return vec
    else:
        return vec / vec_norm * min(vec_norm, mag)
    #return vec / np.linalg.norm(vec) * min(np.linalg.norm(vec), mag)

class Tax3dProcClothRobotEnv(Tax3dProcClothEnv):
    """
    Tax3d + robot environment for HangProcCloth task.
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # Setting default task-specific parameters.
        self.scene_name = 'hangcloth'
        self.args.node_density = 25
        self.args.num_holes = 1
    
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
            ee_pos=anchor_positions[0] + np.array([2.8, 0, 2.8]),
            ee_ori=theta_to_sin_cos([0, -3*np.pi / 4, 0]),
            fing_dist=0.0,
            left_ee_pos=anchor_positions[1] + np.array([-2.8, 0, 2.8]) if self.num_anchors > 1 else None,
            left_ee_ori=theta_to_sin_cos([0, 3*np.pi / 4, 0]),
        )
        self.robot.reset_to_qpos(qpos)

        # TODO: this does not work for arbitrary cloth orientations
        # will eventually need to check for which gripper to attach to which corner of the cloth
        self.anchor_constraints = []
        # attaching anchors between robot and cloth
        for i in range(self.num_anchors):
            link_id = self.robot.info.ee_link_id if i == 0 else \
                self.robot.info.left_ee_link_id
            constraint_id = self.sim.createSoftBodyAnchor(
                self.deform_id, preset_dynamic_anchor_vertices[i][0],
                self.robot.info.robot_id, link_id)
            self.anchor_constraints.append(constraint_id)

    def release_anchors(self):
        for constraint_id in self.anchor_constraints:
            self.sim.removeConstraint(constraint_id)

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
    
    def make_final_steps(self):
        self.release_anchors()
        # TODO: maybe open the grippers
        info = super().make_final_steps()
        return info
    
    def get_pcd_obs(self, width=500, height=500):
        obs = super().get_pcd_obs(width, height)

        # Filter out robot link ids from object ids.
        obj_ids = []
        for oid in self.object_ids:
            oid_joints = self.sim.getNumJoints(oid)
            for i in range(-1, oid_joints):
                obj_ids.append(oid + ((i + 1) << 24))
        
        obj_filter = np.isin(obs['ids'], obj_ids)
        obs = {
            'pcd': obs['pcd'][obj_filter],
            'ids': obs['ids'][obj_filter],
        }

        # Map segmentation maks ids to rigid anchor ids.
        for (rigid_anchor_id, rigid_ids) in self.rigid_anchor_ids.items():
            rigid_anchor_seg_ids = []
            for rigid_id in rigid_ids:
                for rigid_id_link in range(-1, self.sim.getNumJoints(rigid_id)):
                    rigid_anchor_seg_ids.append(rigid_id + ((rigid_id_link + 1) << 24))
            obs['ids'][np.isin(obs['ids'], rigid_anchor_seg_ids)] = rigid_anchor_id
        return obs
    
    def do_action_ee_position(self, action):
        """
        Action is delta pose.
        """
        action = action.reshape(self.num_anchors, -1)
        ee_pos, ee_ori, _, _ = self.robot.get_ee_pos_ori_vel()
        tgt_ee_pos = action[0, :3] + ee_pos
        tgt_ee_ori = ee_ori if action.shape[-1] == 3 else action[0, 3:]
        tgt_kwargs = {
            'ee_pos': tgt_ee_pos,
            'ee_ori': tgt_ee_ori,
            'fing_dist': 0.0,
        }
        if self.num_anchors > 1: # dual-arm
            left_ee_pos, left_ee_ori, _, _ = self.robot.get_ee_pos_ori_vel(left=True)
            left_tgt_ee_pos = action[1, :3] + left_ee_pos
            left_tgt_ee_ori = left_ee_ori if action.shape[-1] == 3 else action[1, 3:]
            tgt_kwargs.update({
                'left_ee_pos': left_tgt_ee_pos,
                'left_ee_ori': left_tgt_ee_ori,
                'left_fing_dist': 0.0,
            })
        tgt_qpos = self.robot.ee_pos_to_qpos(**tgt_kwargs)
        n_slack = 1 # use > 1 if robot has trouble reaching the pose
        
        # for now, don't worry about n_slack
        self.robot.move_to_qpos(tgt_qpos, mode=pybullet.POSITION_CONTROL, kp=0.1, kd=1.2)

    def pseudo_expert_action(self, rigid_id, hole_id):       
        goals = self.goal_anchor_positions[rigid_id][hole_id]
        
        # get current loop centroid position
        # loop_vertices = self.args.deform_true_loop_vertices[hole_id]
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        # centroid_points = np.array(vertex_positions)[loop_vertices]
        # centroid_points = centroid_points[~np.isnan(centroid_points).any(axis=1)]
        # centroid = np.mean(centroid_points, axis=0)

        # get positions of anchors
        anchor_positions = []
        for i in range(self.num_anchors):
            anchor_pos = np.array(vertex_positions)[DEFORM_INFO[self.deform_obj]["deform_anchor_vertices"][i]]
            anchor_positions.append(anchor_pos)

        # ------------ FLOW FROM LOOP TO GOAL ------------
        ee_action = goals[0] - anchor_positions[0]
        left_ee_action = goals[1] - anchor_positions[1]
        ee_action = clip_vec_mag(ee_action, 0.1)
        left_ee_action = clip_vec_mag(ee_action, 0.1)

        # ------------- COMPENSATION FOR ORIENTATION ------------
        # compensation factor for orientation
        rigid_vec = (goals[0] - goals[1])[0, 0:2]
        rigid_vec = rigid_vec / np.linalg.norm(rigid_vec)

        anchor_vec = (anchor_positions[0] - anchor_positions[1])[0, 0:2]
        anchor_vec = anchor_vec / np.linalg.norm(anchor_vec)

        # find angle between anchor vector and rigid vector
        cos_theta = np.dot(anchor_vec, rigid_vec)
        sin_theta = np.cross(anchor_vec, rigid_vec)
        theta = np.arctan2(sin_theta, cos_theta)

        # rotate anchor vec by 90 degrees
        ori_offset = np.array([anchor_vec[1], -anchor_vec[0]])
        ee_ori_offset = -ori_offset * theta
        left_ee_ori_offset = ori_offset * theta
        ee_ori_offset = np.concatenate([ee_ori_offset, [0]])
        left_ee_ori_offset = np.concatenate([left_ee_ori_offset, [0]])
        ee_ori_offset = clip_vec_mag(ee_ori_offset, 0.1)
        left_ee_ori_offset = clip_vec_mag(left_ee_ori_offset, 0.1)


        # ------------ COMPENSATION FOR CLOTH SHAPE ------------
        # try to keep distance between grippers roughly equal to the width of the cloth
        deform_params = self.deform_data['deform_params']
        xy_vec = anchor_positions[0][0, 0:2] - anchor_positions[1][0, 0:2]
        cloth_width = DEFORM_INFO[self.deform_obj]["deform_scale"] * deform_params["w"] * 2

        ee_xy_offset =  (cloth_width - np.linalg.norm(xy_vec)) * xy_vec / np.linalg.norm(xy_vec)
        ee_xy_offset = np.concatenate([ee_xy_offset, [0]])
        left_ee_xy_offset = -ee_xy_offset
        ee_xy_offset = clip_vec_mag(ee_xy_offset, 0.1)
        left_ee_xy_offset = clip_vec_mag(left_ee_xy_offset, 0.1)
        
        # ------------ COMPENSATION FOR HEIGHT IMBALANCE ------------
        # try to keep the grippers roughly the same height
        z_average = np.mean([anchor_positions[0][0, 2], anchor_positions[1][0, 2]])
        ee_z_offset = z_average - anchor_positions[0][0, 2]
        left_ee_z_offset = z_average - anchor_positions[1][0, 2]
        ee_z_offset = clip_vec_mag(np.array([0, 0, ee_z_offset]), 0.1)
        left_ee_z_offset = clip_vec_mag(np.array([0, 0, left_ee_z_offset]), 0.1)

        # TODO: compensation for perpendicular standoff?

        ee_action = ee_action + ee_ori_offset + ee_z_offset + ee_xy_offset
        left_ee_action = left_ee_action + left_ee_ori_offset + left_ee_z_offset + left_ee_xy_offset
        ee_action = clip_vec_mag(ee_action, 0.1)
        left_ee_action = clip_vec_mag(left_ee_action, 0.1)
        return np.concatenate([ee_action, left_ee_action], axis=-1).squeeze()
    