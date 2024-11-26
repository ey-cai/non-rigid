import os
import time

from matplotlib import cm  # for colors
import numpy as np
import gym
import pybullet
import pybullet_utils.bullet_client as bclient

from ..utils.init_utils import (
    load_deform_object, load_rigid_object, apply_rigid_params
)
from ..utils.mesh_utils import get_mesh_data
from ..utils.procedural_utils import gen_procedural_hang_cloth
from ..utils.args import preset_override_util

from scipy.spatial.transform import Rotation as R

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import copy

# import "constants" from Tax3dEnv
from .tax3d_env import Tax3dEnv, DEFORM_INFO, SCENE_INFO


def order_loop_vertices(vertices):
    """ Order the vertices of a loop in a clockwise manner. """
    top = []
    left = []
    right = []
    bottom = []
    # getting all top vertices
    for i in range(len(vertices)):
        top.append(vertices[i])
        if vertices[i + 1] - vertices[i] != 1:
            vertices = vertices[i+1:]
            break

    # getting all bottom vertices
    for i in range(1, len(vertices) + 1):
        bottom.append(vertices[-i])
        if vertices[-i] - vertices[-i - 1] != 1:
            vertices = vertices[:-i]
            break

    # getting left and right vertices
    while vertices:
        left.append(vertices[0])
        right.append(vertices[1])
        vertices = vertices[2:]
    
    # reverse left vertices
    left = left[::-1]
    return top + right + bottom + left

class Tax3dProcClothEnv(Tax3dEnv):
    """
    Tax3d environment for HangProcCloth task.
    """

    def __init__(self, args):
        super().__init__(args)

        # Setting default task-specific parameters.
        self.scene_name = 'hangcloth'
        self.args.node_density = 25
        self.args.num_holes = 1

    def load_objects(self, args):
        # ----------------- LOADING DEFORMABLE OBJECT -----------------
        # Generate procedural cloth, and update deform params.
        deform_obj, deform_params = gen_procedural_hang_cloth(
            args, 'procedural_hang_cloth', DEFORM_INFO, self.deform_params
        )
        self.deform_params = deform_params
        preset_override_util(args, DEFORM_INFO[deform_obj])

        # Load deformable texture.
        deform_texture_path = os.path.join(
            args.data_path, self.get_texture_path(args.deform_texture_file)
        )

        # Load the deformable object.
        deform_position = args.deform_init_pos
        deform_orientation = args.deform_init_ori
        deform_id = load_deform_object(
            self.sim, deform_obj, deform_texture_path, args.deform_scale,
            deform_position, deform_orientation,
            args.deform_bending_stiffness, args.deform_damping_stiffness,
            args.deform_elastic_stiffness, args.deform_friction_coeff,
            not args.disable_self_collision, args.debug,
        )

        # ----------------- LOADING RIGID OBJECT -----------------
        # Apply rigid object parameters.
        scene_info_copy = copy.deepcopy(SCENE_INFO[self.scene_name])
        scene_info_copy = apply_rigid_params(self.scene_name, scene_info_copy, self.rigid_params)
        # TODO: apply_rigid_params is obsolete now that there are task-specific classes
        # should replace later, and move code into here
        rigid_ids = []
        for name, kwargs in scene_info_copy['entities'].items():
            # Load rigid texture.
            rgba_color = kwargs['rgbaColor'] if 'rgbaColor' in kwargs else None
            rigid_texture_file = None
            if 'useTexture' in kwargs and kwargs['useTexture']:
                rigid_texture_file = os.path.join(
                    args.data_path, self.get_texture_path(args.rigid_texture_file)
                )

            # Load the rigid object.
            rigid_position = kwargs['basePosition']
            rigid_orientation = kwargs['baseOrientation']
            id = load_rigid_object(
                self.sim, os.path.join(args.data_path, name), kwargs['globalScaling'],
                # kwargs['basePosition'], kwargs['baseOrientation'],
                rigid_position, rigid_orientation,
                kwargs.get('mass', 0.0), rigid_texture_file, rgba_color,
            )
            rigid_ids.append(id)
        
        # ----------------- SETTING UP GOAL POSITION -----------------
        # Mark the goal and store intermediate info for reward computations.
        goal_poses = scene_info_copy['goal_pos']
        goal_poses = np.vstack([goal_poses, goal_poses])
        # if 'rotation' in self.rigid_transform and 'translation' in self.rigid_transform:
        #     goal_poses = [
        #         R.from_euler('xyz', self.rigid_transform['rotation']).apply(goal_pos) + self.rigid_transform['translation']
        #         for goal_pos in goal_poses
        #     ]
        # ----------------- COMPUTING GOAL POSITIONS PRE-TRANSFORMATION -----------------
        _, vertex_positions = get_mesh_data(self.sim, deform_id)
        vertex_positions = np.array(vertex_positions)
        # NOTE: the order here is dependent on DEFORM_INFO - arbitrary object transformations could result in cloth-flipping
        # NOTE: this also means we should be careful about which gripper we attach to which anchor
        anchor_vertices = DEFORM_INFO[deform_obj]['deform_anchor_vertices']
        goal_anchor_positions = []
        
        for hole_id in range(self.deform_params["num_holes"]):
            hole_vertices = self.args.deform_true_loop_vertices[hole_id]
            centroid_points = vertex_positions[hole_vertices]
            centroid_points = centroid_points[~np.isnan(centroid_points).any(axis=1)]
            centroid = centroid_points.mean(axis=0)

            flow = goal_poses[hole_id] - centroid

            # compute the goal positions for each anchor
            hole_goal_anchor_positions = []
            for anchor in range(self.num_anchors):
                anchor_pos = vertex_positions[anchor_vertices[anchor]]
                # small offset to make sure the gripper goes past the hanger
                goal_anchor_pos = anchor_pos + flow + np.array([0, -1.5, 0.5])
                hole_goal_anchor_positions.append(goal_anchor_pos)
            goal_anchor_positions.append(hole_goal_anchor_positions)


        # ----------------- TRANSFORMING ALL OBJECTS AND GOALS -----------------
        # Transform the deformable object, if necessary.
        if 'rotation' in self.deform_transform and 'translation' in self.deform_transform:
            # Apply the transformation to the deformable object.
            deform_rotation = R.from_euler('xyz', self.deform_transform['rotation'])
            deform_translation = self.deform_transform['translation']
            # deform_position = deform_rotation.apply(deform_position) + deform_translation
            deform_position = deform_position + deform_translation
            deform_orientation = (deform_rotation * R.from_euler('xyz', deform_orientation)).as_euler('xyz')
            self.sim.resetBasePositionAndOrientation(deform_id, deform_position, pybullet.getQuaternionFromEuler(deform_orientation))
        elif self.deform_transform:
            raise ValueError('Deformable transformation must specify rotation and translation.')
        
        # Transform the rigid objects and goals, if necessary.
        if 'rotation' in self.rigid_transform and 'translation' in self.rigid_transform:
            rigid_rotation = R.from_euler('xyz', self.rigid_transform['rotation'])
            rigid_translation = self.rigid_transform['translation']
            # Apply the transformation to the rigid objects.
            for i, (name, kwargs) in enumerate(scene_info_copy['entities'].items()):
                rigid_position = kwargs['basePosition']
                rigid_orientation = kwargs['baseOrientation']
                # rigid_position = rigid_rotation.apply(rigid_position) + rigid_translation
                rigid_position = rigid_position + rigid_translation
                rigid_orientation = (rigid_rotation * R.from_euler('xyz', rigid_orientation)).as_euler('xyz')
                self.sim.resetBasePositionAndOrientation(rigid_ids[i], rigid_position, pybullet.getQuaternionFromEuler(rigid_orientation))

            # Apply the transformation to the goal data.
            goal_poses = [
                rigid_rotation.apply(goal_pos) + rigid_translation
                for goal_pos in goal_poses
            ]
            goal_anchor_positions = [
                [rigid_rotation.apply(anchor_goal) + rigid_translation for anchor_goal in hole_goal_anchor_positions]
                for hole_goal_anchor_positions in goal_anchor_positions
            ]
        
        return {
            'deform_id': deform_id,
            'deform_obj': deform_obj,
            'rigid_ids': rigid_ids,
            'goal_poses': np.array(goal_poses),
            'goal_anchor_positions': goal_anchor_positions,
        }


    def pseudo_expert_action(self, hole_id):
        """
        Pseudo-expert action for demonstration generation. This is basic position control using the distance 
        from the loop centroid to the goal position.
        """
        if self.target_action is not None:
            action = self.target_action
        else:
            # default goal pose
            if 'tallrod_scale' in self.rigid_params:
                default_goal_pos = [0, 0.00, 8.2 * self.rigid_params['tallrod_scale']]
            else:
                default_goal_pos = [0, 0.00, 8.2]

            # getting goal position and loop centroid
            # goal_pos = self.goal_pos[hole_id]
            true_loop_vertices = self.args.deform_true_loop_vertices[hole_id]
            _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
            vertex_positions = np.array(vertex_positions)
            centroid_points = vertex_positions[true_loop_vertices]
            centroid_points = centroid_points[~np.isnan(centroid_points).any(axis=1)]
            centroid = centroid_points.mean(axis=0)

            # getting the flow vector
            flow = default_goal_pos - centroid
            flow += np.array([0, -1.5, 0]) # grippers should go slightly past anchor
            grip_obs = self.get_grip_obs()
            a1_pos = grip_obs[0:3]
            a2_pos = grip_obs[6:9]

            if self.rigid_transform is not None:
                # transforming default goal position
                R_default2goal = R.from_euler('xyz', self.rigid_transform['rotation'])
                t_default2goal = np.array(self.rigid_transform['translation'])
                a1_act = R_default2goal.apply(a1_pos + flow) + t_default2goal
                a2_act = R_default2goal.apply(a2_pos + flow) + t_default2goal

            else:
                #a1_act = default_goal_pos - centroid
                #a2_act = default_goal_pos - centroid
                a1_act = a1_pos + flow
                a2_act = a2_pos + flow

            action = np.concatenate([a1_act, a2_act], axis=0).astype(np.float32)
            self.target_action = action

        # goal correction
        goal_pos = self.goal_pos[hole_id]
        true_loop_vertices = self.args.deform_true_loop_vertices[hole_id]
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        vertex_positions = np.array(vertex_positions)
        centroid_points = vertex_positions[true_loop_vertices]
        centroid_points = centroid_points[~np.isnan(centroid_points).any(axis=1)]
        centroid = centroid_points.mean(axis=0)

        correction = goal_pos - centroid
        correction /= max(np.linalg.norm(correction), 1.0)
        action = action + np.concatenate([correction, correction], axis=0).astype(np.float32)
        return action

    def random_deform_transform(self):
        z_rot = np.random.uniform(-np.pi / 3, np.pi / 3)
        rotation = R.from_euler('z', z_rot)
        # translation = np.array([0, 0, 0])

        translation = np.array([
            np.random.uniform(-1, 1),
            np.random.uniform(0, 2),
            np.random.uniform(-1, 1),
        ])
        return rotation, translation

    def random_cloth_transform(self):
        raise NotImplementedError("Need to implement this")

    def random_anchor_transform(self):
        z_rot = np.random.uniform(-np.pi / 3, np.pi / 3)
        rotation = R.from_euler('z', z_rot)
        # translation = np.array([
        #     np.random.uniform() * 5 * np.power(-1, z_rot < 0),
        #     np.random.uniform() * -10,
        #     0.0
        # ])
        translation = np.array([
            np.random.uniform() * 3.5 * np.power(-1, z_rot < 0),
            np.random.uniform() * -7.5,
            0.0
        ])
        return rotation, translation

    def random_anchor_transform_ood(self):
        z_rot = np.random.uniform(-np.pi / 3, np.pi / 3)
        rotation = R.from_euler('z', z_rot)
        # translation = np.array([
        #     np.random.uniform(5, 10) * np.power(-1, z_rot < 0),
        #     np.random.uniform() * -10,
        #     np.random.uniform(1, 5)
        # ])
        # TODO: very hacky for now; just randomly sample until we get a valid position
        while True:
            translation = np.array([
                np.random.uniform() * 7 * np.power(-1, z_rot < 0),
                np.random.uniform() * -12,
                0.0
            ])
            translation_abs = np.abs(translation)
            if translation_abs[0] >= 3.5 or translation_abs[1] >= 7.5:
                break
        return rotation, translation

    def check_pre_release(self):
        # centroid check
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        centroid_checks = []
        centroid_dists = []
        num_holes_to_track = min(
            len(self.args.deform_true_loop_vertices), len(self.goal_pos)
        )
        for i in range(num_holes_to_track):
            true_loop_vertices = self.args.deform_true_loop_vertices[i]
            goal_pos = self.goal_pos[i]
            pts = np.array(vertex_positions)
            cent_pts = pts[true_loop_vertices]
            cent_pts = cent_pts[~np.isnan(cent_pts).any(axis=1)]
            cent_pos = cent_pts.mean(axis=0)
            dist = np.linalg.norm(cent_pos - goal_pos)
            centroid_checks.append(dist < 1.5)
            centroid_dists.append(dist)
        return np.array(centroid_checks), np.array(centroid_dists)

    def check_post_release(self):
        # polygon check
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        polygon_checks = []
        num_holes_to_track = min(
            len(self.args.deform_true_loop_vertices), len(self.goal_pos)
        )
        for i in range(num_holes_to_track):
            true_loop_vertices = self.args.deform_true_loop_vertices[i]
            true_loop_vertices = order_loop_vertices(true_loop_vertices)
            goal_pos = self.goal_pos[i]
            pts = np.array(vertex_positions)
            cent_pts = pts[true_loop_vertices]
            polygon = Polygon(cent_pts[:, :2])
            point = Point(goal_pos[:2])
            polygon_checks.append(polygon.contains(point))
        return np.array(polygon_checks), None