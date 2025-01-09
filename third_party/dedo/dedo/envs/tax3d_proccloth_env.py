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
from scipy.spatial.distance import cdist

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import copy

# import "constants" from Tax3dEnv
from .tax3d_env import Tax3dEnv, DEFORM_INFO, SCENE_INFO

MIN_ANCHOR_DIST = 8.0


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
        # NOTE: this assumes there is only one deformable object
        # ----------------- LOADING DEFORMABLE OBJECT -----------------
        # Generate procedural cloth, and update deform params.
        deform_params = self.deform_data['deform_params']
        deform_obj, deform_params = gen_procedural_hang_cloth(
            args, 'procedural_hang_cloth', DEFORM_INFO, deform_params
        )
        # self.deform_params = deform_params
        self.deform_data['deform_params'] = deform_params
        preset_override_util(args, DEFORM_INFO[deform_obj])

        # Load deformable texture.
        # deform_texture_path = os.path.join(
        #     args.data_path, self.get_texture_path(args.deform_texture_file)
        # )
        deform_texture_path = deform_params['texture_path']

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
        rigid_ids = []
        goal_poses = []
        # scene_info_copy = copy.deepcopy(SCENE_INFO[self.scene_name])
        # scene_info_copy = apply_rigid_params(self.scene_name, scene_info_copy, self.rigid_params)
        

        # TODO: basically, get the necessary info from rigid_params instead\
        num_rigids = len(self.rigid_data.keys())
        # num_entities = len(scene_info_copy['entities'].keys())


        # Load each rigid object.
        scene_info = {}
        # NOTE: confusing variable names for now...rigid_anchor_ids maps rigid objects to rigid_ids
        self.rigid_anchor_ids = {}
        for i in range(num_rigids):
            rigid_data_i = self.rigid_data[i]
            rigid_params_i = rigid_data_i['rigid_params']
            self.rigid_anchor_ids[i + 1] = []

            # Apply rigid object params.
            # TODO: NEED TO KEEP TRACK OF ALL SCENE INFO COPIES
            scene_info_copy = copy.deepcopy(SCENE_INFO[self.scene_name])
            if 'hanger_scale' in rigid_params_i:
                scene_info_copy['entities']['urdf/hanger.urdf']['globalScaling'] *= rigid_params_i['hanger_scale']
            
            if 'tallrod_scale' in rigid_params_i:
                scene_info_copy['entities']['urdf/tallrod.urdf']['globalScaling'] *= rigid_params_i['tallrod_scale']
            
                # Adjust hanger and goal position based on tallrod scale.
                scene_info_copy['entities']['urdf/hanger.urdf']['basePosition'][2] *= rigid_params_i['tallrod_scale']
                scene_info_copy['goal_pos'][0][2] *= rigid_params_i['tallrod_scale']


            # Load entities for each rigid object.
            for name, kwargs in scene_info_copy['entities'].items():
                # Load rigid texture.
                rgba_color = kwargs['rgbaColor'] if 'rgbaColor' in kwargs else None
                rigid_texture_file = None
                if 'useTexture' in kwargs and kwargs['useTexture']:
                    rigid_texture_file = os.path.join(
                        args.data_path, self.get_texture_path(args.rigid_texture_file)
                    )
                
                id = load_rigid_object(
                    self.sim, os.path.join(args.data_path, name), kwargs['globalScaling'],
                    kwargs['basePosition'], kwargs['baseOrientation'],
                    kwargs.get('mass', 0.0), rigid_texture_file, rgba_color,
                )
                rigid_ids.append(id)
                self.rigid_anchor_ids[i + 1].append(id)
            
            # TODO: SETTING UP THE GOAL POSITION
            # what is the current shape of goal?
            # num_holes * (goal shape)
            # num_rigids  x (goal shape)
            # NOTE: NO LONGER STACKING GOAL POSES FOR MULTIPLE LOOPS - will have to manually iterate for this
            goal_poses.append(scene_info_copy['goal_pos'][0])

            # update scene info
            scene_info[i] = scene_info_copy

        # ----------------- COMPUTING GOAL ANCHOR POSITIONS (PRE-TRANSFORMATION) -----------------
        # before, was num_holes x num_anchors x 1 x 3
        # now, should be num_rigids x num_holes x num_anchors x 3
        _, vertex_positions = get_mesh_data(self.sim, deform_id)
        vertex_positions = np.array(vertex_positions)
        # TODO: if we "flip" the rotation when moving anchor to the other side, this should still work
        anchor_vertices = DEFORM_INFO[deform_obj]['deform_anchor_vertices']
        goal_anchor_positions = []

        # iterate through rigids, then iterate through holes
        for rigid in range(num_rigids):
            offset_dir = np.sign(self.rigid_data[i]['rigid_transform']['translation'][1])
            rigid_goal_anchor_positions = []
            for hole in range(deform_params['num_holes']):
                hole_vertices = self.args.deform_true_loop_vertices[hole]
                centroid_points = vertex_positions[hole_vertices]
                centroid_points = centroid_points[~np.isnan(centroid_points).any(axis=1)]
                centroid = centroid_points.mean(axis=0)

                flow = goal_poses[rigid] - centroid

                # compute the goal positions for each anchor
                hole_goal_anchor_positions = []
                for anchor in range(self.num_anchors):
                    anchor_pos = vertex_positions[anchor_vertices[anchor]]
                    # small offset to make sure the gripper goes past the hanger
                    goal_anchor_pos = anchor_pos + flow + np.array([0, 1.5 * offset_dir, 0.5])
                    hole_goal_anchor_positions.append(goal_anchor_pos)
                rigid_goal_anchor_positions.append(hole_goal_anchor_positions)
            goal_anchor_positions.append(np.asarray(rigid_goal_anchor_positions))

        # ----------------- TRANSFORMING ALL OBJECTS AND GOALS -----------------
        # Transform the deformable object, if necessary.
        deform_transform = self.deform_data['deform_transform']
        if 'rotation' in deform_transform and 'translation' in deform_transform:
            # Apply the transformation to the deformable object.
            deform_rotation = R.from_euler('xyz', deform_transform['rotation'])
            deform_translation = deform_transform['translation']
            # deform_position = deform_rotation.apply(deform_position) + deform_translation
            deform_position = deform_position + deform_translation
            deform_orientation = (deform_rotation * R.from_euler('xyz', deform_orientation)).as_euler('xyz')
            self.sim.resetBasePositionAndOrientation(deform_id, deform_position, pybullet.getQuaternionFromEuler(deform_orientation))
        elif deform_transform:
            raise ValueError('Deformable transformation must specify rotation and translation.')
        
        # Transform the rigid objects and goals, if necessary.
        for i in range(num_rigids):
            rigid_data_i = self.rigid_data[i]
            rigid_transform_i = rigid_data_i['rigid_transform']

            if 'rotation' in rigid_transform_i and 'translation' in rigid_transform_i:
                rigid_rotation = R.from_euler('xyz', rigid_transform_i['rotation'])
                rigid_translation = rigid_transform_i['translation']

                # Apply the transformation to the rigid objects.
                scene_info_copy = scene_info[i]
                num_entities = len(scene_info_copy['entities'].keys())
                for j, (name, kwargs) in enumerate(scene_info_copy['entities'].items()):
                    rigid_position = kwargs['basePosition']
                    rigid_orientation = kwargs['baseOrientation']
                    rigid_position = rigid_position + rigid_translation
                    rigid_orientation = (rigid_rotation * R.from_euler('xyz', rigid_orientation)).as_euler('xyz')
                    self.sim.resetBasePositionAndOrientation(
                        rigid_ids[i * num_entities + j], rigid_position, pybullet.getQuaternionFromEuler(rigid_orientation)
                    )
                
                # Apply the transformation to the goal data.
                # TODO: is this bugged? rigid is doing rotation -> translation, but this is flipped?
                goal_poses_i = goal_poses[i]
                goal_anchor_positions_i = goal_anchor_positions[i]
                goal_poses[i] = rigid_rotation.apply(goal_poses_i) + rigid_translation
                goal_anchor_positions[i] = np.asarray([
                    [rigid_rotation.apply(anchor_goal) + rigid_translation for anchor_goal in hole_goal_anchor_positions]
                    for hole_goal_anchor_positions in goal_anchor_positions_i
                ])
        
        return {
            'deform_id': deform_id,
            'deform_obj': deform_obj,
            'rigid_ids': rigid_ids,
            'goal_poses': np.array(goal_poses),
            'goal_anchor_positions': np.array(goal_anchor_positions),
        }

    # TODO: this signature is outdated (should include rigid_id as well) - but low priority
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

    def random_anchor_transform(self, num_anchors, random_flip=False):
        # brute force sampling
        while True:
            z_rot = np.random.uniform(-np.pi / 3, np.pi / 3, size=num_anchors)
            rotation = R.from_euler('z', z_rot).as_euler('xyz')

            translation = np.array([
                np.random.uniform(size=num_anchors) * 5 * np.power(-1, z_rot < 0),
                np.random.uniform(size=num_anchors) * -10,
                np.array([0.0] * num_anchors)
            ]).T

            # randomly flip anchor to the other side (accounting for initial cloth pose)
            if random_flip:
                flip_idx = np.random.choice([True, False], size=num_anchors)
                # translation[flip, 0] = -translation[flip, 0] + 2 * DEFORM_INFO['procedural_hang_cloth']['deform_init_pos'][0]
                translation[flip_idx, 1] = -translation[flip_idx, 1] + 2 * DEFORM_INFO['procedural_hang_cloth']['deform_init_pos'][1]
                rotation[flip_idx, 2] *= -1

            # check if any anchor is too close to another
            dists = cdist(translation, translation, metric='euclidean')

            # check that all anchors are at least MIN_ANCHOR_DIST apart, but ignore the diagonal
            if np.all(dists[np.triu_indices(num_anchors, k=1)] > MIN_ANCHOR_DIST):
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
        vertex_positions = np.array(vertex_positions)

        # output is num_rigids x num_holes
        num_rigids = len(self.goal_pos)
        num_holes = len(self.args.deform_true_loop_vertices)
        centroid_checks = np.zeros((num_rigids, num_holes), dtype=bool)
        centroid_dists = np.zeros((num_rigids, num_holes))

        for j in range(num_holes):
            true_loop_vertices = self.args.deform_true_loop_vertices[j]
            cent_pts = vertex_positions[true_loop_vertices]
            cent_pts = cent_pts[~np.isnan(cent_pts).any(axis=1)]
            cent_pos = cent_pts.mean(axis=0)
            for i in range(num_rigids):
                goal_pos = self.goal_pos[i]
                dist = np.linalg.norm(cent_pos - goal_pos)
                centroid_checks[i, j] = dist < 1.5
                centroid_dists[i, j] = dist
        
        return centroid_checks, centroid_dists

    def check_post_release(self):
        # polygon check
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        vertex_positions = np.array(vertex_positions)

        # output is num_rigids x num_holes
        num_rigids = len(self.goal_pos)
        num_holes = len(self.args.deform_true_loop_vertices)
        polygon_checks = np.zeros((num_rigids, num_holes), dtype=bool)

        for j in range(num_holes):
            true_loop_vertices = self.args.deform_true_loop_vertices[j]
            true_loop_vertices = order_loop_vertices(true_loop_vertices)
            cent_pts = vertex_positions[true_loop_vertices]
            polygon = Polygon(cent_pts[:, :2])
            for i in range(num_rigids):
                goal_pos = self.goal_pos[i]
                point = Point(goal_pos[:2])
                polygon_checks[i, j] = polygon.contains(point)

        return polygon_checks, None