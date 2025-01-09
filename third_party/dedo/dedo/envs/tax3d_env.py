import os
import time

from matplotlib import cm  # for colors
import numpy as np
import gym
import pybullet
import pybullet_utils.bullet_client as bclient

from ..utils.anchor_utils import (
    attach_anchor, command_anchor_velocity, command_anchor_position, create_anchor, create_anchor_geom,
    pin_fixed, change_anchor_color_gray
)
from ..utils.init_utils import (
    reset_bullet,load_floor, get_preset_properties
)
from ..utils.mesh_utils import get_mesh_data
from ..utils.task_info import (
    DEFAULT_CAM_PROJECTION, DEFORM_INFO, SCENE_INFO, TASK_INFO,
    TOTE_MAJOR_VERSIONS, TOTE_VARS_PER_VERSION
)
from ..utils.args import preset_override_util


from scipy.spatial.transform import Rotation as R

import pkgutil
import pybullet_data


class Tax3dEnv(gym.Env):
    """
    This is a base class for the tax3d environment that all of the task-specific classes will inherit 
    form for convenience, with the purpose of consolidating common functionality. Most of the code is borrowed 
    from the original DeformEnv.
    """
    MAX_OBS_VEL = 20.0  # max vel (in m/s) for the anchor observations
    MAX_ACT_VEL = 10.0  # max vel (in m/s) for the anchor actions
    WORKSPACE_BOX_SIZE = 30.0  # workspace box limits (needs to be >=1)
    STEPS_AFTER_DONE = 500     # steps after releasing anchors at the end
    FORCE_REWARD_MULT = 1e-4   # scaling for the force penalties
    FINAL_REWARD_MULT = 400    # multiply the final reward (for sparse rewards)
    SUCESS_REWARD_TRESHOLD = 2.5  # approx. threshold for task success/failure

    def __init__(self, args):
        self.args = args
        
        # Initialize sim.
        if self.args.viz:
            # open GUI, don't use EGL renderer
            self.sim = bclient.BulletClient(connection_mode=pybullet.GUI)
            # no rendering during load
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        else:
            # no GUI, use EGL renderer
            self.sim = bclient.BulletClient(connection_mode=pybullet.DIRECT)
            # use egl for rendering
            self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())
            egl = pkgutil.get_loader('eglRenderer')
            if (egl):
                pluginId = self.sim.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
                pluginId = self.sim.loadPlugin('eglRendererPlugin')
            print('pluginId=', pluginId)

        reset_bullet(self.args, self.sim, debug=args.debug)

        # Check version - v0 uses random textures.
        if args.version == 0:
            args.use_random_textures = True
        # Update data path for assets.
        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        args.data_path = data_path
        self.sim.setAdditionalSearchPath(data_path)
        # setting additional environment params.
        self.food_packing = self.args.env.startswith('FoodPacking')
        self.num_anchors = 1 if self.food_packing else 2
        self.max_episode_len = self.args.max_episode_len

        # Initializing object params - these can be randomized at reset.
        self.deform_data = {}
        self.rigid_data = {}

        # Define sizes of observation and action spaces.
        self.gripper_lims = np.tile(np.concatenate(
            [Tax3dEnv.WORKSPACE_BOX_SIZE * np.ones(3),  # 3D pos
             np.ones(3)]), self.num_anchors)             # 3D linvel/MAX_OBS_VEL
        if args.cam_resolution <= 0:  # report gripper positions as low-dim obs
            self.observation_space = gym.spaces.Box(
                -1.0 * self.gripper_lims, self.gripper_lims)
        else:  # RGB WxHxC
            shape = (args.cam_resolution, args.cam_resolution, 3)
            if args.flat_obs:
                shape = (np.prod(shape),)
            self.observation_space = gym.spaces.Box(
                low=0, high=255 if args.uint8_pixels else 1.0,
                dtype=np.uint8 if args.uint8_pixels else np.float16,
                shape=shape)
        act_sz = 3  # 3D linear velocity for anchors
        self.action_space = gym.spaces.Box(  # [-1, 1]
            -2.0 * np.ones(self.num_anchors * act_sz),
            2.0 * np.ones(self.num_anchors * act_sz))
        if self.args.debug:
            print('Created Tax3dEnv with obs', self.observation_space.shape,
                  'act', self.action_space.shape)

        # Point cloud observations variables - these should be set during reset.
        self.pcd_view_mat = None
        self.pcd_proj_mat = None
        self.object_ids = None

        # Storing rollout video.
        self.rollout_vid = args.rollout_vid
        if args.rollout_vid:
            self.vid_frames = []
            self.color_key = None
            self.vid_width = 500
            self.vid_height = 500

    @staticmethod
    def unscale_vel(act, unscaled):
        if unscaled:
            return act
        return act*Tax3dEnv.MAX_ACT_VEL

    @property
    def anchor_ids(self):
        return list(self.anchors.keys())

    @property
    def _cam_viewmat(self):
        dist, pitch, yaw, pos_x, pos_y, pos_z = self.args.cam_viewmat
        cam = {
            'distance': dist,
            'pitch': pitch,
            'yaw': yaw,
            'cameraTargetPosition': [pos_x, pos_y, pos_z],
            'upAxisIndex': 2,
            'roll': 0,
        }
        view_mat = self.sim.computeViewMatrixFromYawPitchRoll(**cam)
        return view_mat
    
    def get_texture_path(self, file_path):
        # Get either pre-specified texture file or a random one.
        if self.args.use_random_textures:
            parent = os.path.dirname(file_path)
            full_parent_path = os.path.join(self.args.data_path, parent)
            randfile = np.random.choice(list(os.listdir(full_parent_path)))
            file_path = os.path.join(parent, randfile)
        return file_path
    
    def load_objects(self, args):
        raise NotImplementedError("load_objects method must be implemented in the subclass.")
    
    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, deform_data = {}, rigid_data = {}):
        self.stepnum = 0
        self.rigid_pcd = None
        self.rigid_pcd_ids = None
        self.episode_reward = 0.0
        self.anchors = {}
        self.vid_frames = []
        self.color_key = None
        self.target_action = None

        # Updating object data.
        self.deform_data = deform_data
        self.rigid_data = rigid_data


        if self.args.viz:  # no rendering during load
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        # Reset pybullet sim to clear out deformables and reload objects.
        # plane_texture_path = os.path.join(
        #     self.args.data_path,  self.get_texture_path(
        #         self.args.plane_texture_file))
        
        # FIXING THE PLANE TEXTURE FOR NOW
        plane_texture_path = os.path.join(
            self.args.data_path,
            'textures/plane/lightwood.jpg'
        )

        reset_bullet(self.args, self.sim, plane_texture=plane_texture_path)
        res = self.load_objects(self.args)
        self.deform_id = res['deform_id']
        self.deform_obj = res['deform_obj']
        self.rigid_ids = res['rigid_ids']
        self.goal_pos = res['goal_poses']
        self.goal_anchor_positions = res['goal_anchor_positions']

        # Updating point cloud observation variables.
        self.pcd_view_mat = self._cam_viewmat
        self.pcd_proj_mat = DEFAULT_CAM_PROJECTION['projectionMatrix']
        self.object_ids = self.rigid_ids + [self.deform_id]

        # Load the floor.
        load_floor(self.sim, plane_texture=plane_texture_path, debug=self.args.debug)

        self.sim.stepSimulation() # step once to get initial state

        # Handling debug visualization.
        debug_mrks = None
        if self.args.debug and self.args.viz:
            # Visualize goal positions.
            for i, goal_pos in enumerate(self.goal_pos):
                print(f'goal_pos{i}', goal_pos)
                alpha = 1 if i == 0 else 0.3  # primary vs secondary goal
                create_anchor_geom(self.sim, goal_pos, mass=0.0,
                                   rgba=(0, 1, 0, alpha), use_collision=False)
            # Visualize true loops.
            debug_mrks = self.debug_viz_true_loop()

        # Setup dynamic anchors.
        if not self.food_packing:
            self.make_anchors()

        # Set up viz.
        if self.args.viz:  # loading done, so enable debug rendering if needed
            time.sleep(0.1)  # wait for debug visualizer to catch up
            self.sim.stepSimulation()  # step once to get initial state
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

            # Reset debug visualizer camera.
            dist, pitch, yaw, pos_x, pos_y, pos_z = self.args.cam_viewmat
            self.sim.resetDebugVisualizerCamera(
                cameraDistance=dist, cameraPitch=pitch, cameraYaw=yaw,
                cameraTargetPosition=[pos_x, pos_y, pos_z]
            )

            if debug_mrks is not None:
                input('Visualized true loops; press ENTER to continue')
                for mrk_id in debug_mrks:
                    # removeBody doesn't seem to work, so just make invisible
                    self.sim.changeVisualShape(mrk_id, -1,
                                               rgbaColor=[0, 0, 0, 0])           
        
        # Setting rigid point cloud - still using initial frame for now.
        pcd, ids = self.get_pcd_obs().values()
        self.rigid_pcd = pcd[ids > 0]
        self.rigid_pcd_ids = ids[ids > 0]

        obs = self.get_obs()

        # Updating rollout video
        if self.rollout_vid:
            self.vid_frames.append(
                self.render(mode='rgb_array', width=self.vid_width, height=self.vid_height)
            )
            action_pcd = obs['action_pcd']
            self.color_key = action_pcd[:, 0] + action_pcd[:, 2]
        return obs
    
    def make_anchors(self):
        preset_dynamic_anchor_vertices = get_preset_properties(
            DEFORM_INFO, self.deform_obj, 'deform_anchor_vertices')
        _, mesh = get_mesh_data(self.sim, self.deform_id)
        for i in range(self.num_anchors):  # make anchors
            anchor_init_pos = self.args.anchor_init_pos if (i % 2) == 0 else \
                self.args.other_anchor_init_pos
            anchor_id, anchor_pos, anchor_vertices = create_anchor(
                self.sim, anchor_init_pos, i,
                preset_dynamic_anchor_vertices, mesh)
            attach_anchor(self.sim, anchor_id, anchor_vertices, self.deform_id)
            self.anchors[anchor_id] = {'pos': anchor_pos,
                                       'vertices': anchor_vertices}
            
    def debug_viz_true_loop(self):
        # DEBUG visualize true loop vertices
        # Note: this function can be very slow when the number of ground truth
        # vertices marked is large, because it will create many visual elements.
        # So, use it sparingly (e.g. just only at trajectory start/end).
        if not hasattr(self.args, 'deform_true_loop_vertices'):
            return
        true_vs_lists = self.args.deform_true_loop_vertices
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        all_vs = np.array(vertex_positions)
        mrk_ids = []
        clrs = [cm.tab10(i) for i in range(len(true_vs_lists))]
        for l, loop_v_lst in enumerate(true_vs_lists):
            curr_vs = all_vs[loop_v_lst]
            for v in curr_vs:
                mrk_ids.append(create_anchor_geom(
                    self.sim, v, mass=0.0, radius=0.05,
                    rgba=clrs[l], use_collision=False))
        return mrk_ids
    
    def step(self, action, action_type='velocity', unscaled=False, tax3d=False):
        # TODO: this is hacky, maybe just remove the action_space.contains check
        if action_type == 'position':
            unscaled = True

        # print(action)
        if self.args.debug:
            print('action', action)
        # if not unscaled:
        #     assert self.action_space.contains(action)
        #     # assert ((np.abs(action) <= 1.0).all()), 'action must be in [-1, 1]'
        action = action.reshape(self.num_anchors, -1)

        # Step through physics simulation.
        for sim_step in range(self.args.sim_steps_per_action):
            if action_type == 'velocity':
                self.do_action_velocity(action, unscaled)
            elif action_type == 'position':
                self.do_action_position(action, unscaled, tax3d)
            elif action_type == 'ee_position':
                self.do_action_ee_position(action)
            else:
                raise ValueError(f'Unknown action type {action_type}')
            self.sim.stepSimulation()

        # Get next obs, reward, done.
        next_obs = self.get_obs()
        done = next_obs["done"]
        reward = self.get_reward()
        if done:  # if terminating early use reward from current step for rest
            reward *= (self.max_episode_len - self.stepnum)
        done = (done or self.stepnum >= self.max_episode_len)

        # Updating rollout video
        if self.rollout_vid:
            self.vid_frames.append(
                self.render(mode='rgb_array', width=self.vid_width, height=self.vid_height)
            )

        # Update episode info and call make_final_steps if needed.
        if done:
            if self.rollout_vid:
                pre_release_frame = self.render(mode='rgb_array', width=self.vid_width, height=self.vid_height)
            # compute pre-release and post-release success checks
            pre_release_check, pre_release_check_info = self.check_pre_release()
            info = self.make_final_steps()
            post_release_check, post_release_check_info = self.check_post_release()

            # success requires both checks to pass for at least one hole
            info['is_success'] = np.any(pre_release_check * post_release_check)
            info['pre_release_check'] = pre_release_check_info
            info['post_release_check'] = post_release_check_info
            # TODO: this assumes pre-release check is centroid dist
            info['centroid_dist'] = np.min(pre_release_check_info)

            last_rwd = self.get_reward() * Tax3dEnv.FINAL_REWARD_MULT
            reward += last_rwd
            info['final_reward'] = reward

            # Returning rollout video
            if self.rollout_vid:
                info['vid_frames'] = self.vid_frames
                info['color_key'] = self.color_key
                info['viewmat'] = np.array(self._cam_viewmat)
                info['pre_release_frame'] = pre_release_frame
        
        else:
            info = {}

        self.episode_reward += reward  # update episode reward

        if self.args.debug and self.stepnum % 10 == 0:
            print(f'step {self.stepnum:d} reward {reward:0.4f}')
            if done:
                print(f'episode reward {self.episode_reward:0.4f}')
            
        self.stepnum += 1

        return next_obs, reward, done, info
    
    def do_action_velocity(self, action, unscaled):
        # Action is num_anchors x 3 for 3D velocity for anchors/grippers.
        # Assume action in [-1,1], convert to [-MAX_ACT_VEL, MAX_ACT_VEL].
        for i in range(self.num_anchors):
            command_anchor_velocity(
                self.sim, self.anchor_ids[i],
                Tax3dEnv.unscale_vel(action[i], unscaled))

    def do_action_position(self, action, unscaled, tax3d):
        # uses basic proportional position control instead
        for i in range(self.num_anchors):
            command_anchor_position(
                self.sim, self.anchor_ids[i],
                action[i],
                tax3d=tax3d,
                task='proccloth'
            )
 
    def do_action_ee_position(self, action, unscaled):
        raise NotImplementedError("do_action_ee_position method must be implemented in the subclass.")

    def make_final_steps(self):
        # We do no explicitly release the anchors, since this can create a jerk
        # and large forces.
        # release_anchor(self.sim, self.anchor_ids[0])
        # release_anchor(self.sim, self.anchor_ids[1])
        
        # change_anchor_color_gray(self.sim, self.anchor_ids[0])
        # change_anchor_color_gray(self.sim, self.anchor_ids[1])
        info = {'final_obs': []}
        for sim_step in range(Tax3dEnv.STEPS_AFTER_DONE):
            # For lasso pull the string at the end to test lasso loop.
            # For other tasks noop action to let the anchors fall.
            if self.args.task.lower() == 'lasso':
                if sim_step % self.args.sim_steps_per_action == 0:
                    action = [10*Tax3dEnv.MAX_ACT_VEL,
                              10*Tax3dEnv.MAX_ACT_VEL, 0]
                    self.do_action(action, unscaled=True)
            self.sim.stepSimulation()
            if sim_step % self.args.sim_steps_per_action == 0:
                # next_obs, _ = self.get_obs()
                next_obs = self.get_obs()
                info['final_obs'].append(next_obs)

                # Updating rollout video
                if self.rollout_vid:
                    self.vid_frames.append(
                        self.render(mode='rgb_array', width=self.vid_width, height=self.vid_height)
                    )
        return info
    
    def get_pcd_obs(self, width=500, height=500):
        """ 
        Grab Pointcloud observations based from the camera config. Fixed resolution for now. 
        Credit: Zackory Erickson's `mengine' repo (https://github.com/Zackory/mengine/blob/main/mengine/env.py).
        """
        far = 10000.0
        view_matrix = self.pcd_view_mat
        proj_matrix = self.pcd_proj_mat
        # Check if view_matrix and proj_matrix are set.
        if view_matrix is None or proj_matrix is None:
            raise ValueError("view_matrix and proj_matrix must be set - reset the environment first.")

        # Grab depth render from PyBullet camera.
        _, _, _, depth, segment_mask = self.sim.getCameraImage(
            width=width, height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            shadow=False,
            lightDirection=[1,1,1],
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )
        depth = depth.flatten()
        segment_mask = segment_mask.flatten()

        # Create 4x4 transfrom from pixel coordinates (and depth values) to world coordinates.
        view_matrix = np.array(view_matrix).reshape((4, 4), order="F")
        proj_matrix = np.array(proj_matrix).reshape((4, 4), order="F")
        pix2world_matrix = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # Create grid with pixel coordinates and depth values.
        y, x = np.mgrid[-1:1:2 / width, -1:1:2 / height]
        y *= -1.0
        x, y, z = x.reshape(-1), y.reshape(-1), depth
        h = np.ones_like(z)
        pixels = np.stack([x, y, z, h], axis=1)

        # Filter point cloud to only include object points.
        if self.object_ids is not None:
            # Retain unknown ids.
            unique_ids = np.unique(segment_mask)
            object_ids = self.object_ids + list(unique_ids[unique_ids>1E6])
            mask = np.zeros_like(segment_mask)
            for object_id in object_ids:
                mask = np.logical_or(mask, segment_mask == object_id)
            # Mask out points that are not part of the object.    
            pixels = pixels[mask]
            z = z[mask]
            segment_mask = segment_mask[mask]

        # Filter out points that are too far away.
        pixels = pixels[z < far]
        segment_mask = segment_mask[z < far]
        pixels[:, 2] = 2.0 * pixels[:, 2] - 1.0

        # Transform pixel coordinates to world coordinates.
        points = np.matmul(pixels, pix2world_matrix.T)
        points = points[:, :3] / points[:, 3:4]

        # Package and return.
        obs = {
            'pcd': points,
            'ids': segment_mask,
        }
        return obs
    
    def get_obs(self):
        # --- Getting gripper state observation ---
        grip_obs = self.get_grip_obs()
        done = False
        grip_obs = np.nan_to_num(np.array(grip_obs))
        if (np.abs(grip_obs) > self.gripper_lims).any():  # at workspace lims
            if self.args.debug:
                print('clipping grip_obs', grip_obs)
            grip_obs = np.clip(
                grip_obs, -1.0*self.gripper_lims, self.gripper_lims)
            done = True

        # --- Getting object-centric point clouds ---

        # Get action-object (deformable) point cloud from mesh.
        _, action_pcd = get_mesh_data(self.sim, self.deform_id)
        action_pcd = np.array(action_pcd)
        action_seg = np.zeros(action_pcd.shape[0])

        # Get anchor-object (rigid) from first-frame point cloud.
        if self.rigid_pcd is not None:
            rigid_pcd = self.rigid_pcd
            rigid_pcd_ids = self.rigid_pcd_ids
        else:
            raise ValueError("Rigid pcd not set - must be set during reset.")

        obs_dict = {
            'gripper_state': grip_obs,
            'done': done,
            'action_pcd': action_pcd,
            'action_seg': action_seg,
            'anchor_pcd': rigid_pcd,
            'anchor_seg': rigid_pcd_ids,
        }
        return obs_dict
    
    def get_grip_obs(self):
        anc_obs = []
        for i in range(self.num_anchors):
            pos, _ = self.sim.getBasePositionAndOrientation(
                self.anchor_ids[i])
            linvel, _ = self.sim.getBaseVelocity(self.anchor_ids[i])
            anc_obs.extend(pos)
            anc_obs.extend((np.array(linvel)/Tax3dEnv.MAX_OBS_VEL))
        return anc_obs
    
    def get_reward(self, debug=False):
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        dist = []
        if not hasattr(self.args, 'deform_true_loop_vertices'):
            return 0.0  # no reward info without info about true loops
        # Compute distance from loop/hole to the corresponding target.
        num_holes_to_track = min(
            len(self.args.deform_true_loop_vertices), len(self.goal_pos))
        for i in range(num_holes_to_track):  # loop through goal vertices
            true_loop_vertices = self.args.deform_true_loop_vertices[i]
            goal_pos = self.goal_pos[i]
            pts = np.array(vertex_positions)
            cent_pts = pts[true_loop_vertices]
            cent_pts = cent_pts[~np.isnan(cent_pts).any(axis=1)]  # remove nans
            if len(cent_pts) == 0 or np.isnan(cent_pts).any():
                dist = Tax3dEnv.WORKSPACE_BOX_SIZE*num_holes_to_track
                dist *= Tax3dEnv.FINAL_REWARD_MULT
                # Save a screenshot for debugging.
                # obs = self.render(mode='rgb_array', width=300, height=300)
                # pth = f'nan_{self.args.env}_s{self.stepnum}.npy'
                # np.save(os.path.join(self.args.logdir, pth), obs)
                break
            cent_pos = cent_pts.mean(axis=0)
            if debug:
                print('cent_pos', cent_pos, 'goal_pos', goal_pos)
            dist.append(np.linalg.norm(cent_pos - goal_pos))

        if self.args.env.startswith('HangProcCloth'):
            dist = np.min(dist)
        else:
            dist = np.mean(dist)
        rwd = -1.0 * dist / Tax3dEnv.WORKSPACE_BOX_SIZE
        return rwd
    
    def check_centroid(self):
        raise NotImplementedError("check_centroid method must be implemented in the subclass.")
    
    def check_polygon(self):
        raise NotImplementedError("check_polygon method must be implemented in the subclass.")
    
    def render(self, mode='rgb_array', width=300, height=300):
        assert (mode == 'rgb_array')
        w, h, rgba_px, _, _ = self.sim.getCameraImage(
            width=width, height=height,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=self._cam_viewmat, **DEFAULT_CAM_PROJECTION)
        # If getCameraImage() returns a tuple instead of numpy array that
        # means that pybullet was installed without numpy being present.
        # Uninstall pybullet, then install numpy, then install pybullet.
        assert (isinstance(rgba_px, np.ndarray)), 'Install numpy, then pybullet'
        img = rgba_px[:, :, 0:3]
        return img
    
    def pseudo_expert_action(self):
        raise NotImplementedError("pseudo_expert method must be implemented in the subclass.")
    
    def random_anchor_transform(self):
        raise NotImplementedError("random_anchor_transform method must be implemented in the subclass.")
    
    def random_anchor_transform_ood(self):
        raise NotImplementedError("random_anchor_transform_ood method must be implemented in the subclass.")
    
    def check_pre_release(self):
        raise NotImplementedError("check_pre_release method must be implemented in the subclass.")
    
    def check_post_release(self):
        raise NotImplementedError("check_post_release method must be implemented in the subclass.")