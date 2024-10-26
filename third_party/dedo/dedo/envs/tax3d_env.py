import os
import time

from matplotlib import cm  # for colors
import numpy as np
import gym
import pybullet
import pybullet_utils.bullet_client as bclient

from ..utils.anchor_utils import (
    attach_anchor, command_anchor_velocity, command_anchor_position, create_anchor, create_anchor_geom,
    pin_fixed, change_anchor_color_gray)
from ..utils.init_utils import (
    load_deform_object, load_rigid_object, reset_bullet, load_deformable, 
    load_floor, get_preset_properties, apply_anchor_params)
from ..utils.mesh_utils import get_mesh_data
from ..utils.task_info import (
    DEFAULT_CAM_PROJECTION, DEFORM_INFO, SCENE_INFO, TASK_INFO,
    TOTE_MAJOR_VERSIONS, TOTE_VARS_PER_VERSION)
from ..utils.procedural_utils import (
    gen_procedural_hang_cloth, gen_procedural_button_cloth)
from ..utils.args import preset_override_util
from ..utils.process_camera import ProcessCamera, cameraConfig

from scipy.spatial.transform import Rotation as R

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import plotly.graph_objects as go
from PIL import Image
import copy

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
    WORKSPACE_BOX_SIZE = 20.0  # workspace box limits (needs to be >=1)
    STEPS_AFTER_DONE = 500     # steps after releasing anchors at the end
    FORCE_REWARD_MULT = 1e-4   # scaling for the force penalties
    FINAL_REWARD_MULT = 400    # multiply the final reward (for sparse rewards)
    SUCESS_REWARD_TRESHOLD = 2.5  # approx. threshold for task success/failure

    def __init__(self, args):
        self.args = args
        self.cam_on = args.cam_resolution > 0
        self.task_name = ... # TODO: TAKE THIS FROM ARGS?

        # this is hacky - used for proper storing of anchor pcd from first frame
        self.camera_config = None

        # storing scene name
        # TODO: do we need this?
        scene_name = self.args.task.lower()
        if scene_name in ['hanggarment', 'bgarments', 'sewing','hangproccloth']:
           self.scene_name = 'hangcloth'  # same hanger for garments and cloths
        elif scene_name.startswith('button'):
            self.scene_name = 'button'
        elif scene_name.startswith('dress'):
            self.scene_name = 'dress'  # same human figure for dress and mask tasks


        # # Initialize sim and load objects.
        # self.sim = bclient.BulletClient(
        #     connection_mode=pybullet.GUI if args.viz else pybullet.DIRECT)
        # # self.sim = bclient.BulletClient(connection_mode=pybullet.DIRECT)
        # if self.args.viz:  # no rendering during load
        #     self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        # # use egl for rendering
        # self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        # egl = pkgutil.get_loader('eglRenderer')
        # if (egl):
        #     pluginId = self.sim.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        # else:
        #     pluginId = self.sim.loadPlugin('eglRendererPlugin')
        # print('pluginId=', pluginId)


        # Initialize sim.
        if self.args.viz:
            # open GUI, don't use EGL renderer
            self.sim = bclient.BulletClient(connection_mode=pybullet.GUI)
            # no rendering during load
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        else:
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

        # reset_bullet(args, self.sim, debug=args.debug)
        self.food_packing = self.args.env.startswith('FoodPacking')
        self.num_anchors = 1 if self.food_packing else 2
        # TODO: CHANGE OUTPUT OF LOAD OBJECTS TO BE THE DICTIONARIES
        res = self.load_objects(self.sim, self.args, debug=True)
        self.rigid_ids, self.deform_id, self.deform_obj, self.goal_pos, _ = res

        # Step 3: Load floor
        load_floor(self.sim, debug=args.debug)

        self.max_episode_len = self.args.max_episode_len
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

        # Point cloud observation initialization
        self.pcd_mode = args.pcd
        if args.pcd:
            self.camera_config = cameraConfig.from_file(args.cam_config_path)
            self.object_ids = res[0]
            self.object_ids.append(res[1])

            print(f"Starting object ids: {self.object_ids}")
            print(f"Deformable ID: {res[1]}")

        # Storing rollout video.
        self.rollout_vid = args.rollout_vid
        if args.rollout_vid:
            self.vid_frames = []
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
    
    def load_objects(self, sim, args, debug,
                     deform_pose = {}, rigid_pose = {},
                     deform_params = {}, rigid_params = {}):
        raise NotImplementedError("load_objects method must be implemented in the subclass.")
    
    def seed(self, seed):
        np.random.seed(seed)

    def reset(self,
              deform_pose = {}, rigid_pose = {},
              deform_params = {}, rigid_params = {}):
        self.stepnum = 0
        self.rigid_pcd = None
        self.episode_reward = 0.0
        self.anchors = {}
        self.vid_frames = []
        # TODO: store deform_pos, rigid_pos, deofrm_params as well
        self.rigid_params = rigid_params
        self.target_action = None

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
        # TODO: change output type of load_objects
        res = self.load_objects(self.sim, self.args, self.args.debug, deform_pose, rigid_pose, deform_params, rigid_params)
        self.rigid_ids, self.deform_id, self.deform_obj, self.goal_pos, self.deform_params = res
        load_floor(self.sim, plane_texture=plane_texture_path, debug=self.args.debug)

        # Special case for Procedural Cloth tasks that can have two holes:
        # reward is based on the closest hole.
        if self.args.env.startswith('HangProcCloth'):
            self.goal_pos = np.vstack((self.goal_pos, self.goal_pos))

        self.sim.stepSimulation() # step once to get initial state

        debug_mrks = None
        if self.args.debug and self.args.viz:
           debug_mrks = self.debug_viz_true_loop()

        # Setup dynamic anchors.
        if not self.food_packing:
            self.make_anchors()

        # Set up viz.
        if self.args.viz:  # loading done, so enable debug rendering if needed
            time.sleep(0.1)  # wait for debug visualizer to catch up
            self.sim.stepSimulation()  # step once to get initial state
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
            if debug_mrks is not None:
                input('Visualized true loops; press ENTER to continue')
                for mrk_id in debug_mrks:
                    # removeBody doesn't seem to work, so just make invisible
                    self.sim.changeVisualShape(mrk_id, -1,
                                               rgbaColor=[0, 0, 0, 0])
                    
        obs = self.get_obs()

        # Updating rollout video
        if self.rollout_vid:
            self.vid_frames.append(
                self.render(mode='rgb_array', width=self.vid_width, height=self.vid_height)
            )
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
        if not unscaled:
            assert self.action_space.contains(action)
            # assert ((np.abs(action) <= 1.0).all()), 'action must be in [-1, 1]'
        action = action.reshape(self.num_anchors, -1)

        # Step through physics simulation.
        for sim_step in range(self.args.sim_steps_per_action):
            if action_type == 'velocity':
                self.do_action_velocity(action, unscaled)
            elif action_type == 'position':
                self.do_action_position(action, unscaled, tax3d)
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

            # TODO: GOTTA CLEAN THIS UP - GENERALIZING THE SUCCESS CHECKS
            # centroid_check, centroid_dist = self.check_centroid()
            pre_release_check, pre_release_check_info = self.check_pre_release()
            info = self.make_final_steps()
            # polygon_check = self.check_polygon()
            post_release_check, post_release_check_info = self.check_post_release()
            # success requires both checks to pass for at least one hole
            # info['is_success'] = np.any(centroid_check * polygon_check)
            # info['centroid_dist'] = np.mean(centroid_dist)

            info['is_success'] = np.any(pre_release_check * post_release_check)
            info['pre_release_check'] = pre_release_check_info
            info['post_release_check'] = post_release_check_info

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
 
    def make_final_steps(self):
        # We do no explicitly release the anchors, since this can create a jerk
        # and large forces.
        # release_anchor(self.sim, self.anchor_ids[0])
        # release_anchor(self.sim, self.anchor_ids[1])
        change_anchor_color_gray(self.sim, self.anchor_ids[0])
        change_anchor_color_gray(self.sim, self.anchor_ids[1])
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
    
    def get_pcd_obs(self):
        """ Grab Pointcloud observations based from the camera config. """

        # Grab pcd observation from the camera_config camera
        segmented_pcd, segmented_ids, img = ProcessCamera.render(
            self.sim, self.camera_config, width=self.args.cam_resolution,
            height=self.args.cam_resolution, object_ids=self.object_ids,
            return_rgb=True, retain_unknowns=True,
            debug=False)

        # Process the RGB image
        img = img[...,:-1] # drop the alpha
        if self.args.uint8_pixels:
            img = img.astype(np.uint8)  # already in [0,255]
        else:
            img = img.astype(np.float32)/255.0  # to [0,1]
            img = np.clip(img, 0, 1)
        if self.args.flat_obs:
            img = img.reshape(-1)
        atol = 0.0001
        if ((img < self.observation_space.low-atol).any() or
            (img > self.observation_space.high+atol).any()):
            print('img', img.shape, f'{np.min(img):e}, n{np.max(img):e}')
            assert self.observation_space.contains(img)

        # Package and return
        obs = {'img': img,
               'pcd': segmented_pcd,
               'ids': segmented_ids
              }

        return obs
    
    def get_obs(self):
        grip_obs = self.get_grip_obs()
        done = False
        grip_obs = np.nan_to_num(np.array(grip_obs))
        if (np.abs(grip_obs) > self.gripper_lims).any():  # at workspace lims
            if self.args.debug:
                print('clipping grip_obs', grip_obs)
            grip_obs = np.clip(
                grip_obs, -1.0*self.gripper_lims, self.gripper_lims)
            done = True
        # TODO: TAX3D environment should not need to return the image
        if self.args.cam_resolution <= 0:
            obs = grip_obs
        else:
            obs = self.render(mode='rgb_array', width=self.args.cam_resolution,
                              height=self.args.cam_resolution)
            if self.args.uint8_pixels:
                obs = obs.astype(np.uint8)  # already in [0,255]
            else:
                obs = obs.astype(np.float32)/255.0  # to [0,1]
                obs = np.clip(obs, 0, 1)
        if self.args.flat_obs:
            obs = obs.reshape(-1)
        atol = 0.0001
        if ((obs < self.observation_space.low-atol).any() or
            (obs > self.observation_space.high+atol).any()):
            print('obs', obs.shape, f'{np.min(obs):e}, n{np.max(obs):e}')
            assert self.observation_space.contains(obs)

        # --- Getting object-centric point clouds ---

        # action pcd from ground truth mesh
        _, action_pcd = get_mesh_data(self.sim, self.deform_id)
        action_pcd = np.array(action_pcd)
        rigid_pcd = self.rigid_pcd
  
        # TODO: get anchor pcd from camera?
        # slightly confusing notation - "anchor" refers to gripper in DEDO, but the rigid body in TAX3D
        obs_dict = {
            'img': obs, # TODO: this could technically include gripper state - ignored either way
            'gripper_state': grip_obs,
            'done': done,
            'action_pcd': action_pcd,
            'anchor_pcd': rigid_pcd,
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