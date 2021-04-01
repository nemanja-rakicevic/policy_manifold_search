
"""
Author:         Anonymous
Description:
                Implementation of the Panda robot environment in PyBullet
                - PandaStrikerEnv

Resources:
                Code adapted from:
                https://github.com/mahyaret/gym-panda/blob/master/gym_panda/envs/panda_env.py
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random


_VEL_THRSH = .009  # 0.1
_BALL_DAMPING = 0.05  # 0.1
_MAX_AGENT_STEPS = 100
_EPS = 1e-06
_REW_THRSH = 0.2




class PandaStrikerEnv(gym.Env):
    metadata = {"render.modes": ["human"]}


    MAX_AGENT_STEPS = _MAX_AGENT_STEPS
    VEL_THRSH = _VEL_THRSH
    BALL_DAMPING = _BALL_DAMPING

    EE_LINK = 8  # 11


    def __init__(self):
        self.init = True
        self.range_joints = [[-2.9671, 2.9671],
                             [-1.8326, 1.8326],
                             [-2.9671, 2.9671],
                             [-3.1416, 0.],
                             [-2.9671, 2.9671],
                             [-0.0873, 3.8223],
                             [-2.9671, 2.9671]]
        self.range_fingers = [[0., 0.04],
                              [0., 0.04]]
        self.ball_ranges = np.array([[-0.46, 0.46],
                                     [0.32, 1.41]])
        # Initial Robot and Ball poses
        self.init_ball_pos = [0., 0.55, 0]
        self.init_ee_pos = [0., 0.5, 0.01]
        self.init_robot_pos = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]  # [0, -1.5, .5]
        # Simulator information
        self.num_joints = 7
        self.contact_objects = []
        self._p = p
        self.timestep = 0.0165 / 4
        self.frame_skip = 4
        self.num_solver_iterations = 5
        self.urdfRootPath = pybullet_data.getDataPath()
        self._p.connect(self._p.DIRECT)
        # p.connect(p.GUI)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
        # p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
        # p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        # Action space definition
        self.dim_action = self.num_joints
        self.limit_action = 10
        self.action_space = spaces.Box(
            low=-self.limit_action, 
            high=self.limit_action, 
            shape=(self.dim_action,), 
            dtype=np.float32)
        # self.action_space = spaces.Box(np.array([-1] * 4), np.array([1] * 4))
        # Observation space definition
        self.dim_observation = 2*self.num_joints + 2*3 + 4
        self.limit_joint = 1
        self.observation_space = spaces.Box(
            low=-self.limit_joint, 
            high=self.limit_joint, 
            shape=(self.dim_observation,), 
            dtype=np.float32)
        # self.observation_space = spaces.Box(np.array([-1] * 5), np.array([1] * 5))
        self.param_ranges = np.vstack([self.action_space.low,
                                       self.action_space.high]).T
        self.env_info = dict(
            num_targets=1,
            num_obstacles=0,
            wall_geoms=[0, 1, 2, 3],
            ball_geom=5,
            target_info= [{'xy': (-0.5, 1.), 'radius': 0.25 }] ,
            striker_ranges=self.param_ranges,
            ball_ranges=self.ball_ranges)
        self.init = False


    def _get_observation(self):
        joint_states = self._p.getJointStates(self.pandaUid,
                                              list(range(self.num_joints)))
        robot_joints_pos, robot_joints_vel = np.array(joint_states)[:,:2].T
        robot_ee_pos = self._p.getLinkState(self.pandaUid, self.EE_LINK)[0]
        robot_ee_orient = self._p.getEulerFromQuaternion(
            self._p.getLinkState(self.pandaUid, self.EE_LINK)[1])
        ball_pos, _ = self._p.getBasePositionAndOrientation(self.ballUid)
        ball_vel, _ = self._p.getBaseVelocity(self.ballUid)
        observation = np.concatenate([robot_joints_pos,
                                      robot_joints_vel,
                                      robot_ee_pos,
                                      robot_ee_orient,
                                      ball_pos[:2],
                                      ball_vel[:2]])
        return observation

    def _get_info_dict(self, state=None, action=np.zeros(8)):
        # Get obs vector and info dict
        puck_pos = np.array(
            self._p.getBasePositionAndOrientation(self.ballUid)[0][:2])
        # s_pos, s_orient = self._p.getBasePositionAndOrientation(self.strikerUid)
        s_pos = self._p.getLinkState(self.pandaUid, self.EE_LINK)[0]
        s_orient = self._p.getEulerFromQuaternion(
            self._p.getLinkState(self.pandaUid, self.EE_LINK)[1])
        striker_pose = np.concatenate([np.array(s_pos[:2]), [s_orient[2]]])

        vec = striker_pose[:2] - puck_pos                    
        info_dict = dict(position=puck_pos,  # puck pos
                         position_aux=striker_pose,  # striker pose
                         final_dist=np.linalg.norm(vec), 
                         final_ctrl=np.linalg.norm(action),
                         contact_objects=self.contact_objects)
        return info_dict

    def _get_done(self, state, action):
        # episode is done when the ball stops, or complete miss
        ball_pos, _ = self._p.getBasePositionAndOrientation(self.ballUid)
        ball_vel, _ = self._p.getBaseVelocity(self.ballUid)
        puck_pos = np.linalg.norm(ball_pos[:2])
        puck_vel = np.linalg.norm(ball_vel[:2])
        strk_vel = np.linalg.norm(action)
        # Termination conditions
        done = puck_vel<=self.VEL_THRSH and puck_pos>_EPS or \
               puck_vel<=self.VEL_THRSH and strk_vel<=self.VEL_THRSH 
               # and np.isclose(puck_pos, 0., atol=_EPS)
        return done and not self.init

    def _get_reward(self, state):
        # Reward vector contains: distances to targets, ball coordinates (x,y)
        # target_coms = [self.get_body_com(n)[:2] \
        #                   for n in self.unwrapped.model.body_names \
        #                   if 'target' in n]
        # target_dist = [np.linalg.norm(tc - self.get_body_com("ball")[:2]) \
        #                   for tc in target_coms]
        # striker_pos = self._p.getBasePositionAndOrientation(self.strikerUid)[0]
        striker_pos = self._p.getLinkState(self.pandaUid, self.EE_LINK)[0]
        target_dist = -np.linalg.norm(striker_pos)
        return target_dist #+ [tuple(state[3:5])]

    def _check_contacts(self):
        """
            Hack to get proper contacts and ball bounces
        """
        puck_pos, _ = self._p.getBasePositionAndOrientation(self.ballUid)
        ball_x, ball_y = puck_pos[:2]
        puck_vel, _ = self._p.getBaseVelocity(self.ballUid)
        ball_vx, ball_vy = puck_vel[:2]
        # check wall vicinity        
        wall_W, wall_E = np.isclose(ball_x, self.ball_ranges[0,:], atol=0.05)
        wall_S, wall_N = np.isclose(ball_y, self.ball_ranges[1,:], atol=0.05)
        # check change of direction
        dv_x = np.sign(ball_vx) != np.sign(self.prev_ball_vx)
        dv_y = np.sign(ball_vy) != np.sign(self.prev_ball_vy)
        # evaluate contacts
        contact = np.array([wall_S*dv_y, wall_E*dv_x, wall_N*dv_y, wall_W*dv_x])
        # make a proper bounce, keep 90% of velocity
        if contact.any() and self.nstep_internal > 1:
            ball_vel, ball_avel = self._p.getBaseVelocity(self.ballUid)
            if dv_x: ball_vx = -0.9 * self.prev_ball_vx
            if dv_y: ball_vy = -0.9 * self.prev_ball_vy
            self._p.resetBaseVelocity(
                self.ballUid,
                linearVelocity=[ball_vx, ball_vy, ball_vel[2]],
                angularVelocity=[0, 0, 0]) 
                # angularVelocity=ball_avel) 
        # update prev ball_xy
        self.prev_ball_vx = ball_vx
        self.prev_ball_vy = ball_vy
        # return wall indices
        return np.where(contact)[0]+1

    def initialize(self, seed_task, init_params=None, **kwargs):
        """ Override default initialisation """
        # Set task seed
        self.seed(seed_task)
        self.action_space.seed(seed_task)
        # standard reset
        state = self.reset()
        info_dict = self._get_info_dict(state)
        return state, info_dict['position'], info_dict['position_aux']

    def finalize(self, state, traj_aux, **kwargs):
        """ Define outcome: target index if within range, or -1 if failed """
        reward = self._get_reward(state)
        # returns closest target index if within threshold, otherwise -1
        trial_outcome = -1
        # trial_outcome = np.argmin(reward[:-1]) \
        #                 if np.sum(reward[-1])>0. and \
        #                    np.min(reward[:-1])<=_REW_THRSH else -1
        trial_fitness = -len(np.unique(traj_aux.astype(np.float16), axis=0))
        return np.array([trial_outcome, trial_fitness])
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        ### CHECK steps
        if self.nstep_internal > self.MAX_AGENT_STEPS: 
            action = 0 * action
        self.nstep_internal += 1
        ### CHECK ACTION
        assert (np.isfinite(action).all())
        # action = np.clip(action, -1, +1).astype(float)
        # Set joint velocities
        self._p.setJointMotorControlArray(
            bodyIndex=self.pandaUid,
            jointIndices=list(range(self.num_joints)),
            controlMode=self._p.VELOCITY_CONTROL,
            targetVelocities=action,
            # controlMode=self._p.TORQUE_CONTROL,
            # forces=action,
        )
        # Execute simulation step
        self._p.stepSimulation()
        # Get observation in info
        observation = self._get_observation()
        done = self._get_done(observation, action)
        reward = self._get_reward(observation)
        info_dict = self._get_info_dict(observation, action)
        alt_contact = self._check_contacts()
        if len(alt_contact):
            self.contact_objects.append(alt_contact[0])
        return np.array(observation).astype(np.float32), reward, done, info_dict

    def reset(self):
        self.nstep_internal = -1
        self.contact_objects = []
        self.prev_ball_vx = 0
        self.prev_ball_vy = 0
        # Restart simulator
        # p.connect(p.GUI)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
        # p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
        # p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        self._p.resetDebugVisualizerCamera(
            cameraDistance=1.7,
            cameraYaw=90,
            cameraPitch=-40,
            cameraTargetPosition=[-0.35, 0.8, 0.2])
        # self._p.resetDebugVisualizerCamera(
        #     cameraDistance=.9,
        #     cameraYaw=0,
        #     cameraPitch=-80,
        #     cameraTargetPosition=[0., 0.5, 0.2])

        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self.timestep * self.frame_skip,
            numSolverIterations=self.num_solver_iterations,
            numSubSteps=self.frame_skip)
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)  
        # Load table playground and configure dynamics
        self._p.setGravity(0, 0, -9.81)
        planeUid = self._p.loadURDF(
            os.path.join(
                self.urdfRootPath,
                "plane.urdf"),
            basePosition=[0, 0, -0.65])
        tableUid = self._p.loadURDF(
            os.path.join(
                os.path.dirname(__file__),
                "assets/table/table.urdf"),
            basePosition=[0., 0.6, -0.65],
            baseOrientation=[0.0, 0.0, 0.7071067811865475, 0.7071067811865476])
        self._p.changeDynamics(tableUid, -1,
            lateralFriction=.8,
            restitution=0.5,
            rollingFriction=0.005)
        self.borderUid = self._p.loadURDF(
            os.path.join(
                os.path.dirname(__file__),
                "assets/objects/tray_box.urdf"),
            basePosition=[0., 0.2, 0.],
            baseOrientation=[0.0, 0.0, 0.7071067811865475, 0.7071067811865476])
        self._p.changeDynamics(self.borderUid, -1,
            lateralFriction=.8,
            restitution=0.5,
            rollingFriction=0.005)
        # Load robot and configure dynamics
        self.pandaUid = self._p.loadURDF(
            os.path.join(
                os.path.dirname(__file__),
                "assets/franka_panda/panda.urdf"),
            baseOrientation=[0.0, 0.0, 0.7071067811865475, 0.7071067811865476],
            useFixedBase=True)
        # put arm close to striker
        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        for i, jval in enumerate(rest_poses):
            self._p.resetJointState(self.pandaUid, i, jval)
        init_orientation = p.getQuaternionFromEuler([0.0, -math.pi, math.pi / 2.0])
        init_joints = p.calculateInverseKinematics(
            self.pandaUid, self.EE_LINK, self.init_ee_pos, init_orientation)[:self.num_joints]
        for i, jval in enumerate(init_joints):
            self._p.resetJointState(self.pandaUid, i, jval)
        # Load ball and configure dynamics
        self.ballUid = self._p.loadURDF(
            os.path.join(
                os.path.dirname(__file__),
                "assets/objects/ball_small.urdf"),
            basePosition=self.init_ball_pos)
        self._p.changeDynamics(self.ballUid, -1, restitution=1, mass=2.5)
        # Get observation
        observation = self._get_observation()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        return np.array(observation).astype(np.float32)

    def render(self, mode="human"):
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.7, 0, 0.05],
            distance=0.7,
            yaw=90,
            pitch=-70,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(960) / 720, nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=960,
            height=720,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        self._p.disconnect()
