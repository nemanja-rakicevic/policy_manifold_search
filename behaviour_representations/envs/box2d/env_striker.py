
"""
Author:         Anonymous
Description:
                Implementation of the PuckStriker environment in Box2D
                - StrikerAugmentedEnv
                - StrikerAugmentedMixScaleEnv
"""

import sys
import math

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, 
                      revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle

"""
Resources:
- https://github.com/pybox2d/pybox2d/wiki/manual
- https://www.iforce2d.net/b2dtut/constant-speed
"""

### TASK VARIABLES # NEW STRIKER => _VEL_THRSH = 0.1, BALL_DAMPING = 0.1
_VEL_THRSH = .009  # 0.1
_BALL_DAMPING = 0.05  # 0.1
_MAX_AGENT_STEPS = 100
_EPS = 1e-06
_REW_THRSH = 0.2


### SIMULATOR VARIABLES
SCALE  = 5.0   # affects how fast-paced the game is, 
               # forces should be adjusted as well
VEL_SCALE = 100
MOTOR_GAIN = 1.
VIEWPORT_W = 500
VIEWPORT_H = 500
FPS = 12500/VIEWPORT_W
FPS = 25
W = VIEWPORT_W/SCALE
H = VIEWPORT_H/SCALE
DV = W/100  # W/SCALE/2
DH = H/100  # H/SCALE/2
# DD = VIEWPORT_W/SCALE/40
DD = W/40  #VIEWPORT_W/SCALE/(2000/SCALE)
STIKER_POLY =[(-2*DD, DD), (2*DD, DD), (2*DD, -DD), (-2*DD, -DD)]
WALL_VERT = [(0, H/2), (0, -H/2), (DV, H/2), (DV, -H/2)]
WALL_HORZ = [(W/2, 0), (-W/2, 0), (W/2, DH), (-W/2, DH)]
WALL_POS = [(W/2, 0), (W-DH, H/2), (W/2, H-DV), (0, H/2)] # S,E,N,W



def pixel2xy(pixel, axis):
    if axis == 'H':
        return (pixel - H/4) / (H/2)
    elif axis == 'W':
        return (pixel - W/2) / (H/2)


def xy2pixel(xy, axis):
    if axis == 'H':
        return xy * H/2 + H/4
    elif axis == 'W':
        return xy * W/2 + W/2



class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    
    def BeginContact(self, contact):
        if self.env.puck==contact.fixtureA.body or \
           self.env.puck==contact.fixtureB.body:
            if contact.fixtureA.body in self.env.walls:
                wall_idx = 1 + self.env.walls.index(contact.fixtureA.body)
                self.env.contact_objects.append(wall_idx)
            if contact.fixtureB.body in self.env.walls:
                wall_idx = 1 + self.env.walls.index(contact.fixtureB.body)
                self.env.contact_objects.append(wall_idx)
    def EndContact(self, contact):
        pass



class StrikerEnv(gym.Env, EzPickle):

    MAX_AGENT_STEPS = _MAX_AGENT_STEPS
    VEL_THRSH = _VEL_THRSH
    BALL_DAMPING = _BALL_DAMPING

    def __init__(self):
        EzPickle.__init__(self)
        # Initialise environment objects
        self.init = True
        self.world = Box2D.b2World(gravity=(0, 0)) #, doSleep=True)
        self.striker = None
        self.puck = None
        self.target = None  # make this a list in the future
        self.walls = []
        self.contact_objects = []
        # Define observation and action spaces
        self.dim_observation = 10
        self.dim_action = 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.dim_observation,), 
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1*MOTOR_GAIN, high=1*MOTOR_GAIN, 
                                       shape=(self.dim_action,), dtype=np.float32)
        # Set environment info_dict
        self.viewer = None
        _offset = DH/H + DD/W  # due to the wall thickness and ball size
        self.ball_ranges = np.array([[ -1.+_offset, 1.-_offset ],
                                     [ -.5+_offset, 1.5-_offset ]])     
        self.param_ranges = np.array([[-1.+_offset, 1.-_offset ],
                                      [-.5+_offset, 1.5-_offset],
                                      [      -3.14,        3.14]])
        self.env_info = dict(
            num_targets=1, 
            num_obstacles=0,
            wall_geoms=[0, 1, 2, 3],
            ball_geom=0,
            striker_ranges=self.param_ranges[:2, :],
            ball_ranges=self.ball_ranges)
        # Reset environment
        self.reset()
        self.init = False



    def initialize(self, seed_task, init_params=None, **kwargs):
        """ Override default initialisation """
        # Set task seed
        self.seed(seed_task)
        self.action_space.seed(seed_task)
        # Reset env
        self.reset()
        # Set initial striker pose from parameters
        if init_params is None:
            init_params = np.mean(self.param_ranges, axis=1)
            init_params[1] = -0.3 #np.min(self.param_ranges[1])
            
        self.striker.position.x = xy2pixel(init_params[0], 'W')
        self.striker.position.y = xy2pixel(init_params[1], 'H')
        self.striker.angle = init_params[2]

        # Set initial puck pose to zero
        # self.puck.position.x = xy2pixel(0, 'W')
        # self.puck.position.y = xy2pixel(0, 'H')
        # self.puck.angle = 0
        # self.step(np.array([0,0,0]))
        # Initialise trajectories
        state = self._get_state()
        pos_puck = self.puck.position
        puck_pos = np.array([pixel2xy(pos_puck.x, 'W'), 
                             pixel2xy(pos_puck.y, 'H')])
        traj_ball = puck_pos
        traj_striker = init_params
        return state, traj_ball, traj_striker


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


    def _get_reward(self, state):
        # Reward vector contains: distances to targets, ball coordinates (x,y)
        # target_coms = [self.get_body_com(n)[:2] \
        #                   for n in self.unwrapped.model.body_names \
        #                   if 'target' in n]
        # target_dist = [np.linalg.norm(tc - self.get_body_com("ball")[:2]) \
        #                   for tc in target_coms]
        pos_striker = self.striker.position
        striker_pos = np.array([pixel2xy(pos_striker.x, 'W'), 
                                pixel2xy(pos_striker.y, 'H')])
        target_dist = -np.linalg.norm(striker_pos)
        return target_dist #+ [tuple(state[3:5])]


    def _get_state(self):
        # Get variables
        pos_striker = self.striker.position
        vel_striker = self.striker.linearVelocity
        avel_striker = self.striker.angularVelocity
        pos_puck = self.puck.position
        vel_puck = self.puck.linearVelocity
        pos_target = self.target.position
        # Convert
        striker_x = pixel2xy(pos_striker.x, 'W')
        striker_y = pixel2xy(pos_striker.y, 'H')
        puck_x = pixel2xy(pos_puck.x, 'W')
        puck_y = pixel2xy(pos_puck.y, 'H')
        target_x = pixel2xy(pos_target.x, 'W')
        target_y = pixel2xy(pos_target.y, 'H')
        # Get observation
        state = np.array([
                    # position data - striker 
                    striker_x, striker_y, self.striker.angle,
                    # position data - puck 
                    puck_x, puck_y,
                    # velocity data - striker
                    vel_striker.x, vel_striker.y, avel_striker,
                    # position data - puck 
                    vel_puck.x, vel_puck.y,
                    ], dtype=np.float32)
        assert len(state)==self.dim_observation
        return state


    def _get_info_dict(self, action, state):
        # Get obs vector and info dict
        pos_puck = self.puck.position
        puck_pos = np.array([pixel2xy(pos_puck.x, 'W'), 
                             pixel2xy(pos_puck.y, 'H')])
        pos_striker = self.striker.position
        striker_pos = np.array([pixel2xy(pos_striker.x, 'W'), 
                                pixel2xy(pos_striker.y, 'H'),
                                self.striker.angle])
        vec = striker_pos[:2] - puck_pos                    
        info_dict = dict(position=puck_pos,  # puck pos
                         position_aux=striker_pos,  # striker pose
                         final_dist=np.linalg.norm(vec), 
                         final_ctrl=np.linalg.norm(action),
                         contact_objects=self.contact_objects)
        return info_dict


    def _get_done(self, action, state):
        # episode is done when the ball stops, or complete miss
        pos_puck = self.puck.position
        puck_pos = np.linalg.norm([pixel2xy(pos_puck.x, 'W'), 
                                   pixel2xy(pos_puck.y, 'H')])
        puck_vel = np.linalg.norm(self.puck.linearVelocity)
        strk_vel = np.linalg.norm(action)
        # Termination conditions
        done = puck_vel<=self.VEL_THRSH and puck_pos>_EPS or \
               puck_vel<=self.VEL_THRSH and strk_vel<=self.VEL_THRSH 
               # and np.isclose(puck_pos, 0., atol=_EPS)
        # print("\n===", self.nstep_internal, puck_vel, strk_vel, action)
        # print("===", puck_vel<=_VEL_THRSH , puck_pos>_EPS)
        # print("===", puck_vel<=_VEL_THRSH ,puck_vel<=_VEL_THRSH, np.isclose(puck_pos, 0., atol=_EPS))
        # print("===", done)
        return done and not self.init


    def reset(self):
        self.nstep_internal = -1
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        # Walls
        for i, w in enumerate(WALL_POS):
            wbody = self.world.CreateStaticBody(
                        position=w,
                        fixtures=fixtureDef(
                            shape=polygonShape(
                                    vertices=WALL_VERT if i%2 else WALL_HORZ),
                            density=5.0,
                            friction=0.1,
                            # categoryBits=0x0010,
                            # maskBits=0x001,  # collide only with ground
                            restitution=0.0)
                                )
            wbody.color1 = (0.0,0.0,0.0)
            wbody.color2 = (0.0,0.0,0.0)
            self.walls.append(wbody)
        # Target
        target_x = xy2pixel(-0.5, 'W')
        target_y = xy2pixel( 1., 'H')
        self.target = self.world.CreateStaticBody(
                        position = (target_x, target_y),
                        angle=0.0,
                        fixtures = fixtureDef(
                            shape=circleShape(pos=(0,0), radius=4*DD),
                            density=5.0,
                            friction=0.1,
                            categoryBits=0x0010,
                            # maskBits=0x001,  # collide only with ground
                            # restitution=0.0
                            )
                        )
        self.target.color1 = (0.4,0.9,0.4)
        self.target.color2 = (0.4,0.9,0.4)
        # Striker
        striker_init_x = xy2pixel(0., 'W')
        striker_init_y = xy2pixel(0., 'H') 
        self.striker = self.world.CreateDynamicBody(
            position = (striker_init_x, striker_init_y),
            angle=0.0,
            linearDamping = 0.05,
            angularDamping = 0.5,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=STIKER_POLY),
                density=5.0,
                friction=0.1,
                # categoryBits=0x0010,
                # maskBits=0x0010,  # collide only with ground
                restitution=0.0) # 0.99 bouncy
                )
        self.striker.color1 = (0.9,0.4,0.4)
        self.striker.color2 = (0.9,0.4,0.4)
        # Puck
        puck_init_x = xy2pixel(0., 'W')
        puck_init_y = xy2pixel(0., 'H')
        self.puck = self.world.CreateDynamicBody(
            position = (puck_init_x, puck_init_y),
            angle=0.0,
            linearDamping = self.BALL_DAMPING,  # EFFECTIVE!
            fixtures = fixtureDef(
                shape=circleShape(pos=(0,0), radius=DD),
                density=5.0,
                friction=0.9,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.1) # 0.99 bouncy
                )
        self.puck.color1 = (0.4,0.4,0.9)
        self.puck.color2 = (0.4,0.4,0.9)
        # Draw objects
        self.drawlist = [self.target] + self.walls + [self.striker, self.puck]
        return self._get_state() 
        # return self.step(np.array([0,0,0]))[0]


    def step(self, action):
        # print("\n====", self.nstep_internal, self.puck.linearVelocity, np.linalg.norm(self.puck.linearVelocity))
        if self.nstep_internal > self.MAX_AGENT_STEPS: 
            action = 0 * action
        self.nstep_internal += 1
        action = np.clip(action, -1, +1)
        action = np.array([10., 10., 1.]) * action
        # Apply action
        self.striker.linearVelocity.x = float(action[0])
        self.striker.linearVelocity.y = float(action[1])
        self.striker.angularVelocity  = float(action[2])
        # Simulator step
        # self.world.Step(1.0/FPS, 6*int(SCALE), 2*int(SCALE))
        self.world.Step(1.0/FPS, 10, 10)
        # Get return vars
        state = self._get_state()
        reward = self._get_reward(state)
        done = self._get_done(action, state)
        info_dict = self._get_info_dict(action, state)
        return state, reward, done, info_dict


    def _destroy(self):
        if not self.striker: return
        self.world.contactListener = None
        self.contact_objects = []
        self.world.DestroyBody(self.striker)
        self.striker = None
        self.world.DestroyBody(self.puck)
        self.puck = None
        self.world.DestroyBody(self.target)
        self.target = None
        for w in self.walls:
            self.world.DestroyBody(w)
        self.walls = []


    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, 
                                            color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, 
                                            color=obj.color2, filled=False, 
                                            linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, 
                                              linewidth=2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]




class StrikerAugmentedEnv(StrikerEnv):

    def __init__(self):
        EzPickle.__init__(self)
        # Initialise environment objects
        self.init = True
        self.world = Box2D.b2World(gravity=(0, 0)) #, doSleep=True)
        self.striker = None
        self.puck = None
        self.target = None  # make this a list in the future
        self.walls = []
        self.contact_objects = []
        # Define observation and action spaces
        self.dim_observation = 14
        self.dim_action = 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.dim_observation,), 
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1*MOTOR_GAIN, high=1*MOTOR_GAIN, 
                                       shape=(self.dim_action,), dtype=np.float32)
        # Set environment info_dict
        self.viewer = None
        target_info = [{'xy': (-0.5, 1.), 'radius': 0.25 }]         ### FIX THIS
        _offset = DH/H + DD/W  # due to the wall thickness and ball size
        self.ball_ranges = np.array([[ -1.+_offset, 1.-_offset ],
                                     [ -.5+_offset, 1.5-_offset ]])     
        self.param_ranges = np.array([[-1.+_offset, 1.-_offset ],
                                      [-.5+_offset, 1.5-_offset],
                                      [      -3.14,        3.14]])
        self.env_info = dict(
            num_targets=1, 
            num_obstacles=0,
            wall_geoms=[0, 1, 2, 3],
            ball_geom=0,
            target_info=target_info,
            striker_ranges=self.param_ranges[:2, :],
            ball_ranges=self.ball_ranges)
        # Reset environment
        self.reset()
        self.init = False


    def _get_state(self):
        # Get variables
        pos_striker = self.striker.position
        vel_striker = self.striker.linearVelocity
        avel_striker = self.striker.angularVelocity
        pos_puck = self.puck.position
        vel_puck = self.puck.linearVelocity
        pos_target = self.target.position
        # Convert
        striker_x = pixel2xy(pos_striker.x, 'W')
        striker_y = pixel2xy(pos_striker.y, 'H')
        puck_x = pixel2xy(pos_puck.x, 'W')
        puck_y = pixel2xy(pos_puck.y, 'H')
        target_x = pixel2xy(pos_target.x, 'W')
        target_y = pixel2xy(pos_target.y, 'H')
        # Wall distances
        wall_dist_S = pixel2xy(pos_puck.y, 'H') - pixel2xy(0, 'H') 
        wall_dist_E = pixel2xy(W, 'W') - pixel2xy(pos_puck.x, 'W')
        wall_dist_N = pixel2xy(H, 'H') - pixel2xy(pos_puck.y, 'H')
        wall_dist_W = pixel2xy(pos_puck.x, 'W') - pixel2xy(0, 'W') 
        # Get observation
        state = np.array([
                    # position data - striker 
                    striker_x, striker_y, self.striker.angle,
                    # position data - puck 
                    puck_x, puck_y,
                    # velocity data - striker
                    vel_striker.x,
                    vel_striker.y,
                    avel_striker,
                    # position data - puck 
                    vel_puck.x,
                    vel_puck.y,
                    # distance error
                    wall_dist_S,
                    wall_dist_E,
                    wall_dist_N,
                    wall_dist_W,
                    ], dtype=np.float32)
        assert len(state)==self.dim_observation
        return state
