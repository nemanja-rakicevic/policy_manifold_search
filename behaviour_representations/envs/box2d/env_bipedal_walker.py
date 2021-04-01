

import os
import math
import numpy as np

from gym import utils
from gym import spaces
from gym.envs.box2d import BipedalWalker

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

from behaviour_representations.utils.utils import _SEED

### REFERENCE:
# https://github.com/alirezamika/bipedal-es/blob/master/bipedal.py
# https://github.com/openai/gym/blob/52e66f38081548e38711f51d4439d8bcc136d19e/gym/envs/box2d/bipedal_walker.py#L357

REWARD_THRSH = 20
_VEL_THRSH = .0005

# affects how fast-paced the game is, forces should be adjusted as well
FPS    = 50
SCALE  = 30.0   

MOTORS_TORQUE = 80
SPEED_HIP     = 4
SPEED_KNEE    = 6
LIDAR_RANGE   = 160/SCALE

INITIAL_RANDOM = 5

HULL_POLY = [
    (-30,+9), (+6,+9), (+34,+1),
    (+34,-8), (-30,-8)
    ]
LEG_DOWN = -8/SCALE
LEG_W, LEG_H = 8/SCALE, 34/SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = TERRAIN_LENGTH/2    # in steps
FRICTION = 2.5


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.hull in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True
    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False


class BipedalWalkerBaseEnv(BipedalWalker):

    def __init__(self):
        self.init_body = 0
        super().__init__()
        self.param_ranges = np.vstack([self.action_space.low,
                                       self.action_space.high]).T
        self.env_info = dict(
            num_targets=1,
            num_obstacles=0,
            wall_geoms=None,
            ball_geom=None,
            target_info=[{'xy': (20, 0)}],
            striker_ranges=None,
            ball_ranges=None)


    def _get_info_dict(self, obs=None):
        contact_info = obs[np.array([8, 13])] if obs is not None else []
        velocity_info = np.array(self.unwrapped.hull.linearVelocity)
        hip_angle = obs[np.array([4, 9])] if obs is not None else [1, 1]
        rel_angle = np.abs(hip_angle[0] - hip_angle[1])  # 0-2
        # rel_angle = cosine(*leg_angle)  # 0-2
        info_dict = dict(position=np.append(
                            np.array(self.unwrapped.hull.position), 
                            self.unwrapped.hull.angle),
                         position_aux=np.hstack([contact_info,
                                                 rel_angle, 
                                                 velocity_info]),
                         velocity=velocity_info,
                         angle=np.array(self.unwrapped.hull.angle))
        return info_dict


    def initialize(self, seed_task, **kwargs):
        # restart seed
        self.seed(seed_task)
        self.action_space.seed(seed_task)
        # standard reset
        obs = self.reset()
        info_dict = self._get_info_dict(obs)
        self.init_body = info_dict['position'][0]

        return obs, info_dict['position'], info_dict['position_aux']


    def finalize(self, rew_list, **kwargs):
        info_dict = self._get_info_dict()
        reward_len = np.linalg.norm(info_dict['position'][0]-self.init_body)
        outcome = -1  # 0 if reward_len >= REWARD_THRSH else -1
        return np.array([outcome, np.sum(rew_list)])


    def step(self, action):
        obs, rew, done, _ = super().step(action)
        info_dict = self._get_info_dict(obs)
        done = done or np.linalg.norm(info_dict['velocity'])<=_VEL_THRSH
        return obs, rew, done, info_dict

#####

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W/SCALE + self.scroll, 
                               0, VIEWPORT_H/SCALE)
        self.viewer.draw_polygon([
            (self.scroll,                  0),
            (self.scroll+VIEWPORT_W/SCALE, 0),
            (self.scroll+VIEWPORT_W/SCALE, VIEWPORT_H/SCALE),
            (self.scroll,                  VIEWPORT_H/SCALE),
            ], color=(0.9, 0.9, 1.0) )
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll/2: continue
            if x1 > self.scroll/2 + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon([(p[0]+self.scroll/2, p[1]) for p in poly],
                                     color=(1,1,1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render+1) % 100
        i = self.lidar_render
        if i < 2*len(self.lidar):
            l = self.lidar[i] if i < len(self.lidar) \
                              else self.lidar[len(self.lidar)-i-1]
            self.viewer.draw_polyline( [l.p1, l.p2], color=(1,0,0), linewidth=1)

        # Left edge flag
        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*0
        self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], 
                                  color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        # Right edge flag
        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*(TERRAIN_LENGTH-1)
        self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], 
                                  color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        # Start flag
        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*TERRAIN_STARTPAD
        self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], 
                                  color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(1.,1.,1.) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        # Rest of the env
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, 
                                            color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, 
                                            color=obj.color2, filled=False, 
                                            linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, 
                                              color=obj.color2, linewidth=2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def reset(self):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        # init_x = TERRAIN_STARTPAD/4
        init_x = TERRAIN_STEP*TERRAIN_STARTPAD
        init_y = TERRAIN_HEIGHT+2*LEG_H
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[(x/SCALE,y/SCALE) \
                                                    for x,y in HULL_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0020,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy
                )
        self.hull.color1 = (0.5,0.4,0.9)
        self.hull.color2 = (0.3,0.3,0.5)
        self.hull.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)

        self.legs = []
        self.joints = []
        for i in [-1,+1]:
            leg = self.world.CreateDynamicBody(
                position = (init_x, init_y - LEG_H/2 - LEG_DOWN),
                angle = (i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            leg.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = i,
                lowerAngle = -0.8,
                upperAngle = 1.1,
                )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position = (init_x, init_y - LEG_H*3/2 - LEG_DOWN),
                angle = (i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(0.8*LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            lower.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            lower.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H/2),
                localAnchorB=(0, LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = 1,
                lowerAngle = -1.6,
                upperAngle = -0.1,
                )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return 1
                self.p2 = point
                self.fraction = fraction
                return 0
        self.lidar = [LidarCallback() for _ in range(10)]

        return self.step(np.array([0,0,0,0]))[0]



class BipedalWalkerAugmentedEnv(BipedalWalkerBaseEnv):

    def __init__(self):
        self.init_body = 0
        super().__init__()
        high = np.array([np.inf] * 26)
        self.action_space = spaces.Box(np.array([-1, -1, -1, -1]), 
                                       np.array([1, 1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.param_ranges = np.vstack([self.action_space.low,
                                       self.action_space.high]).T
        self.env_info = dict(
            num_targets=1,
            num_obstacles=0,
            wall_geoms=None,
            ball_geom=None,
            target_info=[{'xy': (20, 0)}],
            striker_ranges=None,
            ball_ranges=None)

    def step(self, action):
        obs, rew, done, _ = super().step(action)
        # Augment observation with robot's absolute x,y position
        obs = np.concatenate([obs, 
                              self.hull.position])
        info_dict = self._get_info_dict(obs)
        done = done or np.linalg.norm(info_dict['velocity'])<=_VEL_THRSH
        return obs, rew, done, info_dict
