
"""
Author:         Anonymous
Description:
                Classes for calculating the behaviour descriptors
                Environments:
                - Striker
                    - contact_grid
                - Biped
                    - gait_grid
                    - simple_grid
"""

import numpy as np
import multiprocessing as mpi

from functools import partial
from itertools import combinations

from scipy.spatial.distance import euclidean


class BehaviourMetric(object):
    """ 
        Behaviour Metric base class, for basic interactions
    """

    @property
    def metric_size(self):
        return np.prod(self.metric_dim)


    def calculate(self, **kwargs):
        return self._get_metric_vect(**kwargs)


    def restart(self, **kwargs):
        raise NotImplementedError

    def update(self, **kwargs):
        raise NotImplementedError


###############################################################################
# WALKER ENVIRONMENT


class BipedalWalkerAugmentedGaitGrid(object):
    """ 
        Behaviour descriptor based on gait type.
        
        Returns:
            - D0 : average hull y-axis height 
            - D1 : final hull x-axis bin
            - D2 : left leg duty factor
            - D3 : right leg duty factor
    """

    Y_LOW = 4.5  # 3.5
    Y_HIGH = 6.2 # 7
    REWARD_THRSH = 20
    TERRAIN_LENGTH = 14/30*200
    HLEN = TERRAIN_LENGTH/2


    def __init__(self, dim=10, **kwargs):
        self.dim = dim
        self.dim_05 = self.dim//2
        self.dim_x = self.dim**2
        self.num_bins = self.dim_05**3 * self.dim_x
        self.restart()


    def _get_metric_vect(self, traj, traj_aux):
        """ Compile the metric vector based on experiment data """
        nstep = len(traj)
        
        # Descriptor based on most frequent hull height
        dist_y = np.clip(np.mean(traj[:, 1]), self.Y_LOW, self.Y_HIGH)
        dist_y = (dist_y - self.Y_LOW) / (self.Y_HIGH-self.Y_LOW)
        # d0_idx = int(dist_y * (self.dim_05 - 1))
        d0_idx = int(np.clip(dist_y * self.dim_05, 0, self.dim_05-1))

        # Descriptor based length traversed
        dist_x = traj[-1, 0] / self.TERRAIN_LENGTH
        d1_idx = int(np.clip(dist_x * self.dim_x, 0, self.dim_x-1))

        # Descriptor based on leg contacts
        contacts = traj_aux[:, :2]
        # d2_idx, d3_idx = (np.mean(contacts, axis=0) * (self.dim-1)).astype(int)
        d2_idx, d3_idx = np.clip(np.mean(contacts, axis=0) * self.dim, 
                                 0, self.dim-1).astype(int)

        # Calculate final index
        idx = d0_idx * self.dim_05**2 * self.dim_x + \
              d1_idx * self.dim_05**2 + \
              d2_idx * self.dim_05 + \
              d3_idx


        print("\n\n= IDX: {}/{} [d0_idx: {}/5, d1_idx: {}/100 !!! d2_idx: {}/5, d3_idx: {}/5]".format(
            idx, len(self.metric_vect),
            d0_idx, d1_idx, d2_idx, d3_idx))


        assert idx < len(self.metric_vect), \
            "\n === METRIC (idx {}); y_axis: {}; leg_angle: {}; "\
            "leg_duty_factor: left {}, right {}".format(
                idx, d0_idx, d1_idx, d2_idx, d3_idx)

        self.metric_vect[idx] = 1  
        return self.metric_vect


    @property
    def metric_dim(self):
        return [self.dim_05, self.dim_x, self.dim_05, self.dim_05] 


    @property
    def metric_size(self):
        return np.prod(self.metric_dim)


    @property
    def metric_name(self):
        return 'gait_grid'


    def restart(self, **kwargs):
        self.metric_vect = np.zeros(self.num_bins, dtype=np.int16)


    def update(self, **kwargs):
        pass


    def calculate(self, **kwargs):
        return self._get_metric_vect(**kwargs)


###############################################################################
# KICKER ENVIRONMENT


class BipedalKickerAugmentedSimpleGrid(object):
    """ 
        Behaviour descriptor based on the balistic trajectory of the ball.
        
        Returns:
            vector = [max y-axis coordinate,
                      final x-axis coordinate]
                      
    """

    Y_LOW = 4  # 3.5
    Y_HIGH = 7 # 10
    REWARD_THRSH = 20
    TERRAIN_LENGTH = 14/30*200
    HLEN = TERRAIN_LENGTH/2
    X_LIMIT_LEFT = 0 # HLEN
    X_LIMIT_RIGHT = 20 # HLEN


    def __init__(self, dim=10, **kwargs):
        self.dim = dim
        self.dim_y = 5*self.dim # self.dim//2
        self.dim_x = 2*self.dim**2
        self.num_bins = self.dim_y * self.dim_x
        self.restart()


    def _get_metric_vect(self, traj, traj_aux):
        """ Compile the metric vector based on experiment data """
        nstep = len(traj)

        # Descriptor based on most frequent hull height
        dist_y = np.clip(np.max(traj[:, 1]), self.Y_LOW, self.Y_HIGH)
        dist_y = (dist_y - self.Y_LOW) / (self.Y_HIGH-self.Y_LOW)
        d0_idx = int(np.clip(dist_y * self.dim_y, 0, self.dim_y-1))

        # Descriptor based length traversed
        dist_x = np.clip(traj[-1, 0], self.HLEN - self.X_LIMIT_LEFT, 
                                      self.HLEN + self.X_LIMIT_RIGHT)
        dist_x = (dist_x - (self.HLEN - self.X_LIMIT_LEFT)) \
                            / (self.X_LIMIT_LEFT + self.X_LIMIT_RIGHT)
        d1_idx = int(np.clip(dist_x * self.dim_x, 0, self.dim_x-1))

        # Calculate final index
        idx = d0_idx * self.dim_x + d1_idx

        assert idx < len(self.metric_vect), \
            "\n === METRIC (idx {}); y_axis: {}; leg_angle: {}; "\
            "leg_duty_factor: left {}, right {}".format(
                idx, d0_idx, d1_idx, d2_idx, d3_idx)

        self.metric_vect[idx] = 1  
        return self.metric_vect


    @property
    def metric_dim(self):
        return [self.dim_y, self.dim_x] 


    @property
    def metric_size(self):
        return np.prod(self.metric_dim)


    @property
    def metric_name(self):
        return 'simple_grid'


    def restart(self, **kwargs):
        self.metric_vect = np.zeros(self.num_bins, dtype=np.int16)


    def update(self, **kwargs):
        pass


    def calculate(self, **kwargs):
        return self._get_metric_vect(**kwargs)


###############################################################################
### STRIKER ENVIRONMENT


class StrikerAugmentedContactGrid(object):
    """ 
        Behaviour descriptor based on basic wall contacts and final resting
        position of the puck.
        
        Returns:
            vector = [1 (no contact)
                      4 x single contact walls
                      4 x 3 double contact walls]
    """

    def __init__(self, wall_geoms, ball_ranges, ball_geom, dim, **kwargs):
        self.wall_geoms = wall_geoms
        self.ball_geom = ball_geom
        self.num_bins = dim
        self.num_gridcells = dim**2
        self.num_walls = len(self.wall_geoms)
        self.last_geoms = []
        self.min_x = ball_ranges[0][0]
        self.min_y = ball_ranges[1][0]
        self.range_x = ball_ranges[0][1] - self.min_x
        self.range_y = ball_ranges[1][1] - self.min_y
        self.rem_walls = {0: [1,2,3], 1: [0,2,3], 2: [0,1,3], 3: [0,1,2]}
        # Storing 1 collision in first part, and for 2 collisions only bins of 
        # the second collision and wall of first.
        self.restart()


    def _update_bins(self, geom):
        _wall = geom-1
        if _wall not in [b['wall'] for b in self.bin_history]:
            self.bin_history.append({'wall': _wall})


    def _get_metric_vect(self, traj, **kwargs):
        # Get grid index based on contacts
        if len(self.bin_history) == 0: 
            grid_idx = 0
        elif len(self.bin_history) == 1:  
            grid_idx = (1 + self.bin_history[0]['wall'])
        else:
            wall0 = self.bin_history[0]['wall']
            wall1_raw = self.bin_history[1]['wall']
            wall1 = np.where(np.array(self.rem_walls[wall0])==wall1_raw)[0][0]
            grid_idx = (1 + self.num_walls) + wall0*(self.num_walls-1) + wall1

        # get ball end position
        last_x, last_y = traj[-1]
        # px_idx = int((last_x-self.min_x)/self.range_x*(self.num_bins-1))
        px_idx = (last_x - self.min_x) / self.range_x
        px_idx = int(np.clip(px_idx * self.num_bins, 0, self.num_bins-1))
        # py_idx = int((last_y-self.min_y)/self.range_y*(self.num_bins-1))
        py_idx = (last_y - self.min_y) / self.range_y
        py_idx = int(np.clip(py_idx * self.num_bins, 0, self.num_bins-1))
        # get which grid cell it is in 
        idx = grid_idx*self.num_gridcells + py_idx*self.num_bins + px_idx
        
        assert idx < len(self.contact_grids), \
            "\n === METRIC idx: {}; grid: {}; xy: {},{}".format(
                  idx, grid_idx, px_idx, py_idx)

        self.contact_grids[idx] = 1
        return self.contact_grids


    @property
    def metric_dim(self):
        return [17, self.num_bins, self.num_bins] 


    @property
    def metric_size(self):
        return np.prod(self.metric_dim)


    @property
    def metric_name(self):
        return 'contact_grid'


    def restart(self, **kwargs):
        self.bin_history = []
        self.contact_grids = np.zeros((self.num_walls+1)*self.num_gridcells +\
                                      self.num_walls*(self.num_walls-1) *\
                                      self.num_gridcells, dtype=np.int16)


    def update(self, info_dict, **kwargs):
        if 'sim_data' in info_dict.keys():
            sim_data = info_dict['sim_data']
            for c in range(sim_data.ncon):
                contact = sim_data.contact[c]
                if contact.geom1 == self.ball_geom or \
                   contact.geom2 == self.ball_geom:
                    # Only for ball-wall contacts
                    if contact.geom1 in self.wall_geoms and \
                        not contact.geom1 in self.last_geoms:
                        self._update_bins(contact.geom1)
                    if contact.geom2 in self.wall_geoms and \
                        not contact.geom2 in self.last_geoms:
                        self._update_bins(contact.geom2)
                    self.last_geoms = [contact.geom1, contact.geom2]
        elif 'contact_objects' in info_dict.keys():
            [self._update_bins(wc) for wc in info_dict['contact_objects']]
        else:
            raise ValueError('Wrong info_dict passed!')


    def calculate(self, **kwargs):
        return self._get_metric_vect(**kwargs)


###############################################################################
### PANDA STRIKER ENVIRONMENT


class PandaStrikerContactGrid(StrikerAugmentedContactGrid):
    pass
