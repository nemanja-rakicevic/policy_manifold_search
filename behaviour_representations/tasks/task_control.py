
"""
Author:         Anonymous
Description:
                Classes for creating and running experiments

                - Experiment: Main class handles interfacing simulated 
                              environement and using policies. 
                              Both sequential and parallelized versions.

                - Simulator:  Wrapper for running a trial (environment episode)

                - Environment: Wrapper over the environment to keep track of 
                               additional statistics

                - DisplacementController: Simple displacement-based control of 
                                          the agent.
                - NNPolicyController: NN policy based controller
                - CNNPolicyController: CNN policy based controller
"""

import os
import gym
import logging
gym.logger.set_level(40)
import numpy as np
import multiprocessing as mpi
import tensorflow as tf 
tf.logging.set_verbosity(tf.logging.ERROR)
# import keras
# from keras import layers as ly
layer = tf.keras.layers

from operator import itemgetter 
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import behaviour_representations.envs
import behaviour_representations.utils.behaviour_metrics as bmet
import behaviour_representations.tasks.policy_archs as parch

from behaviour_representations.utils.utils import timing
from behaviour_representations.utils.utils import _SEED

logger = logging.getLogger(__name__)



_MAX_CPU = 32


###############################################################################
###############################################################################
###############################################################################
### EXPERIMENT - TOP LEVEL


class Experiment():

    def __init__(self, environment, controller, seed_task=100,
                       # run_distributed=True, 
                       run_distributed=False, 
                       **kwargs):
        self.run_distributed = run_distributed
        # Load Environment
        self.environment = Environment(environment['id'], **kwargs)
        self.traj_dim = self.environment.traj_dim
        self.num_outcomes = self.environment.num_outcomes
        self.param_ranges = self.environment.param_ranges
        self.env_name = environment['id']
        self.ctrl_name = controller['type']
        # Load Policy
        if 'displacement' in controller['type']:
            Controller = DisplacementController
        elif 'nn_policy' in controller['type']:
            Controller = NNPolicyController
        elif 'cnn_policy' in controller['type']:
            Controller = CNNPolicyController
        else:
            raise ValueError("Undefined controller type: {}".format(
                                                         controller['type']))
        # Set up Experiment
        if self.run_distributed:
            # Initialise the processes
            def proc(in_queue, out_queue):
                controller_obj = Controller(environment_obj=self.environment, 
                                            controller=controller, **kwargs)
                sim_instance = Simulator(environment_obj=self.environment, 
                                         controller_obj=controller_obj,
                                         seed_task=seed_task)
                while True:
                    i, data = in_queue.get()
                    result = sim_instance.run_episode(data)
                    out_queue.put((i, result))    

            self.in_queue = mpi.Queue()
            self.out_queue = mpi.Queue()
            queues = (self.in_queue, self.out_queue)

            # for _ in range(mpi.cpu_count()-2):
            for _ in range(min(_MAX_CPU, mpi.cpu_count()-2)):
                process = mpi.Process(target=proc, args=queues)
                process.daemon = True
                process.start()

            # TODO: Make this better one instance to get parameters
            self.controller_obj = Controller(environment_obj=self.environment, 
                                             controller=controller, **kwargs)
        else:
            # Initialise one instance
            self.controller_obj = Controller(environment_obj=self.environment, 
                                             controller=controller, **kwargs)
            self.sim_instance = Simulator(environment_obj=self.environment, 
                                          controller_obj=self.controller_obj,
                                          seed_task=seed_task)

    # @timing
    def evaluate_population(self, param_population):
        """ 
            Takes a list of policy NN weights and evaluates these
            policies in the environment.
        """
        # if len(param_population) == 1 \
        #         or isinstance(param_population[0], float): 
        #     param_population = [param_population]

        if self.run_distributed == True:
            # enqueue params to execute
            for i, p in enumerate(param_population):
                self.in_queue.put((i, p))
            # collect the results
            results = [None] * len(param_population)
            for _ in range(len(param_population)):
                i, r = self.out_queue.get()
                results[i] = r
        else:
            # execute sequentially and collect the results
            results = [self.sim_instance.run_episode(p) \
                          for p in param_population]
        # Unpack results
        outcomes, metric_bd, traj_main, traj_aux = \
            list(zip(*results))
        # Quickfix
        if len(traj_main) == 1:
            traj_main = np.array(traj_main)
            traj_aux = np.array(traj_aux)
        else:
            traj_main = list(traj_main)
            traj_aux = list(traj_aux)

        return dict(
                    # param_original=np.array(param_population, dtype=np.float32), 
                    param_original=param_population, 
                    outcomes=np.array(outcomes), 
                    metric_bd=np.array(metric_bd), 
                    traj_main=traj_main, 
                    traj_aux=traj_aux)


    def balance_outcomes(self, param_population, outcome_list,
                         metric_bd_list,
                         traj_ball, traj_striker):
        """ Balance the outcome categories for better model fitting """
        # Get number of different successful and failed outcomes
        idx_failed = np.where(outcome_list==-1)[0]
        num_failed = len(idx_failed)
        idx_list = [np.where(outcome_list==t)[0] \
                   for t in range(self.num_targets)]
        num_list = [len(i) for i in idx_list]
        # Balance the multiple successful outcomes categories equally (+-_SUCC_DIFF%)
        if len(idx_list)>1:
            _SUCC_DIFF = 0.1
            _fs_ratio = 0.4  # 0.4 --> f:s=40:60
            cutoff = int(np.ceil(_SUCC_DIFF * (max(num_list) - min(num_list)))
                         + min(num_list))
            idx_success = np.empty(0, int)
            for i,n in enumerate(num_list):
                if n>0: idx_success = \
                            np.append(idx_success, 
                                      np.random.choice(idx_list[i], 
                                                       size=min(n, cutoff), 
                                                       replace=False))
        else:
            _fs_ratio = 0.55  # 0.5 --> f:s=50:50
            idx_success = idx_list[0]
        # check num_succes
        num_success = len(idx_success)
        assert num_success>0, "There are no successful trials"
        # Balance fail:success ratio using _fs_ratio
        if num_failed > num_success:
            _fs_coeff = _fs_ratio/(1-_fs_ratio) 
            balanced_idx = np.random.choice(idx_failed,
                                            size=int(min([num_failed, 
                                                 _fs_coeff*num_success])),
                                            replace=False)
            balanced_idx = np.append(balanced_idx, idx_success)
        else:
            return param_population, outcome_list, metric_bd_list, \
                   traj_ball, traj_striker
        #     _fs_coeff = (1-_fs_ratio)/_fs_ratio
        #     balanced_idx = np.random.choice(idx_success, size=int(min([num_success, _fs_coeff*num_failed])), replace=False)
        #     balanced_idx = np.append(balanced_idx, idx_failed)
        return param_population[balanced_idx], outcome_list[balanced_idx], \
               metric_bd_list[balanced_idx], \
               list(itemgetter(*balanced_idx)(traj_ball)), \
               list(itemgetter(*balanced_idx)(traj_striker))


###############################################################################
###############################################################################
###############################################################################
### SIMULATOR 


class Simulator(object):
    """ 
        Main simulator class that takes policies (object) and executes them 
        in the environment (object).
    """
    def __init__(self, environment_obj, controller_obj, seed_task):
        # np.random.seed(seed_task)
        self.environment = environment_obj
        self.controller = controller_obj
        self.seed_task = seed_task

    def _disp(self, action):
        """ Start recorder and display """
        if action == 'start':
            if self.environment.disp: self.environment.env.render()
            if self.environment.video_rec_file is not None:
                self.rec = VideoRecorder(self.environment.env, 
                                         path=self.environment.video_rec_file)
        if action == 'show':
            if self.environment.disp: self.environment.env.render()
            if self.environment.video_rec_file is not None: 
                self.rec.capture_frame()
        if action == 'stop':
            if self.environment.disp: self.environment.env.close()
            if self.environment.video_rec_file is not None: self.rec.close()


    def _collect_trajectories(self, step=None, done=None, info_dict=None):
        if step is not None and done is not None:
            if not step % self.environment.ds_rate or done:
                self.traj_main = np.vstack((self.traj_main, 
                                            info_dict['position']))
                self.traj_aux = np.vstack((self.traj_aux, 
                                       info_dict['position_aux']))
        else:
            if len(self.traj_main) > self.environment.env.max_episode_length:
                self.traj_main = \
                    self.traj_main[:self.environment.env.max_episode_length]
                self.traj_aux = \
                    self.traj_aux[:self.environment.env.max_episode_length]


    def run_episode(self, policy_parameters):
        """ Execute one episode in the environment """
        self._disp('start')
        # Initialize agent and statistics
        self.environment.bd_metric.restart()
        self.controller.initialize_parameters(policy_parameters)
        obs, self.traj_main, self.traj_aux = self.environment.env.initialize(
                                        init_params=self.controller.get_init(), 
                                        seed_task=self.seed_task)
        done = False
        step = 0
        rew_list = []
        while not done and step<self.environment.max_episode_steps:
            self._disp('show')
            # get action and check striker step limit
            action = self.controller.get_action(obs)
            # run simulation step
            obs, rew, done, info_dict = self.environment.env.step(action) 
            # additinal info
            rew_list.append(rew)
            self.environment.bd_metric.update(info_dict=info_dict)
            self._collect_trajectories(step, done, info_dict)
            step += 1
        # Stop recorder and display
        self._disp('stop')
        # Return trial statistics
        self._collect_trajectories()
        trial_outcome = self.environment.env.finalize(state=obs, 
                                                      rew_list=rew_list,
                                                      traj=self.traj_main,
                                                      traj_aux=self.traj_aux)
        trial_metric = self.environment.bd_metric.calculate(
                                                      traj=self.traj_main,
                                                      traj_aux=self.traj_aux)
        return trial_outcome, trial_metric, \
                    self.traj_main.astype(np.float16), \
                    self.traj_aux.astype(np.float64)



    def run_replay_episode(self, episode_actions):
        """ Execute one episode in the environment """
        asize = self.environment.env.action_space.shape[0]
        self._disp('start')
        # Initialize agent and statistics
        self.environment.bd_metric.restart()
        # self.controller.initialize_parameters(policy_parameters)
        obs, self.traj_main, self.traj_aux = self.environment.env.initialize(
                                        init_params=None, 
                                        seed_task=self.seed_task)
        done = False
        step = 0
        rew_list = []
        while not done and step<self.environment.max_episode_steps:
        # while not done and step<min(self.environment.max_episode_steps, episode_actions.shape[0]-1):
            self._disp('show')
            # get action and check striker step limit
            try:
                action = episode_actions[step+1, -asize:]
            except:
                action = np.zeros(asize)
            # run simulation step
            obs, rew, done, info_dict = self.environment.env.step(action) 
            # additinal info
            rew_list.append(rew)
            self.environment.bd_metric.update(info_dict=info_dict)
            self._collect_trajectories(step, info_dict)
            step += 1
        # Stop recorder and display
        self._disp('stop')
        # Return trial statistics
        self._collect_trajectories()
        trial_outcome = self.environment.env.finalize(state=obs, 
                                                      rew_list=rew_list,
                                                      traj=self.traj_main,
                                                      traj_aux=self.traj_aux)
        trial_metric = self.environment.bd_metric.calculate(
                                                      traj=self.traj_main,
                                                      traj_aux=self.traj_aux)

        return dict(
                    # param_original=np.array(param_population, dtype=np.float32), 
                    param_original=None, 
                    outcomes=trial_outcome, 
                    metric_bd=trial_metric,
                    traj_main=[self.traj_main], 
                    traj_aux=[self.traj_aux])

###############################################################################
###############################################################################
###############################################################################
### TASK 


class Environment(object):
    """
        Main Environment initialisation class.
        Creates an Environment interface and Task parameters 
        associated with the environment.
    """

    def __init__(self, env_name, metric,
                       disp=False, video_rec_file=None, 
                       **kwargs):
        self.disp = disp
        self.video_rec_file = video_rec_file
        # # Load env by name
        self.env_name = env_name
        env_class = ''.join([c.capitalize() for c in env_name.split('_')])
        self.env = gym.make(env_class+'Env-v0').unwrapped
        # Env setup
        self.num_bins = metric['dim']
        if 'quadruped_kicker'==env_name: 
            self.ds_rate = 100
        elif 'quadruped_walker'==env_name: 
            self.ds_rate = 6
        elif 'bipedal_kicker'==env_name: 
            self.ds_rate = 2
        else:
            self.ds_rate = 20
        self.max_episode_steps = self.env.spec.max_episode_steps
        self.env.max_episode_length = self.max_episode_steps//self.ds_rate
        self.traj_dim = (self.env.max_episode_length, 2)
        # Extract info from environment
        self.env_info = self.env.env_info
        self.param_ranges = self.env.param_ranges
        self.num_outcomes = self.env.env_info['num_targets'] + 1
        self.dim_obs = self.env.observation_space.shape[0]
        self.dim_action = self.env.action_space.shape[0]
        # Initialise behaviour descriptor metric
        bd_class = ''.join([s.capitalize() for s in metric['type'].split('_')])
        self.bd_metric = bmet.__dict__[env_class+bd_class](**metric, 
                                                         **self.env_info)



###############################################################################
###############################################################################
###############################################################################
### CONTROLLERS


class DisplacementController(object):
    """
        Controller whose parameters define the striker's initial position,
        the final position, and speed at which to move.
    """
    def __init__(self, environment_obj, **kwargs):
        self.environment = environment_obj
        self.param_ranges = self.environment.param_ranges
        self.parameter_dims = np.array([2*len(self.param_ranges)+1])
        self.parameter_arch = None
        self.flag_fail = False


    def initialize_parameters(self, parameters):
        self.flag_fail = False
        self.parameters = parameters
        # if parameters are crazy just stop the experiment
        check_ranges = self.environment.env_info['ball_ranges']
        range_high = np.append(check_ranges[:,1], 3.14)
        range_low = np.append(check_ranges[:,0], -3.14)
        # range_low[1] = -0.15
        # Check if parameters out of range
        if (self.parameters[:-1]<np.tile(range_low, 2)).any() or \
           (self.parameters[:-1]>np.tile(range_high, 2)).any():
            self.flag_fail = True
        # check if striker on top of the puck
        if np.linalg.norm(self.parameters[:2])<0.056:
            self.flag_fail = True


    def get_init(self):
        # Starting pose of the agent
        if self.flag_fail:
            return np.zeros(len(self.param_ranges))
        return self.parameters[:len(self.param_ranges)]


    def get_action(self, current_obs):
        if self.flag_fail:
            return np.zeros(self.environment.dim_action)
        _POSE_THRSH = 0.05
        # desired end-pose and speed of the executed movement
        p_pose_end   = self.parameters[len(self.param_ranges):-1]
        p_speed_coef = self.parameters[-1] #* 10.
        # move brick until it reaches the final pose
        desired_vec = p_pose_end - current_obs[:len(self.param_ranges)]

        # print("--- error: ", desired_vec)

        if np.linalg.norm(desired_vec) <= _POSE_THRSH:
            velocity_action = desired_vec * p_speed_coef
        else:
            desired_vec = (desired_vec>_POSE_THRSH).astype(int)
            velocity_action = np.sign(desired_vec) * p_speed_coef
        return velocity_action


    def convert_params(self, uniforms=None, params=None):
        """ Convert the parameter range to uniform and vice versa """           
        _displ_ranges = np.concatenate((self.environment.param_ranges, 
                                        self.environment.param_ranges, 
                                        [[0, 1]]), axis=0)
        # _uniform_ranges = np.array([np.min(uniforms), np.max(uniforms)])
        _uniform_ranges = np.array([0, 1])

        _range_uniform = _uniform_ranges[1] - _uniform_ranges[0]
        _range_param = _displ_ranges[:,1] - _displ_ranges[:,0]

        if uniforms is not None:
            uni_conv = uniforms[:,:len(_displ_ranges)] - _uniform_ranges[0]
            return ((uni_conv*_range_param)/_range_uniform)+_displ_ranges[:,0] 
        elif params is not None:
            param_conv = params[:,:len(_displ_ranges)] - _displ_ranges[:,0]
            return ((param_conv*_range_uniform)/_range_param)+0
        else:
            raise ValueError("No parameters passed.")



class NNPolicyController(object):
    """
        Controller whose parameters define the weights of a neural network,
        that takes the current observation as input and returns an action.
    """

    def __init__(self, environment_obj, controller, use_bias=True, **kwargs):
        self.environment = environment_obj
        self.param_ranges = self.environment.param_ranges
        self.use_bias = use_bias
        self.policy_arch = controller['architecture']
        self.policy_type = controller['type']

        self.parameter_arch = self._get_parameter_arch()
        self.parameter_dims = self._get_parameter_dims(self.parameter_arch)

        # Start tf stuff
        tf.set_random_seed(_SEED)
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True
#         config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess_p = tf.Session(config=config)
        self._build_policy_graph()
        self.sess_p.run(tf.global_variables_initializer())
        # default_graph = tf.get_default_graph()
        # print("====INIT 1", tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
        #                              scope='policy_net'))

        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                     scope='policy_net')
        self.sess_p.graph.finalize()
    

    def get_init(self):
        # Starting pose of the agent
        init_params = np.mean(self.param_ranges, axis=1)
        init_params[1] = -0.3 #np.min(self.param_ranges[1])
        return init_params


    def _get_parameter_arch(self):
        parameter_arch = []
        parameter_arch.append((self.environment.dim_obs + self.use_bias,
                              self.policy_arch[0]))
        for i in range(len(self.policy_arch)-1):
            parameter_arch.append((self.policy_arch[i] + self.use_bias, 
                                  self.policy_arch[i+1]))
        parameter_arch.append((self.policy_arch[-1] + self.use_bias,
                              self.environment.dim_action))
        return parameter_arch


    def _get_parameter_dims(self, arch):
        return np.append(np.max(np.array(arch), axis=0), len(arch))


    def _build_policy_graph(self):
        """ Input placeholder and policy network definition """
        self.S = layer.Input(shape=(self.environment.dim_obs,), 
                              name="policy_input")  # , dtype='float64')
        self.policy_out, self.h_out = self._policy_net(self.S)



    def _assign_weight_parameters(self, params):
        """ Overwrites random networm weights with the given parameters """
        # check if params match policy_arch !@!

        # param_weights = params.reshape(self.policy_arch)

        for i, w in enumerate(params):
            if self.use_bias:
                self.var_list[2*i].load(w[:-1,:], self.sess_p)
                self.var_list[2*i+1].load(w[-1,:], self.sess_p)
            else:
                self.var_list[i].load(w, self.sess_p)


    def _policy_net(self, s_input):
        """ Policy NN with non trainable weights from the parameters """
        action_out, h_out = parch.__dict__[self.policy_type](
                                    state_input=s_input, 
                                    policy_arch=self.policy_arch, 
                                    dim_action=self.environment.dim_action)
        return action_out, h_out
    

    def initialize_parameters(self, parameters):
        # convert weight parameters to proper dimension
        sliced_params = []
        for i, pa in enumerate(self.parameter_arch):
            sliced_params.append(parameters[:pa[0], :pa[1], i])
            
        # assign weight parameters
        self._assign_weight_parameters(sliced_params)


    def get_init_pose(self):
        # Constant starting position of the striker
        return np.zeros(len(self.environment.param_ranges))


    def get_action(self, current_obs):
        current_obs = current_obs.reshape(-1, self.environment.dim_obs)
        action, h_out = self.sess_p.run([self.policy_out, self.h_out], 
                                       feed_dict={self.S: current_obs})
        # action = action * self.ctrl_ranges
        if np.isnan(action).any():
            raise ValueError("NN outputs are NaNs, not trained well!")
        return action.flatten()


    def convert_params(self, uniforms=None, params=None):
        if uniforms is not None:
            return uniforms
        elif params is not None:
            return uniforms
        else:
            raise ValueError("No parameters passed.")

