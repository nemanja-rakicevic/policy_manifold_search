
"""
Author:         Anonymous
Description:
                Data Generation manager
"""


import behaviour_representations.exploration.data_generate as gdata
import behaviour_representations.tasks.task_control as ctrl


class DataGenManager(object):
    """ 
        Select which data generator framework to use, 
        based on the provided task and controller type.
    """

    def __new__(self, **taskargs_dict):      
        experiment_dict=taskargs_dict['experiment']
        seed_task=taskargs_dict['seed_task']

        # Select experiment type and return data generator
        if experiment_dict['type'] == 'displacement':
            task_object = ctrl.Experiment(seed_task=seed_task,
                                          **experiment_dict) 
            return gdata.GenerateDisplacement(task_object=task_object, 
                                              **taskargs_dict)
            
        elif experiment_dict['type'] == 'nn_policy':
            task_object = ctrl.Experiment(seed_task=seed_task,
                                          **experiment_dict) 
            return gdata.GenerateDisplacement(task_object=task_object, 
                                              **taskargs_dict)
        else:
            raise ValueError("Experiment type '{}' "
                             "is not defined!".format(experiment_dict['type']))


