
"""
Author:         Anonymous
Description:
                Obtain the final AE models for training by combining 
                components from ae_archs.py and ae_trianing.py
                
                Variants:
                - PCA training
                - AE training 
                    (param arch: fc, cnn)
"""

import behaviour_representations.training.ae_archs as arch
from behaviour_representations.training.ae_training import TrainPCA, TrainAE


class TrainingManager(object):
    """ 
        Select which parameter embedding model to use.
    """

    def __new__(self, data_object, seed_model, **taskargs_dict):

        ae_type_pca = taskargs_dict['training']['pca_param']
        ae_type_param = taskargs_dict['training']['ae_param']
        ae_type_traj = taskargs_dict['training']['ae_traj']

        # Select and pass the model object
        if ae_type_pca is not None:
            return TrainPCA(data_object=data_object, seed_model=seed_model, 
                            **taskargs_dict['training'])
        elif ae_type_param is not None: 
            ae_type = ae_type_param['type']      
            param_ae_fns = {
                'param_encoder_fn': 
                    arch.__dict__['{}_param_encoder'.format(ae_type)],
                'param_decoder_fn': 
                    arch.__dict__['{}_param_decoder'.format(ae_type)]}    
            return TrainAE(data_object=data_object, seed_model=seed_model, 
                               **param_ae_fns, **taskargs_dict['training'])
        else:
            return None
