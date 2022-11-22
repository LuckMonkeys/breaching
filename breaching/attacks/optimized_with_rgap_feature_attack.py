from tracemalloc import start
from joblib import parallel_backend
import torch
import time

from .base_attack import _BaseAttacker
from .auxiliaries.regularizers import regularizer_lookup, TotalVariation
from .auxiliaries.objectives import Euclidean, CosineSimilarity, objective_lookup
from .auxiliaries.augmentations import augmentation_lookup
from .optimization_based_attack import OptimizationBasedAttacker
from .recursive_attack import RecursiveAttacker
from ..cases.models.model_preparation import VisionContainer
import copy



'''

1. recover the activation value through grap
2. add objective calculate function
    a. define objective name in .yaml file
    b. add the l2/MSE/Cos loss item to gradient matching loss
    c. calculate the gradient of dummpy data x
    d. update dummp data


'''


class Optimization_with_grap_feature_attacker(_BaseAttacker):

    """Implements a wide spectrum of optimization-based attacks."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)


        if isinstance(model, VisionContainer):
            self.model = model.model
        else:
            self.model = model

        self.nbsplit = self._model_split(self.model)
        self.model_cnn =  torch.nn.Sequential(*list(self.model.children())[:self.nbsplit])
        self.model_fc =  torch.nn.Sequential(*list(self.model.children())[self.nbsplit:])

        self.recursive_attacker = RecursiveAttacker(self.model_fc, loss_fn, cfg_attack, setup)
        self.optimization_attacker = OptimizationBasedAttacker(self.model, loss_fn, cfg_attack, setup)

    def _model_split(self, model):
        
        for idx, layer in enumerate(model.children()):
            if isinstance(layer, torch.nn.modules.linear.Linear):
                return idx

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):
        list_paramter = list(self.model_cnn.parameters())
        cnn_params_len = len(list_paramter)

        if isinstance(server_payload, list):
            server_payload = server_payload[0]

        server_payload_fc = copy.deepcopy(server_payload)
        print('cnn_param_len: ', cnn_params_len)
        print('server_payload_fc: ', server_payload_fc)

        # server_payload_cnn['parameters'] = server_payload_cnn['parameters'][:cnn_params_len]
        server_payload_fc['parameters'] = server_payload_fc['parameters'][cnn_params_len:]
        
        server_payload_fc['metadata']['shape'] = (server_payload_fc['parameters'][0].t().shape[0], )

        shared_data_fc =  copy.deepcopy(shared_data)
        shared_data_fc['gradients'] = shared_data_fc['gradients'][cnn_params_len:]
        # shared_data_cnn['gradients'] = shared_data_cnn['gradients'][:cnn_params_len]

        fc_reconstructed_data, fc_stats = self.recursive_attacker.reconstruct([server_payload_fc], [shared_data_fc], {}, dryrun=dryrun)

        #TODO add extra constraints in base optimiation class
        reconstructed_data, stats = self.optimization_attacker.reconstruct([server_payload], [shared_data], {}, dryrun=dryrun, constraints=[fc_reconstructed_data['data']])

        return reconstructed_data, stats

    def __repr__(self):
    #     n = "\n"
    #     return f"""Attacker (of type {self.__class__.__name__}) with settings:
    # Hyperparameter Template: {self.cfg.type}

    # Objective: {repr(self.objective)}
    # Regularizers: {(n + ' '*18).join([repr(r) for r in self.regularizers])}
    # Augmentations: {(n + ' '*18).join([repr(r) for r in self.augmentations])}

    # Optimization Setup:
    #     {(n + ' ' * 8).join([f'{key}: {val}' for key, val in self.cfg.optim.items()])}
    #     """
        return self.recursive_attacker.__repr__()