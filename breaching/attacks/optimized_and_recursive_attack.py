
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

class Optimization_and_recursive_attacker(_BaseAttacker):

    """Implements a wide spectrum of optimization-based attacks."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)


        if isinstance(model, VisionContainer):
            model = model.model
        self.nbsplit = self._model_split(model)
        self.model_cnn =  torch.nn.Sequential(*list(model.children())[:self.nbsplit])
        self.model_fc =  torch.nn.Sequential(*list(model.children())[self.nbsplit:])

        self.recursive_attacker = RecursiveAttacker(self.model_fc, loss_fn, cfg_attack, setup)
        self.optimization_attacker = OptimizationBasedAttacker(self.model_cnn, loss_fn, cfg_attack, setup)

    def _model_split(self, model):
        
        for idx, layer in enumerate(model.children()):
            if isinstance(layer, torch.nn.modules.linear.Linear):
                return idx

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):
        list_paramter = list(self.model_cnn.parameters())
        cnn_params_len = len(list_paramter)
        server_payload_cnn, server_payload_fc = copy.deepcopy(server_payload), copy.deepcopy(server_payload)
        server_payload_cnn['parameters'] = server_payload_cnn['parameters'][:cnn_params_len]
        server_payload_fc['parameters'] = server_payload_fc['parameters'][cnn_params_len:]
        
        server_payload_fc['metadata']['shape'] = (server_payload_fc['parameters'][0].t().shape[0], )

        shared_data_cnn, shared_data_fc = copy.deepcopy(shared_data), copy.deepcopy(shared_data)
        shared_data_cnn['gradients'] = shared_data_cnn['gradients'][:cnn_params_len]
        shared_data_fc['gradients'] = shared_data_fc['gradients'][cnn_params_len:]

        fc_reconstructed_data, fc_stats = self.recursive_attacker.reconstruct([server_payload_fc], [shared_data_fc], {}, dryrun=dryrun)
        reconstructed_data, stats = self.optimization_attacker.reconstruct([server_payload_cnn], [shared_data_cnn], {}, dryrun=dryrun, activation=fc_reconstructed_data['data'])

        return reconstructed_data, stats

    def __repr__(self):
        return self.optimization_attacker.__repr__() + self.recursive_attacker.__repr__()