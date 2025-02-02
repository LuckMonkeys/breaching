"""Implementation for basic gradient inversion attacks.

This covers optimization-based reconstruction attacks as in Wang et al. "Beyond Infer-
ring Class Representatives: User-Level Privacy Leakage From Federated Learning."
and convers subsequent developments such as
* Zhu et al., "Deep Leakage from gradients",
* Geiping et al., "Inverting Gradients - How easy is it to break privacy in FL"
* ?
"""

from audioop import avgpp
import imp
from re import I
from tracemalloc import start
import torch
import time
import numpy as np
import torchvision

from .accelarate.denoise.real_denoise import mirnet_denoise, sucnet_denoise 
from .accelarate.super_resolution import swinir_super_resolution
from .base_attack import _BaseAttacker
from .auxiliaries.regularizers import regularizer_lookup, TotalVariation
from .auxiliaries.objectives import Euclidean, CosineSimilarity, objective_lookup
from .auxiliaries.augmentations import augmentation_lookup

import logging

log = logging.getLogger(__name__)
import os
def cosine_sim_layer(gradient):

    def cal_cosine(a, b):
        scalar_product = (a * b).sum() 
        rec_norm = a.pow(2).sum()
        data_norm = b.pow(2).sum()

        objective = 1 - scalar_product / ((rec_norm.sqrt() * data_norm.sqrt()) + 1e-6)
        
        return objective

    


    cos_0_1 = cal_cosine(gradient[:, 0, :, :],gradient[:, 1, :, :] )
    cos_0_2 = cal_cosine(gradient[:, 0, :, :],gradient[:, 2, :, :] )
    cos_1_2 = cal_cosine(gradient[:, 1, :, :],gradient[:, 2, :, :] )

    # print(cos_0_1, cos_0_2, cos_1_2)

    
    return (cos_0_1 + cos_0_2 + cos_1_2)/3
def save_img(dir, img_tensor, iteration=0, trial=1, is_batch=True, dm=0, ds=1):
    '''save torch tensor to img
    
    : dir save_img dir
    :img_tensor img tensor 
    :iteration iteration
    : is_batch  is img_tensor includes batch dimension
    : dm dataset mean in each channel [1,3,1,1]
    : ds dataset stard variation in each channel [1,3,1,1]
    
    '''
    import torchvision
    import os

    if not os.path.exists(dir):
        os.mkdir(dir)

    trial_path = dir + '/' + f'{trial}' 
    if not os.path.exists(trial_path):
        os.mkdir(trial_path)

    img_tensor = torch.clamp(img_tensor * ds + dm, 0, 1)
    if is_batch:
       for  i, img in enumerate(img_tensor):

            img = torchvision.transforms.ToPILImage()(img)
            path = trial_path + '/' + f'{iteration}_{i}.png' 
            img.save(path)
    else:
        # img.mul_(ds).add_(dm).clamp_(0, 1)
        # img = torchvision.transforms.ToPILImage()(img_tensor)
        # img.save(path)
        raise Exception('Except batch dimension in img tensor')

def save_img_d(path, img_tensor, is_batch=True, dm=0, ds=1):
    '''save torch tensor to img
    
    : dir save_img dir
    :img_tensor img tensor 
    :iteration iteration
    : is_batch  is img_tensor includes batch dimension
    : dm dataset mean in each channel [1,3,1,1]
    : ds dataset stard variation in each channel [1,3,1,1]
    
    '''
    import torchvision
    import os

    img_tensor = torch.clamp(img_tensor * ds + dm, 0, 1)
    if is_batch:
       for  i, img in enumerate(img_tensor):
            img = torchvision.transforms.ToPILImage()(img)
            img.save(path)
    else:
        # img.mul_(ds).add_(dm).clamp_(0, 1)
        # img = torchvision.transforms.ToPILImage()(img_tensor)
        # img.save(path)
        raise Exception('Except batch dimension in img tensor') 
def add_nosie_to_candicate(candidate, iteration, interval=100, num_levels=10, scale=0.1, start_iter=2000):
    """
    : candidate original sample
    : iterations current iteration
    """
    betas = reversed(torch.linspace(0, 10, num_levels) * scale)
    if iteration % interval == 0 and iteration > start_iter:
        idx = iteration // interval - 3

        if idx < num_levels:
            print(f'{iteration} add noise idx {idx}')
            noise = torch.randn_like(candidate).to(candidate.device)
            candidate.add_(noise * betas[idx])  
    


class OptimizationBasedAttacker(_BaseAttacker):
    """Implements a wide spectrum of optimization-based attacks."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)
        objective_fn = objective_lookup.get(self.cfg.objective.type)
        if objective_fn is None:
            raise ValueError(f"Unknown objective type {self.cfg.objective.type} given.")
        else:
            self.objective = objective_fn(**self.cfg.objective)
        self.regularizers = []
        try:
            for key in self.cfg.regularization.keys():
                if self.cfg.regularization[key].scale > 0:
                    self.regularizers += [regularizer_lookup[key](self.setup, **self.cfg.regularization[key])]
        except AttributeError:
            pass  # No regularizers selected.

        try:
            self.augmentations = []
            for key in self.cfg.augmentations.keys():
                self.augmentations += [augmentation_lookup[key](**self.cfg.augmentations[key])]
            self.augmentations = torch.nn.Sequential(*self.augmentations).to(**setup)
        except AttributeError:
            self.augmentations = torch.nn.Sequential()  # No augmentations selected.

    def __repr__(self):
        n = "\n"
        return f"""Attacker (of type {self.__class__.__name__}) with settings:
    Hyperparameter Template: {self.cfg.type}

    Objective: {repr(self.objective)}
    Regularizers: {(n + ' '*18).join([repr(r) for r in self.regularizers])}
    Augmentations: {(n + ' '*18).join([repr(r) for r in self.augmentations])}

    Optimization Setup:
        {(n + ' ' * 8).join([f'{key}: {val}' for key, val in self.cfg.optim.items()])}
        """

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False, activation=None, constraints=None):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        #  TODO: delete all activation arguments
        # attempted in optimization_and_recursive attack, no longer need
        if activation is not None:
            labels = activation

        # Main reconstruction loop starts here:
        scores = torch.zeros(self.cfg.restarts.num_trials)
        candidate_solutions = []
        try:
            for trial in range(self.cfg.restarts.num_trials):
                candidate_solutions += [
                    self._run_trial(rec_models, shared_data, labels, stats, trial, initial_data, dryrun, constraints)
                ]
                scores[trial] = self._score_trial(candidate_solutions[trial], labels, rec_models, shared_data)
        except KeyboardInterrupt:
            print("Trial procedure manually interruped.")
            pass
        optimal_solution = self._select_optimal_reconstruction(candidate_solutions, scores, stats)
        reconstructed_data = dict(data=optimal_solution, labels=labels)
        if server_payload[0]["metadata"].modality == "text":
            reconstructed_data = self._postprocess_text_data(reconstructed_data)
        if "ClassAttack" in server_secrets:
            # Only a subset of images was actually reconstructed:
            true_num_data = server_secrets["ClassAttack"]["true_num_data"]
            reconstructed_data["data"] = torch.zeros([true_num_data, *self.data_shape], **self.setup)
            reconstructed_data["data"][server_secrets["ClassAttack"]["target_indx"]] = optimal_solution
            reconstructed_data["labels"] = server_secrets["ClassAttack"]["all_labels"]
        return reconstructed_data, stats

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, initial_data=None, dryrun=False, constraints=None):
        """Run a single reconstruction trial."""

        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels, constraints)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])

        # Initialize candidate reconstruction data
        candidate = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape])
        if initial_data is not None:
            candidate.data = initial_data.data.clone().to(**self.setup)

        best_candidate = candidate.detach().clone()
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)

        # Initialize optimizers
        optimizer, scheduler = self._init_optimizer([candidate])
        current_wallclock = time.time()

        # gaussian_blur = torchvision.transforms.GaussianBlur(3)
        # savve log info
        if (dir_path:=self.cfg.save.out_dir) is not None:
            log.setLevel(logging.INFO)
            if not os.path.exists( dir_path):
                os.mkdir(dir_path)
            log.addHandler(logging.FileHandler(dir_path + '/log'))
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                closure = self._compute_objective(candidate, labels, rec_model, optimizer, shared_data, iteration)
                time_closure_prefix = time.time() - current_wallclock
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                time_update = time.time() - current_wallclock - time_closure_prefix
                scheduler.step()

                #scheduler pattern size
                # if self.cfg.optim.patched is not None:
                #     if 1000 == iteration:
                #         self.cfg.optim.patched /= 2
                #     elif 2000 == iteration:
                #         self.cfg.optim.patched /= 2
                    # elif 3000 == iteration:
                    #     self.cfg.optim.patched /= 2
                    # elif 3000 == iteration:
                    #     self.cfg.optim.patched /= 2
            
                    

                with torch.no_grad():
                    # Project into image space
                    if self.cfg.optim.boxed:
                        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                    if objective_value < minimal_value_so_far:
                        minimal_value_so_far = objective_value.detach()
                        best_candidate = candidate.detach().clone()
                    
                    # if (noise:=self.cfg.accelerate.add_noise) is not None:
                    #     print(noise.interval, noise.scale, noise.num_levels)
                    #     add_nosie_to_candicate(candidate=candidate, iteration=iteration, interval=noise.interval, num_levels=noise.num_levels,  scale=noise.scale)
                    
                    # if self.cfg.optim.patched is not None and self.cfg.optim.patched > 1:
                    #     pattern = int(self.cfg.optim.patched)
                        #         candidate.grad[:, :, x_idx*pattern:(x_idx+1)*pattern, y_idx*pattern:(y_idx+1)*pattern].copy_(torch.mean(candidate.grad[:, :, x_idx*pattern:(x_idx+1)*pattern, y_idx*pattern:(y_idx+1)*pattern], dim=(2,3), keepdim=True))   

                        # avg_pool = torch.nn.AvgPool2d(pattern, pattern)
                        # mean_candidate = avg_pool(candidate)
                        # reshpae_candidate = mean_candidate.repeat_interleave(pattern, dim=2).repeat_interleave(pattern, dim=3)
                        # candidate.copy_(reshpae_candidate)
                    

                    # if iteration % self.cfg.optim.callback == 0:

                    #     #gaussian_blur image
                    #     candidate_blur = gaussian_blur(candidate)
                    #     candidate.copy_(candidate_blur)
                    
                time_projectImage = time.time() - current_wallclock - time_update
                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    # if iteration > 2000:
                    # with torch.no_grad():
                    #     if self.cfg.accelerate.denoise:
                    #         candidate_weight = 0.9

                    #         # denoise_img = img_denoise(candidate, dm=self.dm, ds=self.ds)
                    #         # denoise_img = sucnet_denoise(candidate, dm=self.dm, ds=self.ds)
                    #         denoise_img = swinir_super_resolution(candidate, dm=self.dm, ds=self.ds) 

                    #         mix_img = candidate_weight * candidate + (1.0 - candidate_weight ) * denoise_img
                    #         # candidate.copy_(denoise_img)
                    #         candidate.copy_(mix_img)

                    f_regularizer = ''
                    for regularizer_objective in self.current_regularizer_objectives:
                        f_regularizer += f'{regularizer_objective:2.4f} '
                # if True:
                    timestamp = time.time()
                    log.info(
                        f"| It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                        f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                        f" regularizer_objective_loss: {f_regularizer}" 
                        # f" T closure_prefix {time_closure_prefix} | T update {time_update} | T project {time_projectImage}"
                    )
                    current_wallclock = timestamp
                    if self.cfg.save.out_dir is not None:
                        if self.cfg.save.idx is not None:
                            path = self.cfg.save.out_dir + f'/lq_{self.cfg.optim.max_iterations}/{self.cfg.save.idx}.png' #only save the last iteration image
                            save_img_d(path, candidate, dm=self.dm, ds=self.ds)

                        else: 
                            save_img(self.cfg.save.out_dir, candidate, iteration=iteration, trial=trial, dm=self.dm, ds=self.ds) # save.out_dir / trail / iteration_{i}

                if not torch.isfinite(objective_value):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break

                stats[f"Trial_{trial}_Val"].append(objective_value.item())

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass

        return best_candidate.detach()

    def _compute_objective(self, candidate, labels, rec_model, optimizer, shared_data, iteration, activation=None):
        def closure():

            start_time = time.time()

            optimizer.zero_grad()
            time_zero_grad = time.time() - start_time


            if self.cfg.differentiable_augmentations:
                candidate_augmented = self.augmentations(candidate)
            else:
                candidate_augmented = candidate
                candidate_augmented.data = self.augmentations(candidate.data)

            time_augmentation = time.time() - start_time - time_zero_grad

            total_objective = 0
            total_task_loss = 0
            for model, data in zip(rec_model, shared_data):
                if self.cfg.layer_weights == None:
                    layer_weights = None
                elif self.cfg.layer_weights == 'equal':
                    layer_weights = [1 for i in range(len(data["gradients"]))]
                elif self.cfg.layer_weights == 'descentWeights': 
                    layer_weights = []
                    for i in range(len(data["gradients"])):
                        if i<130:
                            layer_weights.append(1.0)
                        else:
                            layer_weights.append(1.0)
                elif self.cfg.layer_weights.endswith('idx'):
                    layer_weights = torch.load(f'/home/zx/Gitrepo/breaching/breaching/attacks/auxiliaries/{self.cfg.layer_weights}').to(torch.int64)
                else:
                    raise NotImplementedError('Please input correct layer_weights')

                # print(type(data["gradients"]))
                # print(layer_weights)
                objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, labels, layer_weights=layer_weights)
                total_objective += objective
                total_task_loss += task_loss

            time_loss1back = time.time() - start_time - time_zero_grad - time_augmentation

            regularizer_objectives = []
            for regularizer in self.regularizers:
                regularizer_objective = regularizer(candidate_augmented)
                regularizer_objectives.append(regularizer_objective)
                
                total_objective += regularizer_objective

                # total_objective += regularizer(candidate_augmented)


            time_cal_regularizer = time.time() - start_time - time_zero_grad - time_augmentation - time_loss1back

            if total_objective.requires_grad:
                loss2back_start_time = time.time()
                total_objective.backward(inputs=candidate, create_graph=False)
                time_loss2back = time.time() - loss2back_start_time
                # print(f'the second backward time {time_loss2back}')

            # log.info(f'{cosine_sim_layer(candidate.grad)}')
            with torch.no_grad():
                if self.cfg.optim.langevin_noise > 0:
                    step_size = optimizer.param_groups[0]["lr"]
                    noise_map = torch.randn_like(candidate.grad)
                    candidate.grad += self.cfg.optim.langevin_noise * step_size * noise_map
                if self.cfg.optim.grad_clip is not None:
                    grad_norm = candidate.grad.norm()
                    if grad_norm > self.cfg.optim.grad_clip:
                        candidate.grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                # if self.cfg.optim.patched is not None and self.cfg.optim.patched > 1:
                #     pattern = int(self.cfg.optim.patched)
                #     # b, c, h, w = candidate.grad.shape
                #     # x_len, y_len = h//pattern, w//pattern
                #     # for x_idx in range(x_len):
                #     #     for y_idx in range(y_len):
                #     #         # candidate.grad[:,0, x_idx*pattern:(x_idx+1)*pattern, y_idx*pattern:(y_idx+1)*pattern].copy_(torch.mean(candidate.grad[:, 0, x_idx*pattern:(x_idx+1)*pattern, y_idx*pattern:(y_idx+1)*pattern])) 
                #     #         # candidate.grad[:,1, x_idx*pattern:(x_idx+1)*pattern, y_idx*pattern:(y_idx+1)*pattern].copy_(torch.mean(candidate.grad[:, 1, x_idx*pattern:(x_idx+1)*pattern, y_idx*pattern:(y_idx+1)*pattern])) 
                #     #         # candidate.grad[:,2, x_idx*pattern:(x_idx+1)*pattern, y_idx*pattern:(y_idx+1)*pattern].copy_(torch.mean(candidate.grad[:, 2, x_idx*pattern:(x_idx+1)*pattern, y_idx*pattern:(y_idx+1)*pattern])) 
                #     #         candidate.grad[:, :, x_idx*pattern:(x_idx+1)*pattern, y_idx*pattern:(y_idx+1)*pattern].copy_(torch.mean(candidate.grad[:, :, x_idx*pattern:(x_idx+1)*pattern, y_idx*pattern:(y_idx+1)*pattern], dim=(2,3), keepdim=True))   

                #     avg_pool = torch.nn.AvgPool2d(pattern, pattern)
                #     mean_grad = avg_pool(candidate.grad)
                #     reshpae_grad = mean_grad.repeat_interleave(pattern, dim=2).repeat_interleave(pattern, dim=3)
                #     candidate.grad.copy_(reshpae_grad)




                if self.cfg.optim.signed is not None:
                    if self.cfg.optim.signed == "soft":
                        scaling_factor = (
                            1 - iteration / self.cfg.optim.max_iterations
                        )  # just a simple linear rule for now
                        candidate.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                    elif self.cfg.optim.signed == "hard":
                        candidate.grad.sign_()
                    else:
                        pass
            # log.info(f'{cosine_sim_layer(candidate.grad)}')
            time_defense = time.time() - start_time - time_zero_grad - time_augmentation - time_loss1back - time_cal_regularizer - time_loss2back
            time_total = time.time() - start_time
            # print(f'Total: {time_total} zerograd {time_zero_grad} aug {time_augmentation} loss1back {time_loss1back} regualr {time_cal_regularizer} loss2back {time_loss2back} defense {time_defense}')
            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            self.current_regularizer_objectives = regularizer_objectives
            return total_objective

        return closure

    def _score_trial(self, candidate, labels, rec_model, shared_data):
        """Score candidate solutions based on some criterion."""

        if self.cfg.restarts.scoring in ["euclidean", "cosine-similarity"]:
            objective = Euclidean() if self.cfg.restarts.scoring == "euclidean" else CosineSimilarity()
            objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])
            score = 0
            for model, data in zip(rec_model, shared_data):
                layer_weights = [1 for i in range(len(data["gradients"]))]
                score += objective(model, data["gradients"], candidate, labels, layer_weights)[0]
        elif self.cfg.restarts.scoring in ["TV", "total-variation"]:
            score = TotalVariation(scale=1.0)(candidate)
        else:
            raise ValueError(f"Scoring mechanism {self.cfg.scoring} not implemented.")
        return score if score.isfinite() else float("inf")

    def _select_optimal_reconstruction(self, candidate_solutions, scores, stats):
        """Choose one of the candidate solutions based on their scores (for now).

        More complicated combinations are possible in the future."""
        optimal_val, optimal_index = torch.min(scores, dim=0)
        optimal_solution = candidate_solutions[optimal_index]
        stats["opt_value"] = optimal_val.item()
        if optimal_val.isfinite():
            log.info(f"Optimal candidate solution with rec. loss {optimal_val.item():2.4f} selected.")
            return optimal_solution
        else:
            log.info("No valid reconstruction could be found.")
            return torch.zeros_like(optimal_solution)
