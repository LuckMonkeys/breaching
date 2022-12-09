"""Implementation for basic gradient inversion attacks.

This covers optimization-based reconstruction attacks as in Wang et al. "Beyond Infer-
ring Class Representatives: User-Level Privacy Leakage From Federated Learning."
and convers subsequent developments such as
* Zhu et al., "Deep Leakage from gradients",
* Geiping et al., "Inverting Gradients - How easy is it to break privacy in FL"
* ?
"""

from audioop import avgpp
from re import I
from tracemalloc import start
import torch
import torch.nn.functional as F
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
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from functools import partial

log = logging.getLogger(__name__)
import os

def save_diffusion_img(img, save_dir, name):
    assert os.path.exists(save_dir), f'{save_dir} not exsit!'

    th_save_img = ((img.clone().detach() + 1)/2).clamp(0,1).squeeze(0)
    pil_save_img = torchvision.transforms.ToPILImage()(th_save_img)
    pil_save_img.save(os.path.join(save_dir, name))

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
    

class OptimizationDiffusionAttacker(_BaseAttacker):
    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)

        ## load optimizaiton parameters
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
    
        ## load diffusion parmameters
         
        from .accelarate.guided_diffusion import guided_diffusion 
        
        NUM_CLASSES, model_and_diffusion_defaults, classifier_defaults, create_model_and_diffusion, create_classifier, add_dict_to_argparser, args_to_dict = guided_diffusion.NUM_CLASSES, guided_diffusion.model_and_diffusion_defaults, guided_diffusion.classifier_defaults, guided_diffusion.create_model_and_diffusion, guided_diffusion.create_classifier, guided_diffusion.add_dict_to_argparser, guided_diffusion.args_to_dict
    
        
        self.args = cfg_attack.diffusion 
        self.diff_model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(self.args, model_and_diffusion_defaults().keys())
        )
        
        
        self.diff_model.load_state_dict(
            torch.load(self.args.model_path)
        )
        self.diff_model.to(device=self.setup['device'])
        if self.args.use_fp16:
            self.diff_model.convert_to_fp16()
        self.diff_model.eval()

        self.classifier = create_classifier(**args_to_dict(self.args, classifier_defaults().keys()))
        self.classifier.load_state_dict(
            torch.load(self.args.classifier_path)
        )
        self.classifier.to(device=self.setup['device'])
        if self.args.classifier_use_fp16:
            self.classifier.convert_to_fp16()
        self.classifier.eval()

        self.sample_fn = (
            # self.diffusion.p_sample if not self.args.use_ddim else self.diffusion.ddim_sample
            self.diffusion.sample_or_diffusion if not self.args.use_ddim else self.diffusion.ddim_sample
        )

    def cond_grad_fn(self, x, t, y, rec_models, shared_data):
        assert y is not None
        
        
        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_models, shared_data, y, constraints=None)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])
        
        
        with torch.enable_grad():
            candidate = x.detach().requires_grad_(True)
            
            if self.cfg.differentiable_augmentations:
                candidate_augmented = self.augmentations(candidate)
            else:
                candidate_augmented = candidate
                candidate_augmented.data = self.augmentations(candidate.data)

            total_objective = 0
            total_task_loss = 0
            for model, data in zip(rec_models, shared_data):

                objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, y, layer_weights=None)
                total_objective += objective
                total_task_loss += task_loss

            regularizer_objectives = []
            for regularizer in self.regularizers:
                regularizer_objective = regularizer(candidate_augmented)
                regularizer_objectives.append(regularizer_objective)
                
                total_objective += regularizer_objective

            # total_objective = torch.log(total_objective)
            total_objective = total_objective * -1 # x = x - \eta * x.grad = x + \eta*(-x.grad)
            if total_objective.requires_grad:
                total_objective.backward(inputs=candidate, create_graph=False)

            # print(candidate.grad.shape)
            # print(candidate.grad * self.args.gradient_attack_scale) 
            # exit(0)
            return candidate.grad * self.args.gradient_attack_scale
    
    def cond_fn(self, x, t, y=None):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1) 
            selected = log_probs[range(len(logits)), y.view(-1)] 
            
            return torch.autograd.grad(selected.sum(), x_in)[0] * self.args.classifier_scale
    
    
    def cond_fn_hybrid(self, x, t, y, rec_models, share_data):
        # print('guide from classifier\n', self.cond_fn(x,t,y))
        # print('guide from grad\n', self.cond_grad_fn(x, t, y, rec_models, share_data))
        return self.cond_fn(x, t, y) + self.cond_grad_fn(x, t, y, rec_models, share_data)

    def get_cond_fn(self, strategy='hybrid', rec_models=None, share_data=None):
        assert strategy in ['classifier', 'grad', 'hybrid']
        
        if strategy == 'classifier':
           return self.cond_fn
        elif strategy == 'grad':
           return self.cond_grad_fn  
        else:
           return partial(self.cond_fn_hybrid, rec_models=rec_models, share_data=share_data)
    
    
    def model_fn(self,x, t, y=None):
        assert y is not None
        return self.diff_model(x, t, y if self.args.class_cond else None)

    def __repr__(self):
        return f'Attacker {self.__class__.__name__} smaple ddim {self.args.use_ddim}'
    
    
    def get_resample_indices(self,t_T=250, jump_len=10, jump_n_sample=10):
        jumps = {}
        for j in range(0, t_T - jump_len, jump_len):
            jumps[j] = jump_n_sample - 1
        t = t_T
        ts = []
        while t >= 1:
            t = t-1
            ts.append(t)
            if jumps.get(t, 0) > 0:
                jumps[t] = jumps[t] - 1
                for _ in range(jump_len):
                    t=t+1
                    ts.append(t)
        # ts.append(-1)
        ts.insert(0, 250)
        
        return ts

    
     
    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):

        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        
        scores = torch.zeros(self.cfg.restarts.num_trials)

        if len(shared_data) == 1:
            input_gradient = shared_data[0]["gradients"]
        else:
            raise NotImplementedError('large batch size in GAN attack not implement noew')
        
        
        shape = [shared_data[0]["metadata"]["num_data_points"], *self.data_shape]

        img = torch.randn(*shape, device=self.setup['device'])

        ## change indices to fit resample
        # indices = list(range(self.diffusion.num_timesteps))[::-1]
        indices = self.get_resample_indices(t_T=self.diffusion.num_timesteps, jump_len=10, jump_n_sample=2)
        # print(indices)
        # exit(0)

        model_kwargs = {}
        model_kwargs['y'] = labels
       
        
        # partial_cond_grad_fn = partial(self.cond_grad_fn, rec_model=rec_models, shared_data=shared_data)
        cond_fn = self.get_cond_fn(strategy='classifier')
        # cond_fn = self.get_cond_fn(strategy='hybrid', rec_models=rec_models, share_data=shared_data)

        print(indices)
        for i in tqdm(range(1, len(indices))):
        # for i in tqdm(range(0, len(indices))):
            t = torch.tensor([indices[i]] * shape[0], device=self.setup['device'])
            t_prev = torch.tensor([indices[i-1]] * shape[0], device=self.setup['device'])
            with torch.no_grad():
                # out = self.diffusion.p_sample(
                out = self.sample_fn(
                    self.model_fn,
                    img,
                    t,
                    t_prev,
                    clip_denoised=self.args.clip_denoised,
                    denoised_fn=None,
                    # cond_fn=self.cond_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )

            # if i > len(indices) * 0.5:
            is_grad_gudie = True
            if is_grad_gudie and out.get("mean") is not None:
                th_img = (out["mean"].clone().detach() + 1)/2
                #do normalize
                img_normalize = (th_img - self.dm) / self.ds
                candidate_solutions = []
                for trial in range(self.cfg.restarts.num_trials):
                    candidate_solutions += [
                        self._run_trial(rec_model=rec_models, shared_data=shared_data, labels=labels, stats=stats, dryrun=dryrun, initial_data=img_normalize, trial=trial)
                    ]

                    # scores[trial] = self._score_trial(candidate_solutions[trial], labels, rec_models, shared_data)
                
                # optimal_solution = self._select_optimal_reconstruction(candidate_solutions, scores, stats)
                optimal_solution = candidate_solutions[0]

                # project to [-1, 1] 
                out["mean"] = (optimal_solution.clone().detach() * self.ds + self.dm) * 2 -1 
                noise = torch.randn_like(img)
                img = out["mean"] + out["std"] * noise 
                img = img.detach()
            else:
                img = out["sample"]


            # save_img
            save_mean = False
            if out.get("mean") is not None and self.cfg.save.out_dir is not None:
                if save_mean:
                    save_diffusion_img(out["mean"], self.cfg.save.out_dir, f'mean_{i}.png')
                save_diffusion_img(img, self.cfg.save.out_dir, f'{i}.png')
        
            if dryrun:
                break
        
        # optimal_solution = self._select_optimal_reconstruction([img_optimized], scores, stats)
        # reconstructed_data = dict(data=optimal_solution, labels=labels)
        reconstructed_data = dict(data=img.detach(), labels=labels)
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

        try:
            for iteration in range(self.cfg.optim.max_iterations):
                closure = self._compute_objective(candidate, labels, rec_model, optimizer, shared_data, iteration)
                time_closure_prefix = time.time() - current_wallclock
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                time_update = time.time() - current_wallclock - time_closure_prefix
                scheduler.step()


                with torch.no_grad():
                    # Project into image space
                    if self.cfg.optim.boxed:
                        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                    if objective_value < minimal_value_so_far:
                        minimal_value_so_far = objective_value.detach()
                        best_candidate = candidate.detach().clone()
                    
                if not torch.isfinite(objective_value):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break

                stats[f"Trial_{trial}_Val"].append(objective_value.detach().item())

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

            total_objective = 0
            total_task_loss = 0
            for model, data in zip(rec_model, shared_data):

                objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, labels, layer_weights=None)
                total_objective += objective
                total_task_loss += task_loss

            regularizer_objectives = []
            for regularizer in self.regularizers:
                regularizer_objective = regularizer(candidate_augmented)
                regularizer_objectives.append(regularizer_objective)
                
                total_objective += regularizer_objective

            if total_objective.requires_grad:
                total_objective.backward(inputs=candidate, create_graph=False)

            with torch.no_grad():
                if self.cfg.optim.langevin_noise > 0:
                    step_size = optimizer.param_groups[0]["lr"]
                    noise_map = torch.randn_like(candidate.grad)
                    candidate.grad += self.cfg.optim.langevin_noise * step_size * noise_map
                if self.cfg.optim.grad_clip is not None:
                    grad_norm = candidate.grad.norm()
                    if grad_norm > self.cfg.optim.grad_clip:
                        candidate.grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))


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
    
   ##modify original run trial process 