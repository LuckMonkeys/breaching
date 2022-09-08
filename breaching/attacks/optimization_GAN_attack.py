from calendar import timegm
from tracemalloc import start
import numpy as np
import time
import torch
import torch.nn as nn
import nevergrad as ng
from tqdm import tqdm
from PIL import Image

from pytorch_pretrained_biggan import convert_to_images, truncated_noise_sample
# import defense

# from turbo import Turbo1
from breaching.attacks.auxiliaries.turbo_1 import Turbo1
from .base_attack import _BaseAttacker


class BOReconstructor():
    """
    BO Reconstruction for BigGAN

    """
    def __init__(self, fl_model, generator, loss_fn, num_classes=1000, search_dim=(128,), strategy='BO', budget=1000, use_tanh=False, use_weight=False, defense_setting=None):

        self.generator = generator
        self.budget = budget
        self.search_dim = search_dim
        self.use_tanh = use_tanh
        self.num_samples = 10
        self.weight = None
        self.defense_setting = defense_setting

        self.fl_setting = {'loss_fn':loss_fn, 'fl_model':fl_model, 'num_classes':num_classes}

        if use_weight:
            self.weight = np.ones(62,)
            for i in range(0, 20):
                self.weight[3*i:3*(i+1)] /= 2**i


    def evaluate_loss(self, z, labels, input_gradient):
        return self.ng_loss(z=z, input_gradient=input_gradient, metric='l2',
                        labels=labels, generator=self.generator, weight=self.weight,
                        use_tanh=self.use_tanh, defense_setting=self.defense_setting, **self.fl_setting
                       )

    def reconstruct(self, input_gradient, use_pbar=True):

        labels = self.infer_label(input_gradient)
        print('Inferred label: {}'.format(labels))
        
        if self.defense_setting is not None:
            if 'clipping' in self.defense_setting:
                total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in input_gradient]), 2)
                self.defense_setting['clipping'] = total_norm.item()
                print('Estimated defense parameter: {}'.format(self.defense_setting['clipping']))
            if 'compression' in self.defense_setting:
                n_zero, n_param = 0, 0
                for i in range(len(input_gradient)):
                    n_zero += torch.sum(input_gradient[i]==0)
                    n_param += torch.numel(input_gradient[i])
                self.defense_setting['compression'] = 100 * (n_zero/n_param).item()
                print('Estimated defense parameter: {}'.format(self.defense_setting['compression']))

        c = torch.nn.functional.one_hot(labels, num_classes=self.fl_setting['num_classes']).to(input_gradient[0].device)



        z_lb = -2*np.ones(self.search_dim) # lower bound, you may change -10 to -inf
        z_ub = 2*np.ones(self.search_dim) # upper bound, you may change 10 to inf

        f = lambda z:self.evaluate_loss(z, labels, input_gradient)

        self.optimizer = Turbo1(
                                f=f,  # Handle to objective function
                                lb=z_lb,  # Numpy array specifying lower bounds
                                ub=z_ub,  # Numpy array specifying upper bounds
                                n_init=256,  # Number of initial bounds from an Latin hypercube design
                                max_evals = self.budget,  # Maximum number of evaluations
                                batch_size=10,  # How large batch size TuRBO uses
                                verbose=True,  # Print information from each batch
                                use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                                max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                                n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                                min_cuda=1024,  # Run on the CPU for small datasets
                                device="cuda", #next(generator.parameters()).device,  # "cpu" or "cuda"
                                dtype="float32",  # float64 or float32
                            )

        self.optimizer.optimize()

        X = self.optimizer.X  # Evaluated points of z
        fX = self.optimizer.fX  # Observed values of ng_loss
        ind_best = np.argmin(fX)
        loss_res, z_res = fX[ind_best], X[ind_best, :]

        loss_res = self.evaluate_loss(z_res, labels, input_gradient)
        z_res = torch.from_numpy(z_res).unsqueeze(0).to(input_gradient[0].device)
        if self.use_tanh:
            z_res = z_res.tanh()

        with torch.no_grad():
            x_res = self.generator(z_res.float(), c.float(), 1)
        x_res = nn.functional.interpolate(x_res, size=(224, 224), mode='area')
        img_res = convert_to_images(x_res.cpu())

        return z_res, x_res, img_res, loss_res

    @staticmethod
    def infer_label(input_gradient, num_inputs=1):
        last_weight_min = torch.argsort(torch.sum(input_gradient[-2], dim=-1), dim=-1)[:num_inputs]
        labels = last_weight_min.detach().reshape((-1,)).requires_grad_(False)
        return labels

    @staticmethod
    def ng_loss(z, # latent variable to be optimized
                loss_fn, # loss function for FL model
                input_gradient,
                labels,
                generator,
                fl_model,
                num_classes=1000,
                metric='l2',
                use_tanh=True,
                weight=None, # weight to be applied when calculating the gradient matching loss
                defense_setting=None # adaptive attack against defense
               ):

        z = torch.Tensor(z).unsqueeze(0).to(input_gradient[0].device)
        if use_tanh:
            z = z.tanh()

        c = torch.nn.functional.one_hot(labels, num_classes=num_classes).to(input_gradient[0].device)

        with torch.no_grad():
            x = generator(z, c.float(), 1)

        x = nn.functional.interpolate(x, size=(224, 224), mode='area')

        # compute the trial gradient
        target_loss, _, _ = loss_fn(fl_model(x), labels)
        trial_gradient = torch.autograd.grad(target_loss, fl_model.parameters())
        trial_gradient = [grad.detach() for grad in trial_gradient]

        # adaptive attack against defense
#         if defense_setting is not None:
#             if 'noise' in defense_setting:
#                 pass
# #                 trial_gradient = defense.additive_noise(trial_gradient, std=defense_setting['noise'])
#             if 'clipping' in defense_setting:
#                 trial_gradient = defense.gradient_clipping(trial_gradient, bound=defense_setting['clipping'])
#             if 'compression' in defense_setting:
#                 trial_gradient = defense.gradient_compression(trial_gradient, percentage=defense_setting['compression'])
#             if 'representation' in defense_setting: # for ResNet
#                 mask = input_gradient[-2][0]!=0
#                 trial_gradient[-2] = trial_gradient[-2] * mask

        if weight is not None:
            assert len(weight) == len(trial_gradient)
        else:
            weight = [1]*len(trial_gradient)

        # calculate l2 norm
        dist = 0
        for i in range(len(trial_gradient)):
            if metric == 'l2':
                dist += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum()*weight[i]
            elif metric == 'l1':
                dist += ((trial_gradient[i] - input_gradient[i]).abs()).sum()*weight[i]
        dist /= len(trial_gradient)

        if not use_tanh:
            KLD = -0.5 * torch.sum(1 + torch.log(torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(z.squeeze(), axis=-1).pow(2) - torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2))
            dist += 0.1*KLD

        return dist.item()



class NGReconstructor(_BaseAttacker):
    """
    Reconstruction for BigGAN

    """
    # def __init__(self, fl_model, generator, loss_fn, num_classes=1000, search_dim=(128,), strategy='CMA', budget=500, use_tanh=True, use_weight=False, defense_setting=None):

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)
        self.budget = cfg_attack.optim.budget
        self.search_dim = cfg_attack.optim.search_dim
        self.use_tanh = cfg_attack.optim.use_tanh
        # self.num_samples = 50 # what meaning ????
        self.num_samples = cfg_attack.optim.num_samples # what meaning ????
        self.weight = None
        self.use_weight = cfg_attack.optim.use_weight
        self.strategy = cfg_attack.optim.optimizer
        self.metric = cfg_attack.objective.type
        
        ##fl setting
        self.fl_loss_fn = loss_fn
        self.num_classes = cfg_attack.fl_num_class
        # self.defense_setting = defense_setting

        self.device = setup['device'] 


        # parametrization = ng.p.Array(shape=search_dim)
        parametrization = ng.p.Array(init=np.random.rand(self.search_dim))
        #parametrization = ng.p.Array(init=np.zeros(search_dim))#.set_mutation(sigma=1.0)
        self.optimizer = ng.optimizers.registry[self.strategy](parametrization=parametrization, budget=self.budget)
#         self.optimizer.parametrization.register_cheap_constraint(lambda x: (x>=-2).all() and (x<=2).all())

        # self.fl_setting = {'loss_fn':loss_fn, 'fl_model':fl_model, 'num_classes':num_classes}

        if self.use_weight:
            self.weight = np.ones(62,)
            for i in range(0, 20):
                self.weight[3*i:3*(i+1)] /= 2**i

    def __repr__(self):

        return f'Attacker {self.__class__.__name__} with budget {self.budget}'

    def evaluate_loss(self, z, labels, input_gradient, models):
        # return self.ng_loss(z=z, input_gradient=input_gradient, metric='l2',
        #                 labels=labels, generator=self.generator, weight=self.weight,
        #                 use_tanh=self.use_tanh, defense_setting=self.defense_setting, **self.fl_setting
        #                )

        if len(models) == 1:

            return self.ng_loss(z=z, loss_fn=self.fl_loss_fn, input_gradient=input_gradient, metric=self.metric,
                            labels=labels, generator=self.generator, fl_model=models[0], num_classes=self.num_classes, weight=self.weight,
                            use_tanh=self.use_tanh 
                        )
        else:
            raise NotImplementedError('large batch size in GAN attack not implement noew')
    def reconstruct(self, server_payload, shared_data, generator, use_pbar=True, dryrun=False):

        self.generator = generator
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        if len(shared_data) == 1:
            input_gradient = shared_data[0]["gradients"]
        else:
            raise NotImplementedError('large batch size in GAN attack not implement noew')
        labels = self.infer_label(input_gradient)
        print('Inferred label: {}'.format(labels))
        
        # if self.defense_setting is not None:
        #     if 'clipping' in self.defense_setting:
        #         total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in input_gradient]), 2)
        #         self.defense_setting['clipping'] = total_norm.item()
        #         print('Estimated defense parameter: {}'.format(self.defense_setting['clipping']))
        #     if 'compression' in self.defense_setting:
        #         n_zero, n_param = 0, 0
        #         for i in range(len(input_gradient)):
        #             n_zero += torch.sum(input_gradient[i]==0)
        #             n_param += torch.numel(input_gradient[i])
        #         self.defense_setting['compression'] = 100 * (n_zero/n_param).item()
        #         print('Estimated defense parameter: {}'.format(self.defense_setting['compression']))

        c = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).to(input_gradient[0].device)

        pbar = tqdm(range(self.budget)) if use_pbar else range(self.budget)

        for r in pbar:
            start_time = time.time()
            ng_data = [self.optimizer.ask() for _ in range(self.num_samples)]
            time_gs = 0.0
            loss = []

            for i in range(self.num_samples):
                ls, time_g = self.evaluate_loss(z=ng_data[i].value, labels=labels, input_gradient=input_gradient, models=rec_models)
                loss.append(ls)
                time_gs += time_g

            # loss = [self.evaluate_loss(z=ng_data[i].value, labels=labels, input_gradient=input_gradient) for i in range(self.num_samples)]
            for z, l in zip(ng_data, loss):
                self.optimizer.tell(z, l)

            if use_pbar:
                pbar.set_description("Loss {:.6}".format(np.mean(loss)))
            else:
                print("Round {} - Loss {:.6}".format(r, np.mean(loss)))
            total_time = time.time() - start_time
            # print(f'total time {total_time}, time_gs {time_gs}, time_ratio {time_gs/total_time}')


            recommendation = self.optimizer.provide_recommendation()
            z_res = torch.from_numpy(recommendation.value).unsqueeze(0).to(input_gradient[0].device)
            if self.use_tanh:
                z_res = z_res.tanh()
            with torch.no_grad():
                x_res = self.generator(z_res.float(), c.float(), 1)
            x_res = nn.functional.interpolate(x_res, size=(224, 224), mode='area')
            img_res = convert_to_images(x_res.cpu())
            # for img in img_res:
            #     img.save(f'/home/zx/Gitrepo/breaching/out/GAN/CMA/{r}.png')

            if dryrun:
                break


            
        recommendation = self.optimizer.provide_recommendation()
        z_res = torch.from_numpy(recommendation.value).unsqueeze(0).to(input_gradient[0].device)
        if self.use_tanh:
            z_res = z_res.tanh()
        loss_res = self.evaluate_loss(recommendation.value, labels, input_gradient, models=rec_models)
        with torch.no_grad():
            x_res = self.generator(z_res.float(), c.float(), 1)
        x_res = nn.functional.interpolate(x_res, size=(224, 224), mode='area')
        img_res = convert_to_images(x_res.cpu())

        reconstructed_data = dict(data=x_res, labels=labels, z_res=z_res, img_res=img_res, loss_res=loss_res)

        # return z_res, x_res, img_res, loss_res
        return reconstructed_data, stats

    @staticmethod
    def infer_label(input_gradient, num_inputs=1):
        last_weight_min = torch.argsort(torch.sum(input_gradient[-2], dim=-1), dim=-1)[:num_inputs]
        labels = last_weight_min.detach().reshape((-1,)).requires_grad_(False)
        return labels

    @staticmethod
    def ng_loss(z, # latent variable to be optimized
                loss_fn, # loss function for FL model
                input_gradient,
                labels,
                generator,
                fl_model,
                num_classes=1000,
                metric='l2',
                use_tanh=True,
                weight=None, # weight to be applied when calculating the gradient matching loss
                defense_setting=None # adaptive attack against defense
               ):

        z = torch.Tensor(z).unsqueeze(0).to(input_gradient[0].device)
        if use_tanh:
            z = z.tanh()

        c = torch.nn.functional.one_hot(labels, num_classes=num_classes).to(input_gradient[0].device)

        start_time = time.time()
        with torch.no_grad():
            x = generator(z, c.float(), 1)
        time_generator = time.time() - start_time
        x = nn.functional.interpolate(x, size=(224, 224), mode='area')

        # compute the trial gradient
        target_loss = loss_fn(fl_model(x), labels)
        trial_gradient = torch.autograd.grad(target_loss, fl_model.parameters())
        trial_gradient = [grad.detach() for grad in trial_gradient]

        # adaptive attack against defense
        # if defense_setting is not None:
        #     if 'noise' in defense_setting:
        #         trial_gradient = defense.additive_noise(trial_gradient, std=defense_setting['noise'])
        #     if 'clipping' in defense_setting:
        #         trial_gradient = defense.gradient_clipping(trial_gradient, bound=defense_setting['clipping'])
        #     if 'compression' in defense_setting:
        #         trial_gradient = defense.gradient_compression(trial_gradient, percentage=defense_setting['compression'])
        #     if 'representation' in defense_setting: # for ResNet
        #         mask = input_gradient[-2][0]!=0
        #         trial_gradient[-2] = trial_gradient[-2] * mask

        if weight is not None:
            assert len(weight) == len(trial_gradient)
        else:
            weight = [1]*len(trial_gradient)

        # calculate l2 norm
        dist = 0
        for i in range(len(trial_gradient)):
            if metric == 'l2':
                dist += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum()*weight[i]
            elif metric == 'l1':
                dist += ((trial_gradient[i] - input_gradient[i]).abs()).sum()*weight[i]
        dist /= len(trial_gradient)

        if not use_tanh:
            KLD = -0.5 * torch.sum(1 + torch.log(torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(z.squeeze(), axis=-1).pow(2) - torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2))
            dist += 0.1*KLD

        return dist.item(), time_generator



class AdamReconstructor(_BaseAttacker):
    """
    Reconstruction for BigGAN

    """
    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)
        
        self.budget = cfg_attack.optim.budget
        self.search_dim = cfg_attack.optim.search_dim
        self.use_tanh = cfg_attack.optim.use_tanh
        self.num_samples = 50 # what meaning ????
        self.weight = None
        self.use_weight = cfg_attack.optim.use_weight
        self.strategy = cfg_attack.optim.optimizer
        self.metric = cfg_attack.objective.type
        
        ##fl setting
        self.fl_loss_fn = loss_fn
        self.num_classes = cfg_attack.fl_num_class
        # self.defense_setting = defense_setting

        self.device = setup['device'] 

        self.lr = cfg_attack.optim.step_size

        self.z = torch.tensor(np.random.randn(self.search_dim), dtype=torch.float32, device=self.device, requires_grad=True)

        self.optimizer = torch.optim.Adam([self.z], betas=(0.9, 0.999), lr=self.lr)

#         self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
#                                                             milestones=[budget // 2.667, budget // 1.6,
#                                                                         budget // 1.142], gamma=0.1)

        # self.fl_setting = {'loss_fn':loss_fn, 'fl_model':model, 'num_classes':num_classes}

        if self.use_weight:
            self.weight = np.ones(62,)
            for i in range(0, 20):
                self.weight[3*i:3*(i+1)] /= 2**i

    def __repr__(self):

    #     n = "\n"
    #     return f"""Attacker (of type {self.__class__.__name__}) with settings:
    # Hyperparameter Template: {self.cfg.type}

    # Objective: {repr(self.objective)}
    # Regularizers: {(n + ' '*18).join([repr(r) for r in self.regularizers])}
    # Augmentations: {(n + ' '*18).join([repr(r) for r in self.augmentations])}

    # Optimization Setup:
    #     {(n + ' ' * 8).join([f'{key}: {val}' for key, val in self.cfg.optim.items()])}
        return f'Attacker {self.__class__.__name__} with budget {self.budget}'

    def evaluate_loss(self, z, labels, input_gradient, models):
        if len(models) == 1:

            return self.ng_loss(z=z, loss_fn=self.fl_loss_fn, input_gradient=input_gradient, metric=self.metric,
                            labels=labels, generator=self.generator, fl_model=models[0], num_classes=self.num_classes, weight=self.weight,
                            use_tanh=self.use_tanh 
                        )
        else:
            raise NotImplementedError('large batch size in GAN attack not implement noew')
        
    # def reconstruct(self, input_gradient, use_pbar=True):
    def reconstruct(self, server_payload, shared_data, generator, dryrun=False, use_pbar=True):
        
        self.generator = generator
        
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        if len(shared_data) == 1:
            input_gradient = shared_data[0]["gradients"]
        else:
            raise NotImplementedError('large batch size in GAN attack not implement noew')

        lr_rampdown_length= 0.25
        lr_rampup_length= 0.05

        labels = self.infer_label(input_gradient)
        print('Inferred label: {}'.format(labels))

        # fl_model = self.fl_setting['fl_model']

        c = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).to(input_gradient[0].device)

        pbar = tqdm(range(self.budget)) if use_pbar else range(self.budget)

        for r in pbar:
            iter_start_time = time.time()
            t = r / self.budget
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = self.lr * lr_ramp
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.zero_grad()
            # fl_model.zero_grad() # move to ng_loss()

            loss, time_list = self.evaluate_loss(z=self.z, labels=labels, input_gradient=input_gradient, models=rec_models)

            loss2back_start_time = time.time()
            loss.backward()
            time_loss2back = time.time() - loss2back_start_time
            update_start_time = time.time()
            self.optimizer.step()
#             self.scheduler.step()
            time_update = time.time() - update_start_time

            time_iter = time.time() - iter_start_time

            if use_pbar:
                pbar.set_description("Loss {:.6}, total_time {}  r_g {} r_lossback {} r_loss {} r_loss2back {} r_update {}".format(loss.item(), time_iter, time_list[0]/time_iter, time_list[1]/time_iter, time_list[2]/time_iter, time_loss2back/time_iter, time_update/time_iter))
            else:
                print("Round {} - Loss {:.6}".format(r, loss.item()))
            
            if dryrun:
                break



        z_res = self.z.detach()
        loss_res, _ = self.evaluate_loss(z_res, labels, input_gradient, models=rec_models)
        loss_res = loss_res.item()
        if self.use_tanh:
            z_res = z_res.tanh()
        z_res = z_res.unsqueeze(0)
        with torch.no_grad():
            x_res = self.generator(z_res.float(), c.float(), 1)
        x_res = nn.functional.interpolate(x_res, size=(224, 224), mode='area')
        img_res = convert_to_images(x_res.cpu())

        reconstructed_data = dict(data=x_res, labels=labels, z_res=z_res, img_res=img_res, loss_res=loss_res)

        # return z_res, x_res, img_res, loss_res
        return reconstructed_data, stats

    @staticmethod
    def infer_label(input_gradient, num_inputs=1):
        last_weight_min = torch.argsort(torch.sum(input_gradient[-2], dim=-1), dim=-1)[:num_inputs]
        labels = last_weight_min.detach().reshape((-1,)).requires_grad_(False)
        return labels

    @staticmethod
    def ng_loss(z, # latent variable to be optimized
                loss_fn, # loss function for FL model
                input_gradient,
                labels,
                generator,
                fl_model,
                num_classes=1000,
                metric='l2',
                use_tanh=True,
                weight=None, # weight to be applied when calculating the gradient matching loss
                defense_setting=None # adaptive attack against defense
               ):

        fl_model.zero_grad()

        z = z.unsqueeze(0)
        if use_tanh:
            z = z.tanh()

        c = torch.nn.functional.one_hot(labels, num_classes=num_classes).to(input_gradient[0].device)

        # why use no_grad here, how to calculate the second-order gradient of z
        g_start_time = time.time()
        # with torch.no_grad():
        x = generator(z, c.float(), 1)
        time_generator = time.time() - g_start_time

        x = nn.functional.interpolate(x, size=(224, 224), mode='area')

        # compute the trial gradient
        g_fisrt_gradient_start_time = time.time() 
        target_loss = loss_fn(fl_model(x), labels)
        trial_gradient = torch.autograd.grad(target_loss, fl_model.parameters(), create_graph=True)
        time_first_gradient = time.time() - g_fisrt_gradient_start_time
        # trial_gradient = [grad.detach() for grad in trial_gradient]
        trial_gradient = [grad for grad in trial_gradient]

        # adaptive attack against defense
        # if defense_setting is not None:
        #     if 'noise' in defense_setting:
        #         trial_gradient = defense.additive_noise(trial_gradient, std=defense_setting['noise'])
        #     if 'clipping' in defense_setting:
        #         trial_gradient = defense.gradient_clipping(trial_gradient, bound=defense_setting['clipping'])
        #     if 'compression' in defense_setting:
        #         trial_gradient = defense.gradient_compression(trial_gradient, percentage=defense_setting['compression'])
        #     if 'representation' in defense_setting: # for ResNet
        #         mask = input_gradient[-2][0]!=0
        #         trial_gradient[-2] = trial_gradient[-2] * mask

        if weight is not None:
            assert len(weight) == len(trial_gradient)
        else:
            weight = [1]*len(trial_gradient)

        # calculate l2 norm
        dist = 0
        loss_start_time = time.time()
        for i in range(len(trial_gradient)):
            if metric == 'l2':
                dist += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum()*weight[i]
            elif metric == 'l1':
                dist += ((trial_gradient[i] - input_gradient[i]).abs()).sum()*weight[i]
        dist /= len(trial_gradient)

        if not use_tanh:
            KLD = -0.5 * torch.sum(1 + torch.log(torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(z.squeeze(), axis=-1).pow(2) - torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2))
            dist += 0.1*KLD
        time_loss = time.time() - loss_start_time
        # return dist, time_generator
        return dist, [time_generator, time_first_gradient, time_loss]


    @staticmethod
    def gradient_loss(z, # latent variable to be optimized
                loss_fn, # loss function for FL model
                input_gradient,
                labels,
                generator,
                fl_model,
                num_classes=1000,
                metric='l2',
                use_tanh=True,
                weight=None, # weight to be applied when calculating the gradient matching loss
                defense_setting=None # adaptive attack against defense
               ):

        z = z.unsqueeze(0)
        if use_tanh:
            z = z.tanh()

        c = torch.nn.functional.one_hot(labels, num_classes=num_classes).to(input_gradient[0].device)

        # why use no_grad here, how to calculate the second-order gradient of z
        g_start_time = time.time()
        # with torch.no_grad():
        x = generator(z, c.float(), 1)
        time_generator = time.time() - g_start_time

        x = nn.functional.interpolate(x, size=(224, 224), mode='area')

        # compute the trial gradient
        g_fisrt_gradient_start_time = time.time() 
        target_loss, _, _ = loss_fn(fl_model(x), labels)
        trial_gradient = torch.autograd.grad(target_loss, fl_model.parameters(), create_graph=True)
        time_first_gradient = time.time() - g_fisrt_gradient_start_time
        # trial_gradient = [grad.detach() for grad in trial_gradient]
        trial_gradient = [grad for grad in trial_gradient]

        # adaptive attack against defense
        # if defense_setting is not None:
        #     if 'noise' in defense_setting:
        #         trial_gradient = defense.additive_noise(trial_gradient, std=defense_setting['noise'])
        #     if 'clipping' in defense_setting:
        #         trial_gradient = defense.gradient_clipping(trial_gradient, bound=defense_setting['clipping'])
        #     if 'compression' in defense_setting:
        #         trial_gradient = defense.gradient_compression(trial_gradient, percentage=defense_setting['compression'])
        #     if 'representation' in defense_setting: # for ResNet
        #         mask = input_gradient[-2][0]!=0
        #         trial_gradient[-2] = trial_gradient[-2] * mask

        if weight is not None:
            assert len(weight) == len(trial_gradient)
        else:
            weight = [1]*len(trial_gradient)

        # calculate l2 norm
        dist = 0
        loss_start_time = time.time()
        for i in range(len(trial_gradient)):
            if metric == 'l2':
                dist += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum()*weight[i]
            elif metric == 'l1':
                dist += ((trial_gradient[i] - input_gradient[i]).abs()).sum()*weight[i]
        dist /= len(trial_gradient)

        if not use_tanh:
            KLD = -0.5 * torch.sum(1 + torch.log(torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(z.squeeze(), axis=-1).pow(2) - torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2))
            dist += 0.1*KLD
        time_loss = time.time() - loss_start_time
        # return dist, time_generator
        return dist, [time_generator, time_first_gradient, time_loss]




class AdamReconstructorNoGan():
    """
    Reconstruction for BigGAN

    """
    def __init__(self, fl_model, generator, loss_fn, num_classes=1000, search_dim=(128,), lr=0.1, strategy='Adam', budget=2500, use_tanh=True, use_weight=False, defense_setting=None):

        self.generator = generator
        self.budget = budget
        self.search_dim = search_dim
        self.use_tanh = use_tanh
        self.num_samples = 50
        self.weight = None
        self.defense_setting = defense_setting

        self.x = None
        self.optimizer = None

        self.device = device = next(fl_model.parameters()).device

        self.lr = lr


#         self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
#                                                             milestones=[budget // 2.667, budget // 1.6,
#                                                                         budget // 1.142], gamma=0.1)

        self.fl_setting = {'loss_fn':loss_fn, 'fl_model':fl_model, 'num_classes':num_classes}

        if use_weight:
            self.weight = np.ones(62,)
            for i in range(0, 20):
                self.weight[3*i:3*(i+1)] /= 2**i


    def evaluate_loss(self, x, labels, input_gradient):
        return self.ng_loss(x=x, input_gradient=input_gradient, metric='l2',
                        labels=labels, generator=self.generator, weight=self.weight,
                        use_tanh=self.use_tanh, defense_setting=self.defense_setting, **self.fl_setting
                       )

    def reconstruct(self, input_gradient, use_pbar=True):
        lr_rampdown_length= 0.25
        lr_rampup_length= 0.05

        labels = self.infer_label(input_gradient)
        print('Inferred label: {}'.format(labels))


        fl_model = self.fl_setting['fl_model']

        c = torch.nn.functional.one_hot(labels, num_classes=self.fl_setting['num_classes']).to(input_gradient[0].device)

        z = torch.tensor(np.random.randn(self.search_dim[0]), dtype=torch.float32, device=self.device)
        z = z.unsqueeze(0)
        self.x = self.generator(z, c.float(), 1)
        self.x = nn.functional.interpolate(self.x, size=(224, 224), mode='area')
        # self.x.requires_grad = True
        self.x = self.x.detach().clone().to(self.device)
        print(self.x.is_leaf)

        self.optimizer = torch.optim.Adam([self.x], betas=(0.9, 0.999), lr=self.lr)

        pbar = tqdm(range(self.budget)) if use_pbar else range(self.budget)

        for r in pbar:
            iter_start_time = time.time()
            t = r / self.budget
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = self.lr * lr_ramp
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.zero_grad()
            fl_model.zero_grad()

            loss = self.evaluate_loss(x=self.x, labels=labels, input_gradient=input_gradient)

            loss.backward()
            self.optimizer.step()
#             self.scheduler.step()

            time_iter = time.time() - iter_start_time

            if use_pbar:
                pbar.set_description("Loss {:.6}, total_time {} ".format(loss.item(), time_iter))
                # pbar.set_description("Loss {:.6}, total_time {}  r_g {} r_lossback {} r_loss {} r_loss2back {} r_update {}".format(loss.item(), time_iter, time_list[0]/time_iter, time_list[1]/time_iter, time_list[2]/time_iter, time_loss2back/time_iter, time_update/time_iter))
            else:
                print("Round {} - Loss {:.6}".format(r, loss.item()))

            if r % 10 == 0:
                imgs = convert_to_images(self.x.cpu())
                for img in imgs:
                    img.save(f'/home/zx/Gitrepo/GGL/out/adam_no_gan_imgs/{r}.png')
        
        loss_res, _ = self.evaluate_loss(self.x, labels, input_gradient).item()
        img_res = convert_to_images(self.x.cpu())

        return None, self.x, img_res, loss_res

    @staticmethod
    def infer_label(input_gradient, num_inputs=1):
        last_weight_min = torch.argsort(torch.sum(input_gradient[-2], dim=-1), dim=-1)[:num_inputs]
        labels = last_weight_min.detach().reshape((-1,)).requires_grad_(False)
        return labels


    @staticmethod
    def ng_loss(x, # latent variable to be optimized
                loss_fn, # loss function for FL model
                input_gradient,
                labels,
                generator,
                fl_model,
                num_classes=1000,
                metric='l2',
                use_tanh=True,
                weight=None, # weight to be applied when calculating the gradient matching loss
                defense_setting=None # adaptive attack against defense
               ):


        # compute the trial gradient
        g_fisrt_gradient_start_time = time.time() 
        target_loss, _, _ = loss_fn(fl_model(x), labels)
        trial_gradient = torch.autograd.grad(target_loss, fl_model.parameters(), create_graph=True)
        time_first_gradient = time.time() - g_fisrt_gradient_start_time
        # trial_gradient = [grad.detach() for grad in trial_gradient]
        trial_gradient = [grad for grad in trial_gradient]

        # adaptive attack against defense
        # if defense_setting is not None:
        #     if 'noise' in defense_setting:
        #         trial_gradient = defense.additive_noise(trial_gradient, std=defense_setting['noise'])
        #     if 'clipping' in defense_setting:
        #         trial_gradient = defense.gradient_clipping(trial_gradient, bound=defense_setting['clipping'])
        #     if 'compression' in defense_setting:
        #         trial_gradient = defense.gradient_compression(trial_gradient, percentage=defense_setting['compression'])
        #     if 'representation' in defense_setting: # for ResNet
        #         mask = input_gradient[-2][0]!=0
        #         trial_gradient[-2] = trial_gradient[-2] * mask

        if weight is not None:
            assert len(weight) == len(trial_gradient)
        else:
            weight = [1]*len(trial_gradient)

        # calculate l2 norm
        dist = 0
        loss_start_time = time.time()
        for i in range(len(trial_gradient)):
            if metric == 'l2':
                dist += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum()*weight[i]
            elif metric == 'l1':
                dist += ((trial_gradient[i] - input_gradient[i]).abs()).sum()*weight[i]
        dist /= len(trial_gradient)

        # if not use_tanh:
        #     KLD = -0.5 * torch.sum(1 + torch.log(torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(z.squeeze(), axis=-1).pow(2) - torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2))
        #     dist += 0.1*KLD
        # time_loss = time.time() - loss_start_time
        # return dist, time_generator
        return dist