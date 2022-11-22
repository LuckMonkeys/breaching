try:
    import breaching
except ModuleNotFoundError:
    # You only really need this safety net if you want to run these notebooks directly in the examples directory
    # Don't worry about this if you installed the package or moved the notebook to the main directory.
    import os; os.chdir("..")
    import breaching
    
import torch
import argparse
import os
import pickle


parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')

parser.add_argument('--class_cond', default=False, type=str, help='Whether use class cond model')
parser.add_argument('--max_iterations', default=10, required=True, type=int, help='The grad attack interations')
parser.add_argument('--timestep_respacing', default="100", required=True, type=str, help='The respacing timesteps')

opt = parser.parse_args()

import logging, sys
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()

cfg = breaching.get_config(overrides=['attack=invertinggradients_diffusion'])
          
          
          
#change setting for multi-run
if opt.class_cond:
    cfg.attack.diffusion.model_path = "/home/zx/data/GitRepo/breaching/breaching/attacks/accelarate/guided_diffusion/models/256x256_diffusion.pt"
    cfg.attack.diffusion.class_cond = True

cfg.attack.optim.max_iterations = opt.max_iterations
cfg.attack.diffusion.timestep_respacing = opt.timestep_respacing

cfg.attack.save.out_dir = os.path.join("/home/zx/data/GitRepo/breaching/out/diffusion", f"class_cond_{opt.class_cond}_iteration_{cfg.attack.optim.max_iterations}_respacing_{cfg.attack.diffusion.timestep_respacing}")

if not os.path.exists(cfg.attack.save.out_dir):
   os.makedirs(cfg.attack.save.out_dir) 

          
device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))

cfg.case.data.partition="unique-class"
cfg.case.user.user_idx = 24
cfg.case.model='resnet18'
# cfg.case.server.pretrained = False
# cfg.case.data.examples_from_split='train'
cfg.case.data.examples_from_split='validation'
cfg.dryrun=False

print('constructing user and sever')
user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)

print('constructing attacker')
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
breaching.utils.overview(server, user, attacker)
# print(model)

server_payload = server.distribute_payload()
shared_data, true_user_data = user.compute_local_updates(server_payload)

reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun)

metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload], 
                                    server.model, order_batch=True, compute_full_iip=False, 
                                    cfg_case=cfg.case, setup=setup)

metric_file = os.path.join(cfg.attack.save.out_dir, 'metric.pkl')
with open(metric_file, "wb") as f:
    pickle.dump(metrics, f)


