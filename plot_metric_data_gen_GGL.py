try:
    import breaching
except ModuleNotFoundError:
    # You only really need this safety net if you want to run these notebooks directly in the examples directory
    # Don't worry about this if you installed the package or moved the notebook to the main directory.
    import os; os.chdir("..")
    import breaching
    
import os
import torch

import logging, sys
from pathlib import Path
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()
from tqdm import tqdm
import torch, torchvision


cfg = breaching.get_config(overrides=["attack=GAN"])
# cfg = breaching.get_config(overrides=['attack=hybrid_ac'])
          
device = torch.device(f'cuda:7') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))


hq_dir = '/home/zx/data/GitRepo/breaching/rec_datasets/normal/0/hq'

filenames = list(Path(hq_dir).iterdir())
idxs = [int(name.name.split('.')[0]) for name in filenames]

num_samples = 10
valid_trans = {'Resize':{'size':256}, 'CenterCrop':{'size':224}, 'Normalize':{'mean':(0.485, 0.456, 0.406), 'std':(0.229, 0.224, 0.225)}}

trans_list = []
#augmentation
for key, value in valid_trans.items():    
    transform = getattr(torchvision.transforms, key)(**value)
    trans_list.append(transform)
valid_trans_fn = torchvision.transforms.Compose(trans_list)


print('load model')
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                    save_as_images, display_in_terminal, convert_to_images)
generator= BigGAN.from_pretrained('biggan-deep-256').to(setup['device'])

for idx in tqdm(idxs[:num_samples], desc='sample'):

    cfg.case.data.partition="unique-class"
    cfg.case.user.user_idx = idx
    cfg.case.model='resnet18'
    cfg.attack.optim.callback=100
    cfg.case.server.pretrained = False

    cfg.attack.layer_weights = None

    
    cfg.attack.optim.optimizer='CMA'
    cfg.attack.optim.budget=500
    cfg.attack.optim.num_samples=50
    cfg.attack.attack_type='optimization_GAN_CMA'
    cfg.attack.optim.patched=None
    
    

    # cfg.case.data.examples_from_split='train'
    cfg.case.data.examples_from_split='validation'

    base_folder = f'/home/zx/data/GitRepo/breaching/out/GAN/{cfg.attack.optim.optimizer}'
    # cfg.attack.out_dir = os.path.join(base_folder, f'{idx}')
    cfg.attack.out_dir = os.path.join(base_folder, f'num_{cfg.attack.optim.num_samples}', f'{idx}')

    if not os.path.exists(cfg.attack.out_dir):
        os.makedirs(cfg.attack.out_dir)
    

    print('create user and server')
    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    breaching.utils.overview(server, user, attacker)
    # print(model)


    print('compute payload')
    server_payload = server.distribute_payload()
    shared_data, true_user_data = user.compute_local_updates(server_payload)

    print('start attack')
    reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], generator=generator, dryrun=False)
    # break








