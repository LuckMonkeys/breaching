try:
    import breaching
except ModuleNotFoundError:
    # You only really need this safety net if you want to run these notebooks directly in the examples directory
    # Don't worry about this if you installed the package or moved the notebook to the main directory.
    import os; os.chdir("..")
    import breaching
    
import torch

import logging, sys
from pathlib import Path
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()
from tqdm import tqdm
import torch, torchvision


cfg = breaching.get_config(overrides=[])
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

for idx in tqdm(idxs[:num_samples]):

    cfg.case.data.partition="unique-class"
    cfg.case.user.user_idx = idx
    cfg.case.model='resnet18'
    cfg.attack.optim.callback=100
    # cfg.case.server.pretrained = False

    # cfg.case.data.examples_from_split='train'
    cfg.case.data.examples_from_split='validation'

    # cfg.attack.impl.sparse = 0.01
    # cfg.attack.save.out_dir = f'/home/zx/data/GitRepo/breaching/out/psnr_plot/{idx}'
    cfg.attack.save.out_dir = f'/home/zx/data/GitRepo/breaching/out/from_same_class/{idx}'

    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    breaching.utils.overview(server, user, attacker)
    # print(model)

    server_payload = server.distribute_payload()
    shared_data, true_user_data = user.compute_local_updates(server_payload)

    # user.plot(true_user_data)

    #load same class image from start 1
    same_class_dir = '/home/zx/data/GitRepo/breaching/rec_datasets/normal/1/hq'
    s_img_path = same_class_dir + f'/{idx}.png'
    s_img_th = torchvision.io.read_image(s_img_path).to(torch.float32) / 255.0
    
    # apply augmentation
    s_img_th = valid_trans_fn(s_img_th)

    #unsqueeze to add batch dimension
    s_img_th = s_img_th.unsqueeze(0)

    # print(s_img_th.shape)
    # s_img_th = torch.randn((1, 3, 224, 224))
    
    reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun, initial_data=s_img_th)
    # break








