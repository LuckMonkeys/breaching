try:
    import breaching
except ModuleNotFoundError:
    # You only really need this safety net if you want to run these notebooks directly in the examples directory
    # Don't worry about this if you installed the package or moved the notebook to the main directory.
    import os; os.chdir("..")
    import breaching
    
import torch

import logging, sys
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()


cfg = breaching.get_config(overrides=["case=11_single_celebahq_gender"])
# cfg = breaching.get_config(overrides=['attack=hybrid_ac'])
          
device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
setup

# ### Modify config options here

# You can use `.attribute` access to modify any of these configurations for the attack, or the case:

# cfg.case.data.partition="unique-class"
# cfg.case.data.partition="balanced"
# cfg.case.user.user_idx = 24
# cfg.case.model='resnet18'
cfg.case.server.pretrained = True
# cfg.case.data.examples_from_split='train'
cfg.case.data.examples_from_split='valid'

cfg.attack.save.out_dir = "/home/zx/breaching/out/celeba_hq/"

# ### Instantiate all parties
# The following lines generate "server, "user" and "attacker" objects and print an overview of their configurations.

user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
breaching.utils.overview(server, user, attacker)
# print(model)

# ### Simulate an attacked FL protocol

# This exchange is a simulation of a single query in a federated learning protocol. The server sends out a `server_payload` and the user computes an update based on their private local data. This user update is `shared_data` and contains, for example, the parameter gradient of the model in the simplest case. `true_user_data` is also returned by `.compute_local_updates`, but of course not forwarded to the server or attacker and only used for (our) analysis.

server_payload = server.distribute_payload()
shared_data, true_user_data = user.compute_local_updates(server_payload)

# user.plot(true_user_data)

# ### Reconstruct user data:

# Now we launch the attack, reconstructing user data based on only the `server_payload` and the `shared_data`. 
# You can interrupt the computation early to see a partial solution.

reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun)

# Next we'll evaluate metrics, comparing the `reconstructed_user_data` to the `true_user_data`.

# metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload], 
                                    # server.model, order_batch=True, compute_full_iip=False, 
                                    # cfg_case=cfg.case, setup=setup)

# And finally, we also plot the reconstructed data:

# user.plot(reconstructed_user_data)
