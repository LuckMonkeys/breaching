# %%
try:
    import breaching
except ModuleNotFoundError:
    # You only really need this safety net if you want to run these notebooks directly in the examples directory
    # Don't worry about this if you installed the package or moved the notebook to the main directory.
    import os; os.chdir("/home/zx/Gitrepo/breaching")
    import breaching
    
import torch


# Redirects logs directly into the jupyter notebook
import logging, sys
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()


cfg = breaching.get_config(overrides=["attack=hybrid", "case=1_single_image_small", "case.model=cnn6"])
          
device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
setup
# %%
# print(cfg)
cfg.case.user.user_idx = 1

cfg.case.user.num_data_points = 1 # The attack is designed for only one data point
cfg.attack.optim.max_iterations = 24_000


user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
server.loss = torch.jit.script(torch.nn.MSELoss())
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
# breaching.utils.overview(server, user, attacker)

server_payload = server.distribute_payload()
shared_data, true_user_data = user.compute_local_updates(server_payload)
# %%

user.plot(true_user_data)



# %%
reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, {}, dryrun=cfg.dryrun)

# # %% [markdown]
# # Next we'll evaluate metrics, comparing the `reconstructed_user_data` to the `true_user_data`.

# # %%
# metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload], 
#                                     server.model, order_batch=True, compute_full_iip=False, 
#                                     cfg_case=cfg.case, setup=setup)

## %%


# user.plot(true_user_data)



 # %%

user.plot(reconstructed_user_data)


# # %%
# print(cfg.case.user.user_type)


# %%

# type(server.loss)
