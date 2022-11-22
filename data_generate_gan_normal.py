"""This script computes a suite of benchmark numbers for the given attack.


The arguments from the default config carry over here.
"""

import hydra
from omegaconf import OmegaConf

import datetime
import time
import logging

import breaching

import os
import torch

os.environ["HYDRA_FULL_ERROR"] = "0"
log = logging.getLogger(__name__)


def save_img(path, img_tensor, is_batch=True, dm=0, ds=1):
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

def main_process(process_idx, local_group_size, cfg, num_trials=100):
    """This function controls the central routine."""
    total_time = time.time()  # Rough time measurements here
    setup = breaching.utils.system_startup(process_idx, local_group_size, cfg)
    
    # TODO: why directly override case.data=ImageNet not work??
    # print(cfg.attack, '--------------------\n', cfg.case.data) 
    # exit(0)
    model, loss_fn = breaching.cases.construct_model(cfg.case.model, cfg.case.data, cfg.case.server.pretrained)

    if cfg.num_trials is not None:
        num_trials = cfg.num_trials

    ## set num_clients = 1000
    cfg.case.data.default_clients = 1000

    server = breaching.cases.construct_server(model, loss_fn, cfg.case, setup)
    model = server.vet_model(model)
    attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg.attack, setup)
    if cfg.case.user.user_idx is not None:
        print("The argument user_idx is disregarded during the benchmark. Data selection is fixed.")
    log.info(
        f"Partitioning is set to {cfg.case.data.partition}. Make sure there exist {num_trials} users in this scheme."
    )

    if cfg.case.user.user_idx <= 0:
        cfg.case.user.user_idx = -1
    run = 0

    # test on selected images
    # selected_idxs = [2, 10, 11, 12, 13, 15, 16, 19, 20, 22, 24, 28, 31, 36, 38, 40, 42, 53, 56, 65, 70, 75, 76, 77, 79, 80, 83, 84, 90, 92, 95, 96, 98, 100, 106, 107, 109, 113, 116, 117, 121, 123, 125, 127, 128, 131, 132, 136, 137, 138, 139, 140, 142, 143, 145, 150, 156, 158, 169, 177, 210, 214, 215, 216, 227, 229, 239, 247, 249, 253, 260, 262, 274, 275, 276, 280, 283, 290, 296, 298, 299, 301, 306, 309, 313, 316, 317, 318, 321, 322, 323, 324, 327, 328, 329, 333, 334, 337, 338, 339, 342, 343, 349, 351, 354, 362, 363, 364, 365, 366, 368, 376, 377, 384, 387, 389, 392, 393, 394, 398, 399, 400, 401, 404, 405, 410, 412, 415, 417, 420, 422, 430, 434, 435, 437, 445, 446, 448, 451, 457, 459, 467, 468, 470, 471, 475, 478, 480, 483, 489, 490, 495, 503, 507, 512, 514, 520, 522, 529, 530, 534, 535, 546, 548, 552, 553, 556, 560, 564, 565, 566, 569, 571, 572, 573, 575, 576, 580, 590, 592, 594, 597, 601, 604, 607, 611, 614, 616, 617, 621, 628, 632, 633, 642, 644, 645, 646, 647, 649, 651, 659, 661, 665, 666, 668, 669, 676, 679, 680, 682, 684, 685, 686, 688, 691, 694, 695, 697, 699, 701, 703, 704, 708, 714, 715, 719, 723, 727, 736, 741, 745, 750, 751, 753, 757, 765, 768, 770, 771, 772, 773, 778, 779, 780, 783, 789, 790, 791, 792, 796, 798, 802, 805, 807, 808, 812, 820, 822, 826, 845, 850, 852, 857, 863, 869, 870, 872, 875, 888, 890, 891, 901, 917, 918, 919, 924, 927, 933, 934, 936, 941, 946, 949, 952, 956, 957, 958, 965, 967, 969, 971, 979, 980, 981, 984, 986, 989, 991, 993, 994, 997, 998]
    selected_idxs = [0]
    print('choose specific img, len ', len(selected_idxs))
    # exit(0)

    if len(selected_idxs) > 0:
        num_trials = len(selected_idxs)
    
    while run < num_trials:
        local_time = time.time()
        # Select data that has not been seen before:
        if len(selected_idxs) > 0:
            cfg.case.user.user_idx = selected_idxs[run]
        else:
            cfg.case.user.user_idx += 1
        # try:
        user = breaching.cases.construct_user(model, loss_fn, cfg.case, setup)

        # except ValueError:
        #     log.info("Cannot find other valid users. Finishing benchmark.")
        #     break
        if cfg.case.data.modality == "text":
            dshape = user.dataloader.dataset[0]["input_ids"].shape
            data_shape_mismatch = any([d != d_ref for d, d_ref in zip(dshape, cfg.case.data.shape)])
        else:
            data_shape_mismatch = False  # Handled by preprocessing for images
        if len(user.dataloader.dataset) < user.num_data_points or data_shape_mismatch:
            log.info(f"Skipping user {user.user_idx} (has not enough data or data shape mismatch).")
        else:
            log.info(f"Now evaluating user {user.user_idx} in trial {run}.")
            run += 1
            # Run exchange
            shared_user_data, payloads, true_user_data = server.run_protocol(user)

            ## Get dm and ds
            metadata = payloads[0]["metadata"]
            if hasattr(metadata, "mean"):
                dm = torch.as_tensor(metadata.mean, **setup)[None, :, None, None]
                ds = torch.as_tensor(metadata.std, **setup)[None, :, None, None]
            else:
                dm, ds = torch.tensor(0, **setup), torch.tensor(1, **setup)

            ## Save true_user_data
            # cfg.attack.save.out_dir = '/home/zx/Gitrepo/breaching/rec_datasets/scale_4'
            cfg.attack.save.idx =  cfg.case.user.user_idx
            save_img(cfg.attack.save.out_dir + f'/hq/{cfg.attack.save.idx}.png', true_user_data['data'].clone().detach(), dm=dm, ds=ds)    

            ## Set out_dir in reconstruction
            # Evaluate attack:
            # try:

            if cfg.attack.attack_type == 'optimization_GAN_CMA':
                from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                                    save_as_images, display_in_terminal, convert_to_images)
                generator= BigGAN.from_pretrained('biggan-deep-256').to(setup['device'])

                reconstruction, stats = attacker.reconstruct(
                    payloads, shared_user_data, generator=generator, dryrun=cfg.dryrun
                )
            else:

                reconstruction, stats = attacker.reconstruct(
                    payloads, shared_user_data, server.secrets, dryrun=cfg.dryrun
                )

            # Run the full set of metrics:
            # metrics = breaching.analysis.report(
            #     reconstruction,
            #     true_user_data,
            #     payloads,
            #     server.model,
            #     order_batch=True,
            #     compute_full_iip=True,
            #     compute_rpsnr=True,
            #     compute_ssim=True,
            #     cfg_case=cfg.case,
            #     setup=setup,
            # )
            # # Add query metrics
            # metrics["queries"] = user.counted_queries

            # # Save local summary:
            # breaching.utils.save_summary(cfg, metrics, stats, time.time() - local_time, original_cwd=False)
            # overall_metrics.append(metrics)
            # Save recovered data:
            # if cfg.save_reconstruction:
            #     breaching.utils.save_reconstruction(reconstruction, payloads, true_user_data, cfg)
            if cfg.dryrun:
                break
            # except Exception as e:  # noqa # yeah we're that close to the deadlines
            #     log.info(f"Trial {run} broke down with error {e}.")

    # Compute average statistics:
    # average_metrics = breaching.utils.avg_n_dicts(overall_metrics)

    # Save global summary:
    # breaching.utils.save_summary(
    #     cfg, average_metrics, stats, time.time() - total_time, original_cwd=True, table_name="BENCHMARK_breach"
    # )


@hydra.main(version_base="1.1", config_path="breaching/config", config_name="cfg")
def main_launcher(cfg):
    """This is boiler-plate code for the launcher."""

    log.info("--------------------------------------------------------------")
    log.info("-----Launching federating learning breach experiment! --------")

    launch_time = time.time()
    if cfg.seed is None:
        cfg.seed = 233  # The benchmark seed is fixed by default!

    log.info(OmegaConf.to_yaml(cfg))
    breaching.utils.initialize_multiprocess_log(cfg)  # manually save log configuration
    main_process(0, 1, cfg)

    log.info("-------------------------------------------------------------")
    log.info(
        f"Finished computations with total train time: " f"{str(datetime.timedelta(seconds=time.time() - launch_time))}"
    )
    log.info("-----------------Job finished.-------------------------------")


if __name__ == "__main__":
    main_launcher()
