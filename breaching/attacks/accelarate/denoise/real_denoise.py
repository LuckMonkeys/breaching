"""
## Learning Enriched Features for Fast Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
## https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/
"""
##--------------------------------------------------------------
##------- Demo file to test MIRNet-V2 on your own images---------
## Example usage on directory containing several images:   python demo.py --task real_denoising --input_dir './demo/degraded/' --result_dir './demo/restored/'
## Example usage on a image directly: python demo.py --task real_denoising --input_dir './demo/degraded/noisy.png' --result_dir './demo/restored/'
## Example usage with tile option on a large image: python demo.py --task real_denoising --input_dir './demo/degraded/noisy.png' --result_dir './demo/restored/' --tile 720 --tile_overlap 32
##--------------------------------------------------------------

from re import I
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
from pdb import set_trace as stx
import numpy as np



# Get model weights and parameters
# parameters = {
#     'inp_channels':3,
#     'out_channels':3, 
#     'n_feat':80,
#     'chan_factor':1.5,
#     'n_RRG':4,
#     'n_MRB':2,
#     'height':3,
#     'width':2,
#     'bias':False,
#     'scale':1,
#     'task': task
#     }

parameters = {
    'inp_channels':3,
    'out_channels':3, 
    'n_feat':80,
    'chan_factor':1.5,
    'n_RRG':4,
    'n_MRB':2,
    'height':3,
    'width':2,
    'bias':False,
    'scale':1,
    'task':'real_denoising' 
    }


# weights, parameters = get_weights_and_parameters(task, parameters)


def mirnet_denoise(img_tensor, dm=0, ds=1,tile=None, tile_overlap=32):
    # img_tensor = torch.clamp(img_tensor * ds + dm, 0, 1)
    img_tensor = img_tensor * ds + dm
    device = img_tensor.device
    weights = '/home/zx/Gitrepo/breaching/breaching/attacks/accelarate/denoise/pre_trained/real_denoising.pth'

    load_arch = run_path(os.path.join('breaching', 'attacks', 'accelarate', 'denoise', 'mirnet_v2_arch.py'))

    model = load_arch['MIRNet_v2'](**parameters)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['params'])
    model.eval()

    img_multiple_of = 4

    # print(f"\n ==> Running {task} with weights {weights}\n ")
    # print('start denoise')


    with torch.inference_mode():
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        # img = load_img(file_)

        # input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

        input_ = img_tensor

        # Pad the input if not_multiple_of 8
        height,width = input_.shape[2], input_.shape[3]
        H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        if tile is None:
            ## Testing on the original resolution image
            restored = model(input_)
        else:
            # test the image tile by tile
            b, c, h, w = input_.shape
            tile = min(tile, h, w)
            assert tile % 4 == 0, "tile size should be multiple of 4"

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h, w).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                    W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
            restored = E.div_(W)

        # restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:,:,:height,:width]
        # restored = torch.clamp((restored - dm) / ds, 0, 1)
        restored = (restored - dm) / ds

        #     restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        #     restored = img_as_ubyte(restored[0])

        #     f = os.path.splitext(os.path.split(file_)[-1])[0]
        #     # stx()

        #     save_img((os.path.join(out_dir, f+'.png')), restored)

        # print(f"\nRestored images are saved at {out_dir}")
    return restored




def sucnet_denoise(img_tensor, dm=0, ds=1):
    from .scunet import SCUNet as net

    img_tensor = img_tensor * ds + dm
    device = img_tensor.device

    n_channels = 3
    model_path = '/home/zx/Gitrepo/breaching/breaching/attacks/accelarate/denoise/pre_trained/scunet_color_real_psnr.pth'

    model = net(in_nc=n_channels,config=[4,4,4,4,4,4,4],dim=64)

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    denoised_img = model(img_tensor)
    denoised_img = (denoised_img - dm) / ds

    return denoised_img

