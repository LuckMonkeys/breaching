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


def mirnet_super_resolution(img_tensor, dm=0, ds=1,tile=None, tile_overlap=32):
    # img_tensor = torch.clamp(img_tensor * ds + dm, 0, 1)
    img_tensor = img_tensor * ds + dm
    device = img_tensor.device
    weights = '/home/zx/Gitrepo/breaching/breaching/attacks/accelarate/super_resolution/pre_trained/sr_x4.pth'

    load_arch = run_path(os.path.join('breaching', 'attacks', 'accelarate', 'super_resolution', 'mirnet_v2_arch.py'))

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


def swinir_super_resolution(img_tensor, dm=0, ds=1, tile=None, large_model=False, scale=4, window_size=8):


    def test(img_lq, model, tile, window_size):
        if tile is None:
            # test the image as a whole
            output = model(img_lq)
        else:
            # test the image tile by tile
            b, c, h, w = img_lq.size()
            tile = min(tile, h, w)
            assert tile % window_size == 0, "tile size should be a multiple of window_size"
            tile_overlap = tile_overlap
            sf = scale

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
            output = E.div_(W)

        return output
        

    img_tensor = img_tensor * ds + dm
    device = img_tensor.device
    weights = '/home/zx/Gitrepo/breaching/breaching/attacks/accelarate/super_resolution/pre_trained/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'

    pretrained_model = torch.load(weights)

    from .swinir import SwinIR as net
    if not large_model:
    # use 'nearest+conv' to avoid block artifacts
        model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
    else:
        # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
        model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                    num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')

    param_key_g = 'params_ema'
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    model.to(device)
    model.eval()

    _, _, h_old, w_old = img_tensor.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
    img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [3])], 3)[:, :, :, :w_old + w_pad]
    output = test(img_lq=img_tensor, model=model, tile=tile, window_size=window_size)
    output = output[..., :h_old * scale, :w_old * scale]

    #rescale to origin size
    output = torchvision.transforms.Resize((h_old, w_old))(output)
    # normalize
    output = (output - dm) / ds

    return output