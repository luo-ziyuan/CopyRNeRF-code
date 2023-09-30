import os, sys
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from run_copyrnerf_helpers import *

from load_llff import load_llff_data, get_random_poses_llff, get_fix_poses_llff
from load_blender import load_blender_data, get_random_poses_blender, get_render_poses
from encoder import Encoder, Encoder_MLP, Encoder_Tri_MLP_add, Encoder_Tri_MLP_f
from decoder import Decoder_sigmoid
from vgg_loss import VGGLoss
from gaussian_noise import addNoise
from resize import Resize
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark=True
np.random.seed(0)
DEBUG = False


def batchify(fn, encoder_fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    if encoder_fn is not None:
        def ret(inputs, message):
            re = []
            for i in range(0, inputs.shape[0], chunk):
                # print('chunk', i)
                rgbsigma = fn(inputs[i:i + chunk])
                re.append(encoder_fn(inputs[i:i + chunk], rgbsigma, message))
            return torch.cat(re, 0)
    else:
        def ret(inputs, message):
            re = []
            for i in range(0, inputs.shape[0], chunk):
                # print('chunk', i)
                rgbsigma = fn(inputs[i:i + chunk])
                re.append(torch.cat((rgbsigma[:, 0:3], rgbsigma[:, 0:3], rgbsigma[:, 3:4]), -1))
                # re.append(encoder_fn(inputs[i:i + chunk], rgbsigma, message))
            return torch.cat(re, 0)

    return ret


def run_network(inputs, message, viewdirs, fn, encoder_fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, encoder_fn, netchunk)(embedded, message)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, encoder, message, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    # chunk = rays_flat.shape[0]
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], encoder, message, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, encoder, message, chunk=1024 * 32, rays=None, c2w=None,
           ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    # if depths_map is not None:
    #     depths_map = torch.reshape(depths_map, [-1, 1]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, encoder, message, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'target_rgb_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_whole_image(render_poses, message, hwf, K, chunk, patch_size, render_kwargs, device, savedir=None,
                render_factor=0,
                encoder=None, decoder=None):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    rgbs_target = []
    decoded_messages = []
    decoded_messages_bit = []
    message_mse = []
    message_BER = []
    image_mse = []
    image_PSNR = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()

        message_i =message[1,:]

        pose = c2w[:3, :4].to(device)
        rays_o, rays_d = get_rays(H, W, K, pose)


        batch_rays_patch = torch.stack([rays_o, rays_d], 0)

        rgb, disp, _, rgb_target, _ = render(H, W, K, encoder, message_i.unsqueeze(0), chunk=chunk,
                                             rays=batch_rays_patch,
                                             **render_kwargs)

        img_mse = img2mse(rgb, rgb_target)
        image_mse.append(img_mse.cpu().numpy())
        image_PSNR.append(mse2psnr(img_mse).cpu().numpy())

        rgb = torch.reshape(rgb, [H, W, 3])
        rgb_target = torch.reshape(rgb_target, [H, W, 3])
        disp = torch.reshape(disp, [H, W, 1])
        # depth_map_patch = torch.reshape(depth_map_patch_flat, [patch_size, patch_size, 1])
        if decoder is not None:
            rgb_decode = rgb.unsqueeze(0).permute(0, 3, 2, 1).contiguous()
            rgb_target_decode = rgb_target.unsqueeze(0).permute(0, 3, 2, 1).contiguous()
            decoded_message = decoder(rgb_decode)
            decoded_message = decoded_message[0, :]
            # decoded_message_bit = torch.heaviside(decoded_message - 0.5, torch.tensor([1.0]))
            decoded_message_bit = torch.round(decoded_message)
            message_BER.append(img2mse(decoded_message_bit, message_i).cpu().numpy())
            message_mse.append(img2mse(decoded_message, message_i).cpu().numpy())
            decoded_messages.append(decoded_message.cpu().numpy())
            decoded_messages_bit.append(decoded_message_bit.cpu().numpy())

        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        rgbs_target.append(rgb_target.cpu().numpy())
        if i == 0:
            print(rgb.shape, rgb_target.shape)


        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, 'lego-{:d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            rgb8 = to8b(rgbs_target[-1])
            filename = os.path.join(savedir, '{:03d}_target.png'.format(i))
            imageio.imwrite(filename, rgb8)

            rgb8 = to8b(disps[-1] / np.max(disps[-1]))
            filename = os.path.join(savedir, '{:03d}_disp.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    rgbs_target = np.stack(rgbs_target, 0)
    if decoder is not None:
        decoded_messages = np.stack(decoded_messages, 0)
        decoded_messages_bit = np.stack(decoded_messages_bit, 0)
        message_mse = np.stack(message_mse, 0)
        message_BER = np.stack(message_BER, 0)
        message_mse_mean = np.mean(message_mse).reshape(1)
        message_BER_mean = np.mean(message_BER).reshape(1)
        image_mse_mean = np.mean(image_mse).reshape(1)
        image_PSNR_mean = np.mean(image_PSNR).reshape(1)

        # print('mse_mean', message_mse_mean, 'ber_mean', message_ber_mean)
        if savedir is not None:
            filename = os.path.join(savedir, 'message.txt')
            np.savetxt(filename, message.cpu().numpy(), fmt='%f')

            filename = os.path.join(savedir, 'decoded_messages.txt')
            np.savetxt(filename, decoded_messages, fmt='%f')
            filename = os.path.join(savedir, 'message_mse.txt')
            np.savetxt(filename, message_mse, fmt='%f')
            filename = os.path.join(savedir, 'message_mse_mean.txt')
            np.savetxt(filename, message_mse_mean[None, :], fmt='%f')

            filename = os.path.join(savedir, 'decoded_messages_bit.txt')
            np.savetxt(filename, decoded_messages_bit, fmt='%f')
            filename = os.path.join(savedir, 'message_BER.txt')
            np.savetxt(filename, message_BER, fmt='%f')
            filename = os.path.join(savedir, 'message_BER_mean.txt')
            np.savetxt(filename, message_BER_mean, fmt='%f')

            filename = os.path.join(savedir, 'image_mse.txt')
            np.savetxt(filename, image_mse, fmt='%f')
            filename = os.path.join(savedir, 'image_mse_mean.txt')
            np.savetxt(filename, image_mse_mean, fmt='%f')

            filename = os.path.join(savedir, 'image_PSNR.txt')
            np.savetxt(filename, image_PSNR, fmt='%f')
            filename = os.path.join(savedir, 'image_PSNR_mean.txt')
            np.savetxt(filename, image_PSNR_mean, fmt='%f')

    # return rgbs, disps, decoded_messages
    re = {'rgbs': rgbs, 'rgbs_target': rgbs_target, 'message_BER_mean': message_BER_mean,
          'message_mse_mean': message_mse_mean, 'image_mse_mean': image_mse_mean, 'image_PSNR_mean': image_PSNR_mean,
          'decoded_messages': decoded_messages, 'decoded_messages_bit': decoded_messages_bit, 'message': message}
    return re

def render_path(render_poses, message, hwf, K, chunk, patch_size, render_kwargs, device, savedir=None,
                render_factor=0,
                encoder=None, decoder=None):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    rgbs_target = []
    decoded_messages = []
    decoded_messages_bit = []
    message_mse = []
    message_BER = []
    image_mse = []
    image_PSNR = []
    disps = []

    t = time.time()
    # d = 0
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        message_i = message[i, :]
        pose = c2w[:3, :4].to(device)
        rays_o, rays_d = get_rays(H, W, K, pose)
        H_min = int(np.random.randint(low=0, high=H - patch_size + 1, size=1))
        W_min = int(np.random.randint(low=0, high=W - patch_size + 1, size=1))
        coords_patch = torch.stack(torch.meshgrid(torch.linspace(H_min, H_min + patch_size - 1, patch_size),
                                                  torch.linspace(W_min, W_min + patch_size - 1, patch_size)),
                                   -1)  # (patch_size, patch_size, 2)
        coords_patch = torch.reshape(coords_patch, [-1, 2])
        select_coords = coords_patch.long()
        rays_o_patch = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d_patch = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        batch_rays_patch = torch.stack([rays_o_patch, rays_d_patch], 0)

        rgb, disp, _, rgb_target, _ = render(H, W, K, encoder, message_i.unsqueeze(0), chunk=chunk,
                                             rays=batch_rays_patch,
                                             **render_kwargs)

        img_mse = img2mse(rgb, rgb_target)
        image_mse.append(img_mse.cpu().numpy())
        image_PSNR.append(mse2psnr(img_mse).cpu().numpy())

        rgb = torch.reshape(rgb, [patch_size, patch_size, 3])
        rgb_target = torch.reshape(rgb_target, [patch_size, patch_size, 3])
        disp = torch.reshape(disp, [patch_size, patch_size, 1])
        # depth_map_patch = torch.reshape(depth_map_patch_flat, [patch_size, patch_size, 1])
        if decoder is not None:
            rgb_decode = rgb.unsqueeze(0).permute(0, 3, 2, 1).contiguous()
            rgb_target_decode = rgb_target.unsqueeze(0).permute(0, 3, 2, 1).contiguous()

            decoded_message = decoder(rgb_decode)
            decoded_message = decoded_message[0, :]
            # decoded_message_bit = torch.heaviside(decoded_message - 0.5, torch.tensor([1.0]))
            decoded_message_bit = torch.round(decoded_message)
            message_BER.append(img2mse(decoded_message_bit, message_i).cpu().numpy())
            message_mse.append(img2mse(decoded_message, message_i).cpu().numpy())
            decoded_messages.append(decoded_message.cpu().numpy())
            decoded_messages_bit.append(decoded_message_bit.cpu().numpy())

        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        rgbs_target.append(rgb_target.cpu().numpy())
        if i == 0:
            print(rgb.shape, rgb_target.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            rgb8 = to8b(rgbs_target[-1])
            filename = os.path.join(savedir, '{:03d}_target.png'.format(i))
            imageio.imwrite(filename, rgb8)

            rgb8 = to8b(disps[-1] / np.max(disps[-1]))
            filename = os.path.join(savedir, '{:03d}_disp.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    rgbs_target = np.stack(rgbs_target, 0)
    if decoder is not None:
        decoded_messages = np.stack(decoded_messages, 0)
        decoded_messages_bit = np.stack(decoded_messages_bit, 0)
        message_mse = np.stack(message_mse, 0)
        message_BER = np.stack(message_BER, 0)
        message_mse_mean = np.mean(message_mse).reshape(1)
        message_BER_mean = np.mean(message_BER).reshape(1)
        image_mse_mean = np.mean(image_mse).reshape(1)
        image_PSNR_mean = np.mean(image_PSNR).reshape(1)

        # print('mse_mean', message_mse_mean, 'ber_mean', message_ber_mean)
        if savedir is not None:
            filename = os.path.join(savedir, 'message.txt')
            np.savetxt(filename, message.cpu().numpy(), fmt='%f')

            filename = os.path.join(savedir, 'decoded_messages.txt')
            np.savetxt(filename, decoded_messages, fmt='%f')
            filename = os.path.join(savedir, 'message_mse.txt')
            np.savetxt(filename, message_mse, fmt='%f')
            filename = os.path.join(savedir, 'message_mse_mean.txt')
            np.savetxt(filename, message_mse_mean[None, :], fmt='%f')

            filename = os.path.join(savedir, 'decoded_messages_bit.txt')
            np.savetxt(filename, decoded_messages_bit, fmt='%f')
            filename = os.path.join(savedir, 'message_BER.txt')
            np.savetxt(filename, message_BER, fmt='%f')
            filename = os.path.join(savedir, 'message_BER_mean.txt')
            np.savetxt(filename, message_BER_mean, fmt='%f')

            filename = os.path.join(savedir, 'image_mse.txt')
            np.savetxt(filename, image_mse, fmt='%f')
            filename = os.path.join(savedir, 'image_mse_mean.txt')
            np.savetxt(filename, image_mse_mean, fmt='%f')

            filename = os.path.join(savedir, 'image_PSNR.txt')
            np.savetxt(filename, image_PSNR, fmt='%f')
            filename = os.path.join(savedir, 'image_PSNR_mean.txt')
            np.savetxt(filename, image_PSNR_mean, fmt='%f')
    print('message_BER_mean', message_BER_mean)
    print('image_PSNR_mean', image_PSNR_mean)
    # return rgbs, disps, decoded_messages
    re = {'rgbs': rgbs, 'rgbs_target': rgbs_target, 'message_BER_mean': message_BER_mean,
          'message_mse_mean': message_mse_mean, 'image_mse_mean': image_mse_mean, 'image_PSNR_mean': image_PSNR_mean,
          'decoded_messages': decoded_messages, 'decoded_messages_bit': decoded_messages_bit, 'message': message}
    return re


def create_nerf(encoder, args, rank):
    """Instantiate NeRF's MLP model.
    """
    device = torch.device(f"cuda:{rank}")
    assert os.path.exists(args.pretrained_model), "Pre_trained model not found!"
    # pretained = args.pretained_model
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)


    grad_vars = list(model.parameters())
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

        grad_vars += list(model_fine.parameters())
        model_fine.eval()
        for name, param in model_fine.named_parameters():
            param.requires_grad = False

    network_query_fn = lambda inputs, message, viewdirs, network_fn, encoder_fn: run_network(inputs, message, viewdirs,
                                                                                             network_fn, encoder_fn,
                                                                                             embed_fn=embed_fn,
                                                                                             embeddirs_fn=embeddirs_fn,
                                                                                             netchunk=args.netchunk)


    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    ckpts = [args.pretrained_model]
    # if args.ft_path is not None and args.ft_path!='None':
    #
    # else:
    #     ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found pretrianed ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars


def create_decoder(args, rank):
    device = torch.device(f"cuda:{rank}")
    decoder_net = Decoder_sigmoid(decoder_channels=64, decoder_blocks=args.decoder_blocks, message_length=args.message_dim).to(device)
    decoder_net = DDP(decoder_net, device_ids=[rank], find_unused_parameters=True)
    # decoder_net = torch.nn.DataParallel(decoder_net)
    # Create optimizer

    optimizer_decoder = torch.optim.Adam(params=decoder_net.parameters(), lr=args.lrate_decoder, betas=(0.9, 0.999))

    basedir = args.basedir
    expname = args.expname

    if args.ckpt is not None:
        ckpts = [args.ckpt]
    else:
        ckpts = []
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading decoder from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        if 'optimizer_decoder_state_dict' in ckpt.keys() and 'decoder_net_state_dict' in ckpt.keys():
            optimizer_decoder.load_state_dict(ckpt['optimizer_decoder_state_dict'])

            # Load model
            decoder_net.load_state_dict(ckpt['decoder_net_state_dict'])

    return decoder_net, optimizer_decoder


def create_encoder(H, W, args, rank):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    # skips = [4]
    device = torch.device(f"cuda:{rank}")
    encoder_net = Encoder_Tri_MLP_f(D=3, W=256, input_ch=input_ch, input_ch_color=4, input_ch_message=args.message_dim,
                                  input_ch_views=input_ch_views, output_ch=output_ch,
                                  skips=[-1], use_viewdirs=args.use_viewdirs).to(device)
    encoder_net = DDP(encoder_net, device_ids=[rank], find_unused_parameters=True)
    # encoder_net = Encoder_MLP(D=5, W=256, input_ch=input_ch, input_ch_color=4, input_ch_message=args.message_dim,
    #                               input_ch_views=input_ch_views, output_ch=output_ch,
    #                               skips=[-1], use_viewdirs=args.use_viewdirs).to(device)

    # encoder_net = torch.nn.DataParallel(encoder_net)
    # Create optimizer
    optimizer_encoder = torch.optim.Adam(params=encoder_net.parameters(), lr=args.lrate_encoder, betas=(0.9, 0.999))

    basedir = args.basedir
    expname = args.expname
    start = 0
    if args.ckpt is not None:
        ckpts = [args.ckpt]
    else:
        ckpts = []
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading encoder from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        start = ckpt['global_step']
        if 'optimizer_encoder_state_dict' in ckpt.keys() and 'encoder_net_state_dict' in ckpt.keys():
            optimizer_encoder.load_state_dict(ckpt['optimizer_encoder_state_dict'])

            # Load model
            encoder_net.load_state_dict(ckpt['encoder_net_state_dict'])

    return encoder_net, optimizer_encoder, start


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists.device)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    rgb_target = torch.sigmoid(raw[..., 3:6])
    
    rgb = torch.where(torch.isnan(rgb), torch.full_like(rgb, 0), rgb)
    rgb_target = torch.where(torch.isnan(rgb_target), torch.full_like(rgb_target, 0), rgb_target)
    
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 6].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 6] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    # rgb
    # rgb_encoded = encoder(rgb, weights, message)

    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    rgb_map_target = torch.sum(weights[..., None] * rgb_target, -2)  # [N_rays, 3]
    
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    # disp_map = torch.nan_to_num(disp_map, nan=0.0)
    disp_map = torch.where(torch.isnan(disp_map), torch.full_like(disp_map, 0), disp_map)

    acc_map = torch.sum(weights, -1)
    # tmp = weights.cpu().numpy()
    # tmp2 = rgb[:, :, 0].cpu().numpy()
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
        rgb_map_target = rgb_map_target + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map, rgb_map_target


def render_rays(ray_batch,
                encoder,
                message,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    
    t_vals = torch.linspace(0., 1., steps=N_samples).to(near.device)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])
#     perturb = 1.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.randn(z_vals.shape).to(lower.device)*0.2

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    #     raw = run_network(pts)
    raw = network_query_fn(pts, None, viewdirs, network_fn, None)

    # raw = torch.cat(all_raw, 0).detach_()
    # raw = network_query_fn(pts, viewdirs, network_fn)

    _, _, _, weights, _, _ = raw2outputs(raw,
                                         z_vals,
                                         rays_d,
                                         raw_noise_std,
                                         white_bkgd,
                                         pytest=pytest)

    if N_importance > 0:
        # rgb_map_0, disp_map_0, acc_map_0 = None, None, None

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, message, viewdirs, run_fn, encoder)

        rgb_map, disp_map, acc_map, weights, depth_map, target_rgb_map = raw2outputs(
            raw, z_vals, rays_d,
            raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'target_rgb_map': target_rgb_map}

    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        # ret['rgb0'] = rgb_map_0
        # ret['disp0'] = disp_map_0
        # ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 32,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--factor_res", type=float, default=-1,
                        help='downsample factor for blender images')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=1000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=500,
                        help='frequency of testset saving')
    parser.add_argument("--i_testset_out", type=int, default=500,
                        help='frequency of testset saving')

    parser.add_argument("--lrate_decoder", type=float, default=1e-3,
                        help='learning rate for decoder')
    parser.add_argument("--lrate_decoder_decay", type=int, default=1000,
                        help='exponential learning rate decay (in 1000 steps) for decoder')

    parser.add_argument("--lrate_encoder", type=float, default=5e-4,
                        help='learning rate for decoder')
    parser.add_argument("--lrate_encoder_decay", type=int, default=1000,
                        help='exponential learning rate decay (in 1000 steps) for decoder')
    parser.add_argument("--message_dim", type=int, default=8,
                        help='length of the 0-1 message')
    parser.add_argument("--pretrained_model", type=str, default='./model/200000.tar',
                        help='pretrained_model')

    parser.add_argument("--ckpt", type=str, default=None,
                        help='ckpt')

    parser.add_argument("--batch_size", type=int, default=1,
                        help='batch_size')

    parser.add_argument("--N_iters", type=int, default=200000,
                        help='Training loop times')

    parser.add_argument("--patch_size", type=int, default=100,
                        help='Patch size')

    parser.add_argument("--noise_type", type=str, default=None,
                        help='Noise type')

    parser.add_argument("--w_message", type=float, default=10.0,
                        help='w_message')
    
    parser.add_argument("--w_img", type=float, default=1.0,
                        help='w_img')
    
    parser.add_argument("--angle_range", type=float, default=360.0,
                        help='angle_range')
    
    parser.add_argument("--decoder_blocks", type=int, default=6,
                        help='decoder_blocks')
    
    return parser


def train(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test, llff_render_para = load_llff_data(args.datadir, args.factor,
                                                                                    recenter=True, bd_factor=.75,
                                                                                    spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.factor_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res,
                                                                                    args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(basedir, expname, 'tensorboard'))
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    decoder_net, optimizer_decoder = create_decoder(args, rank)
    encoder_net, optimizer_encoder, start = create_encoder(H, W, args, rank)
    render_kwargs_train, render_kwargs_test, _, grad_vars = create_nerf(encoder_net, args, rank)

    vggloss = VGGLoss(block_no=4, layer_within_block=1, use_batch_norm_vgg=False).to(device)

    noise_type = args.noise_type
    if noise_type is not None:
        if noise_type == 'noise':
            T_Noise = addNoise(0.01)
        elif noise_type == 'rotation':
            T_Noise = T.RandomRotation(30)
        elif noise_type == 'resize':
            T_Noise = Resize([0.75, 1])
        elif noise_type == 'blur':
            T_Noise = T.GaussianBlur(kernel_size=3, sigma=0.1)
        else:
            assert("noise not right")

    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    render_poses = torch.Tensor(render_poses).to(device)
    N_test = 200
    if args.dataset_type == 'blender':
        poses_test = get_random_poses_blender(N_test, theta_bd_min=-args.angle_range/2, theta_bd_max=args.angle_range/2,
                                              phi_bd_min=-30., phi_bd_max=-30., radius_bd_min=4.031128874,
                                              radius_bd_max=4.031128874)
        poses_test = poses_test.to(device)

        poses_test_fix = get_random_poses_blender(1, theta_bd_min=0., theta_bd_max=0.,
                                              phi_bd_min=-30., phi_bd_max=-30., radius_bd_min=4.031128874,
                                              radius_bd_max=4.031128874)
        poses_test_fix = poses_test_fix.to(device)

    elif args.dataset_type == 'llff':
        poses_test = get_random_poses_llff(N_test, llff_render_para['c2w_path'], llff_render_para['up'],
                                           llff_render_para['rads'], llff_render_para['focal'])
        poses_test = torch.Tensor(poses_test).to(device)

        poses_test_fix = get_fix_poses_llff(1, llff_render_para['c2w_path'], llff_render_para['up'],
                                           llff_render_para['rads'], llff_render_para['focal'])
        poses_test_fix = torch.Tensor(poses_test_fix).to(device)
    # angles_test = torch.Tensor(angles_test).to(device)
    message_test = torch.randint(0, 2, (N_test, args.message_dim)).float().to(device)


    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname,
                                           'renderonly_whole_{}_{:06d}'.format('test' if args.render_test else 'path',
                                                                               start))
            os.makedirs(testsavedir, exist_ok=True)
            re = render_whole_image(poses_test_fix, message_test, hwf, K, args.chunk, args.patch_size,
                                       render_kwargs_test, device,
                                       savedir=testsavedir, render_factor=args.render_factor, encoder=encoder_net,
                                       decoder=decoder_net)
            rgbs = re['rgbs']
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            testsavedir = os.path.join(basedir, expname,
                                           'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses_test.shape)

            _ = render_path(poses_test, message_test, hwf, K, args.chunk, args.patch_size,
                                render_kwargs_test, device,
                                savedir=testsavedir, render_factor=args.render_factor, encoder=encoder_net,
                                decoder=decoder_net)

            print('Done rendering', testsavedir)
            return
#         else:
#             return


    N_iters = args.N_iters + 1
    print('Begin')


    w_message = args.w_message
    w_img = args.w_img
    start = start + 1
    patch_size = args.patch_size
    for i in tqdm(range(start, N_iters)):
        time0 = time.time()

        if args.dataset_type == 'blender':
            pose = get_random_poses_blender(args.batch_size, theta_bd_min=-args.angle_range/2, theta_bd_max=args.angle_range/2,
                                            phi_bd_min=-30, phi_bd_max=-30, radius_bd_min=4.031128874,
                                            radius_bd_max=4.031128874)
            pose = pose[0, :3, :4].to(device)
        elif args.dataset_type == 'llff':
            pose = get_random_poses_llff(args.batch_size, llff_render_para['c2w_path'], llff_render_para['up'],
                                         llff_render_para['rads'], llff_render_para['focal'])
            pose = torch.Tensor(pose).to(device)
            pose = pose[0, :3, :4]

        target_s = None

        rays_o, rays_d = get_rays(H, W, K, pose)  # (H, W, 3), (H, W, 3)


        message = torch.randint(0, 2, (args.batch_size, args.message_dim)).float().to(device)

        H_min = int(np.random.randint(low=0, high=H - patch_size + 1, size=1))
        W_min = int(np.random.randint(low=0, high=W - patch_size + 1, size=1))
        coords_patch = torch.stack(torch.meshgrid(torch.linspace(H_min, H_min + patch_size - 1, patch_size),
                                                  torch.linspace(W_min, W_min + patch_size - 1, patch_size)),
                                   -1)  # (patch_size, patch_size, 2)
        coords_patch = torch.reshape(coords_patch, [-1, 2])
        select_coords = coords_patch.long()
        rays_o_patch = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d_patch = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        # depth_map_patch_flat = depth_map[select_coords[:, 0], select_coords[:, 1]]
        batch_rays_patch = torch.stack([rays_o_patch, rays_d_patch], 0)

        rgb_patch, disp, acc, target_s_patch, _ = render(H, W, K, encoder=encoder_net, message=message,
                                                         chunk=args.chunk,
                                                         rays=batch_rays_patch,
                                                         verbose=i < 10, retraw=False,
                                                         **render_kwargs_train)

        # optimizer.zero_grad()
        optimizer_decoder.zero_grad()
        optimizer_encoder.zero_grad()

        img_loss = img2mse(rgb_patch, target_s_patch)

        # rgb_whole = torch.reshape(rgb_whole, [H, W, 3])
        rgb_patch = torch.reshape(rgb_patch, [patch_size, patch_size, 3])
        target_s_patch = torch.reshape(target_s_patch, [patch_size, patch_size, 3])
        # rgb_whole[H_min: H_min + patch_size, W_min: W_min + patch_size] = rgb_patch
        # target_s = torch.reshape(target_s, [H, W, 3])
        rgb_patch = rgb_patch.unsqueeze(0).permute(0, 3, 2, 1).contiguous()
        target_s_patch = target_s_patch.unsqueeze(0).permute(0, 3, 2, 1).contiguous()
        # rgb_whole = rgb_whole.unsqueeze(0).permute(0, 3, 2, 1).contiguous()
        # target_s = target_s[None, :, :, :].permute(0, 3, 2, 1).contiguous()
        perc_loss = img2mse(vggloss(rgb_patch), vggloss(target_s_patch))
        if noise_type is not None:
            rgb_patch = T_Noise(rgb_patch)
        decoded_message = decoder_net(rgb_patch)
        # decoded_message = decoded_message[0, :]
        message_loss = F.binary_cross_entropy(decoded_message, message)
        # trans = extras['raw'][..., -1]
        
        loss = w_img * img_loss + w_message * message_loss + 0.01 * perc_loss
        # loss = img_loss + 0.8 * message_loss + 0.02 * perc_loss
        psnr = mse2psnr(img_loss)

        # if 'rgb0' in extras:
        #     img_loss0 = img2mse(extras['rgb0'], target_s)
        #     loss = loss + img_loss0
        #     psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decoder_decay * 1000
        new_lrate = args.lrate_decoder * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer_decoder.param_groups:
            param_group['lr'] = new_lrate

        decay_rate = 0.1
        decay_steps = args.lrate_encoder_decay * 1000
        new_lrate = args.lrate_encoder * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer_encoder.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if rank == 0:
            if i % args.i_weights == 0:
                path = os.path.join(basedir, expname, 'ckpt.tar')
                torch.save({
                    'global_step': global_step,
                    'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
                    'decoder_net_state_dict': decoder_net.state_dict(),
                    'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                    'encoder_net_state_dict': encoder_net.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
                if i % 10000 == 0:
                    path = os.path.join(basedir, expname, 'ckpt_{:06d}.tar'.format(i))
                    torch.save({
                        'global_step': global_step,
                        'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
                        'decoder_net_state_dict': decoder_net.state_dict(),
                        'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                        'encoder_net_state_dict': encoder_net.state_dict(),
                    }, path)
                print('Saved checkpoints at', path)


            if i % args.i_testset == 0 and i > 0:
                if i % args.i_testset_out == 0:
                    testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
                    os.makedirs(testsavedir, exist_ok=True)
                else:
                    testsavedir = None

                print('test poses shape', poses_test.shape)
                # print('test poses shape', poses[i_test].shape)
                with torch.no_grad():
                    re = render_path(poses_test, message_test, hwf, K, args.chunk, args.patch_size,
                                     render_kwargs_test, device,
                                     savedir=testsavedir, render_factor=args.render_factor, encoder=encoder_net,
                                     decoder=decoder_net)
                    summary_writer.add_scalar('test/BER',
                                              re['message_BER_mean'], global_step=i)
                    summary_writer.add_scalar('test/message_mse',
                                              re['message_mse_mean'], global_step=i)
                    summary_writer.add_scalar('test/image_mse',
                                              re['image_mse_mean'], global_step=i)
                    summary_writer.add_scalar('test/image_PSNR',
                                              re['image_PSNR_mean'], global_step=i)
                    # summary_writer.add_scalar('test/decoded_messages',
                    #                           re['decoded_messages'], global_step=i)
                    # summary_writer.add_scalar('test/decoded_messages_bit',
                    #                           re['decoded_messages_bit'], global_step=i)
                print('Saved test set')

            if i % args.i_print == 0:
                decoded_message_bit = torch.round(decoded_message)
                BER = img2mse(decoded_message_bit, message)
                tqdm.write(
                    f"[TRAIN] Iter: {i} image_loss: {img_loss.item()} message_loss: {message_loss.item()} BER: {BER.item()} "
                    f"PSNR: {psnr.item()} perc_loss: {perc_loss.item()}")
                summary_writer.add_scalar('train/img_loss',
                                          img_loss.item(), global_step=i)
                summary_writer.add_scalar('train/message_loss',
                                          message_loss.item(), global_step=i)
                summary_writer.add_scalar('train/PSNR',
                                          psnr.item(), global_step=i)
                summary_writer.add_scalar('train/BER',
                                          BER.item(), global_step=i)

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    mp.spawn(train, args=(8,), nprocs=8, join=True)
