#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import torch
import lpips
from torch import nn
from utils.loss_utils import l1_loss, ssim, msssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, knn
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader

from utils.mesh_utils import GaussianExtractor
from utils.render_utils import generate_path, create_videos

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def validation(dataset, opt, pipe,checkpoint, gaussian_dim, time_duration, rot_4d, force_sh_3d,
               num_pts, num_pts_ratio):
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()

    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, 
                              rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    
    assert checkpoint, "No checkpoint provided for validation"
    scene = Scene(dataset, gaussians, shuffle=False,num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)
    
    (model_params, first_iter) = torch.load(checkpoint)
    train_dir = os.path.join(dataset.model_path, 'train', "ours_{}".format(first_iter))
    test_dir = os.path.join(dataset.model_path, 'test', "ours_{}".format(first_iter))
    gaussians.restore(model_params, None)

    def _print_rest_param_stats(branch_name, features_rest, sh_mask_logit, threshold):
        """
        只统计 features_rest 的参数量（不含 DC）
        当前只统计前 15 个 spatial SH rest 通道: l1=3, l2=5, l3=7
        """
        if features_rest is None or features_rest.numel() == 0:
            return
        if sh_mask_logit is None or sh_mask_logit.numel() == 0:
            return

        soft_sh = torch.sigmoid(sh_mask_logit.detach())

        r1 = (soft_sh[:, 0] > threshold).float().mean().item()
        r2 = (soft_sh[:, 1] > threshold).float().mean().item()
        r3 = (soft_sh[:, 2] > threshold).float().mean().item()

        n_pts = features_rest.shape[0]
        rest_ch = min(features_rest.shape[1], 15)
        bytes_per_elem = features_rest.element_size()

        ch_l1 = min(max(rest_ch - 0, 0), 3)
        ch_l2 = min(max(rest_ch - 3, 0), 5)
        ch_l3 = min(max(rest_ch - 8, 0), 7)

        dense_rest_params = n_pts * rest_ch * 3
        kept_rest_params = n_pts * (ch_l1 * r1 + ch_l2 * r2 + ch_l3 * r3) * 3

        dense_rest_mb = dense_rest_params * bytes_per_elem / 1024.0 / 1024.0
        kept_rest_mb = kept_rest_params * bytes_per_elem / 1024.0 / 1024.0
        kept_ratio = kept_rest_params / dense_rest_params if dense_rest_params > 0 else 1.0
        compression_ratio = 1.0 - kept_ratio

        print(f"\n[{branch_name} rest params]")
        print(f"  dense rest params        : {dense_rest_params:.0f}")
        print(f"  kept rest params         : {kept_rest_params:.0f}")
        print(f"  dense rest size          : {dense_rest_mb:.4f} MB")
        print(f"  kept rest size           : {kept_rest_mb:.4f} MB")
        print(f"  rest kept ratio          : {kept_ratio*100:.2f}%")
        print(f"  rest compression ratio   : {compression_ratio*100:.2f}%")

    if getattr(opt, "use_pruning", False):
        # ---------------- Phi Distribution ----------------
        print("\n================ Phi Distribution ================\n")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        if hasattr(gaussians, "dynamic_mask_logit") and gaussians.dynamic_mask_logit is not None and gaussians.dynamic_mask_logit.numel() > 0:
            phi_dyn = torch.sigmoid(gaussians.dynamic_mask_logit.detach()).view(-1).cpu()
            print("Total dynamic gaussians:", phi_dyn.shape[0])
            print("dynamic phi min :", phi_dyn.min().item())
            print("dynamic phi max :", phi_dyn.max().item())
            print("dynamic phi mean:", phi_dyn.mean().item())
            print("dynamic phi std :", phi_dyn.std().item())

            print("\nRatio below thresholds:")
            for t in thresholds:
                ratio = (phi_dyn < t).float().mean().item()
                print(f"phi_dyn < {t:.1f} : {ratio*100:.2f}%")

        if hasattr(gaussians, "static_mask_logit") and gaussians.static_mask_logit is not None and gaussians.static_mask_logit.numel() > 0:
            phi_sta = torch.sigmoid(gaussians.static_mask_logit.detach()).view(-1).cpu()
            print("Total static gaussians:", phi_sta.shape[0])
            print("phi min :", phi_sta.min().item())
            print("phi max :", phi_sta.max().item())
            print("phi mean:", phi_sta.mean().item())
            print("phi std :", phi_sta.std().item())

            print("\nRatio below thresholds:")
            for t in thresholds:
                ratio = (phi_sta < t).float().mean().item()
                print(f"phi < {t:.1f} : {ratio*100:.2f}%")

        print("\n==================================================\n")

    # ---------------- SH Mask Distribution ----------------
    if getattr(opt, "use_sh_adaptive", False):
        print("\n================ SH Mask Distribution ================\n")
        sh_threshold = getattr(opt, "phi_prune_sh", 0.1)

        _print_rest_param_stats(
            "dynamic",
            getattr(gaussians, "_features_rest", None),
            getattr(gaussians, "dynamic_sh_mask_logit", None),
            sh_threshold,
        )

        _print_rest_param_stats(
            "static",
            getattr(gaussians, "static_features_rest", None),
            getattr(gaussians, "static_sh_mask_logit", None),
            sh_threshold,
        )
              
    print("\n==================================================\n")


    # ---------------- Hard Pruning ---------------------------
    print("\n================ Hard Pruning Pipeline ================\n")

    dynamic_points_before = gaussians.get_xyz.shape[0]
    static_points_before = gaussians.get_static_xyz.shape[0]
    baseline_points = dynamic_points_before + static_points_before

    phi_prune_dynamic = getattr(opt, "phi_prune_dynamic", 0.1)
    phi_prune_static = getattr(opt, "phi_prune_static", 0.1)
    sh_threshold = getattr(opt, "phi_prune_sh", 0.1)
    
    print(f"\n[Baseline]")
    print(f"Points          : {baseline_points}")
    print(f"Dynamic_Points  : {dynamic_points_before}")
    print(f"Static_Points   : {static_points_before}")

        
    print("\n[Model Size Before Pruning]")

    size_before_dense = gaussians.compute_model_size(
        count_mode="dense",
        include_dynamic=True,
        include_static=True,
        include_temporal=True,
        include_masks=False,
        include_sh_mask_logits=False,
        verbose=False,
    )

    size_before_effective = gaussians.compute_model_size(
        count_mode="effective",
        include_dynamic=True,
        include_static=True,
        include_temporal=True,
        include_masks=False,
        include_sh_mask_logits=False,
        phi_threshold_dynamic=phi_prune_dynamic if opt.use_pruning else None,
        phi_threshold_static=phi_prune_static if opt.use_pruning else None,
        sh_threshold=sh_threshold if opt.use_sh_adaptive else None,
        verbose=False,
    )

    print(f"Dense size before pruning     : {size_before_dense['total_mb']:.4f} MB")
    print(f"Effective size before pruning : {size_before_effective['total_mb']:.4f} MB")
    print(f"Dynamic kept (effective)      : {size_before_effective['num_dynamic_kept']} / {size_before_effective['num_dynamic_points']}")
    print(f"Static kept (effective)       : {size_before_effective['num_static_kept']} / {size_before_effective['num_static_points']}")

    print("\nEvaluating model BEFORE pruning on all test views...")
    all_test_views = list(scene.getTestCameras())
    psnr_before_list = []
    ssim_before_list = []
    lpips_before_list = []

    for gt_image, viewpoint_cam in all_test_views:
        gt_image, viewpoint_cam = gt_image.cuda(), viewpoint_cam.cuda()

        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            
            image_lpips = image.unsqueeze(0) * 2.0 - 1.0
            gt_lpips = gt_image.unsqueeze(0) * 2.0 - 1.0

            psnr_before_list.append(psnr(image, gt_image).mean().item())
            ssim_before_list.append(ssim(image, gt_image).item())
            lpips_before_list.append(lpips_fn(image_lpips, gt_lpips).mean().item())

        del render_pkg, image, gt_image, viewpoint_cam
        torch.cuda.empty_cache()

    psnr_before = sum(psnr_before_list) / len(psnr_before_list)
    ssim_before = sum(ssim_before_list) / len(ssim_before_list)
    lpips_before = sum(lpips_before_list) / len(lpips_before_list)

    print(f"\nPSNR before pruning: {psnr_before:.4f}")
    print(f"SSIM before pruning: {ssim_before:.6f}")
    print(f"LPIPS before pruning: {lpips_before:.6f}")

    print("\nApplying hard pruning...")

    if opt.use_pruning:
        gaussians.prune_low_dynamic_mask_points(threshold=phi_prune_dynamic)
        gaussians.prune_low_mask_points(threshold=phi_prune_static)

    if opt.use_sh_adaptive:
        gaussians.hard_prune_sh(threshold=sh_threshold)
        gaussians.use_sh_adaptive = False
        if hasattr(gaussians, "enable_sh_mask"):
            gaussians.enable_sh_mask = False
    
    if opt.use_pruning:
        dynamic_points_after = gaussians.get_xyz.shape[0]
        static_points_after = gaussians.get_static_xyz.shape[0]
        new_count = dynamic_points_after + static_points_after
        compression = (1 - new_count / baseline_points) * 100 if baseline_points > 0 else 0.0

        print(f"\n[Hard Pruning]")
        print(f"Dynamic before : {dynamic_points_before}")
        print(f"Dynamic after  : {dynamic_points_after}")
        print(f"Static before  : {static_points_before}")
        print(f"Static after   : {static_points_after}")
        print(f"Points before  : {baseline_points}")
        print(f"Points after   : {new_count}")
        print(f"Reduction      : {compression:.2f}%")
        print(f"phi_prune_dynamic = {phi_prune_dynamic}")
        print(f"phi_prune_static  = {phi_prune_static}")

    if opt.use_sh_adaptive:
        print("\nChecking SH zero ratio after hard pruning...")

        dyn_zero_ratio = (
            (gaussians._features_rest == 0).float().mean().item()
        )

        print(f"Dynamic SH zero ratio: {dyn_zero_ratio*100:.2f}%")

        if hasattr(gaussians, "static_features_rest") and gaussians.static_features_rest is not None and gaussians.static_features_rest.numel() > 0:

            sta_zero_ratio = (
                (gaussians.static_features_rest == 0).float().mean().item()
            )

            print(f"Static SH zero ratio: {sta_zero_ratio*100:.2f}%")
    
    print("\n[Model Size After Pruning]")

    size_after_dense = gaussians.compute_model_size(
        count_mode="dense",
        include_dynamic=True,
        include_static=True,
        include_temporal=True,
        include_masks=False,
        include_sh_mask_logits=False,
        verbose=False,
    )

    size_after_effective = gaussians.compute_model_size(
        count_mode="effective",
        include_dynamic=True,
        include_static=True,
        include_temporal=True,
        include_masks=False,
        include_sh_mask_logits=False,
        phi_threshold_dynamic=None,
        phi_threshold_static=None,
        sh_threshold=sh_threshold if opt.use_sh_adaptive else None,
        verbose=False,
    )

    print(f"Dense size after pruning      : {size_after_dense['total_mb']:.4f} MB")
    print(f"Effective size after pruning  : {size_after_effective['total_mb']:.4f} MB")
    print(f"Dynamic kept after pruning    : {size_after_effective['num_dynamic_kept']} / {size_after_effective['num_dynamic_points']}")
    print(f"Static kept after pruning     : {size_after_effective['num_static_kept']} / {size_after_effective['num_static_points']}")

    # ---------------- PSNR Evaluation ----------------
    print("\nEvaluating pruned model on all test views...")
    all_test_views = list(scene.getTestCameras())
    psnr_pruned_list = []
    ssim_pruned_list = []
    lpips_pruned_list = []

    for gt_image, viewpoint_cam in all_test_views:
        gt_image, viewpoint_cam = gt_image.cuda(), viewpoint_cam.cuda()

        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)

            image_lpips = image.unsqueeze(0) * 2.0 - 1.0
            gt_lpips = gt_image.unsqueeze(0) * 2.0 - 1.0

            psnr_pruned_list.append(psnr(image, gt_image).mean().item())
            ssim_pruned_list.append(ssim(image, gt_image).item())
            lpips_pruned_list.append(lpips_fn(image_lpips, gt_lpips).mean().item())

        del render_pkg, image, gt_image, viewpoint_cam
        torch.cuda.empty_cache()

    psnr_pruned = sum(psnr_pruned_list) / len(psnr_pruned_list)
    ssim_pruned = sum(ssim_pruned_list) / len(ssim_pruned_list)
    lpips_pruned = sum(lpips_pruned_list) / len(lpips_pruned_list)

    print(f"\nPSNR after pruning: {psnr_pruned:.4f}")
    print(f"SSIM after pruning: {ssim_pruned:.6f}")
    print(f"LPIPS after pruning: {lpips_pruned:.6f}")
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)   
    
    #########   1. Validation and Rendering ############

    # print("export rendered testing images ...")
    # os.makedirs(test_dir, exist_ok=True)
    # gaussExtractor.reconstruction(scene.getTestCameras(),test_dir,stage = "validation")
    # gaussExtractor.export_image(test_dir,mode = "validation")

    # #########    2. Render Trajectory       ############
    
    # print("rendering trajectory ...")
    # traj_dir = os.path.join(test_dir, 'traj')
    # os.makedirs(traj_dir, exist_ok=True)
    # n_fames = 480
    # cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
    # gaussExtractor.reconstruction(cam_traj, test_dir,stage = "trajectory")
    # gaussExtractor.export_image(traj_dir,mode = "trajectory")
    # create_videos( base_dir =traj_dir,
    #                input_dir=traj_dir, 
    #                out_name='render_traj', 
    #                num_frames=n_fames)
    
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size):
    
    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    scene = Scene(dataset, gaussians, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)
    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    best_psnr = 0.0
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    ema_ssimloss_for_log = 0.0
    lambda_all = [key for key in opt.__dict__.keys() if key.startswith('lambda') and key!='lambda_dssim']
    for lambda_name in lambda_all:
        vars()[f"ema_{lambda_name.replace('lambda_','')}_for_log"] = 0.0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
        
    if pipe.env_map_res:
        env_map = nn.Parameter(torch.zeros((3,pipe.env_map_res, pipe.env_map_res),dtype=torch.float, device="cuda").requires_grad_(True))
        env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
    else:
        env_map = None
        
    gaussians.env_map = env_map
        
    training_dataset = scene.getTrainCameras()
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=12 if dataset.dataloader else 0, collate_fn=lambda x: x, drop_last=True)
     
    iteration = first_iter

    while iteration < opt.iterations + 1:
        for batch_data in training_dataloader:
            iteration += 1
            if iteration > opt.iterations:
                break

            iter_start.record()
            gaussians.update_learning_rate(iteration)
            
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % opt.sh_increase_interval == 0:
                gaussians.oneupSHdegree()
            
            sh_mask_start_iter = getattr(opt, "sh_mask_start_iter", 4500)
            sh_mask_warmup_iters = getattr(opt, "sh_mask_warmup_iters", 1000)

            gaussians.enable_sh_mask = (
                opt.use_sh_adaptive and iteration >= sh_mask_start_iter
            )

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            
            batch_point_grad = []
            batch_visibility_filter = []
            batch_radii = []

            batch_point_grad_static = []
            batch_visibility_filter_static = []
            batch_radii_static = []

            batch_alpha_stat = []
            batch_motion_stat = []
            batch_time_support_stat = []
            
            for batch_idx in range(batch_size):
                gt_image, viewpoint_cam = batch_data[batch_idx]
                gt_image = gt_image.cuda()
                viewpoint_cam = viewpoint_cam.cuda()

                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                depth = render_pkg["depth"]
                alpha = render_pkg["alpha"]
                viewspace_point_tensor_static = render_pkg["viewspace_points_static"]
                visibility_filter_static = render_pkg["visibility_filter_static"]
                radii_static = render_pkg["radii_static"]

                alpha_stat = None
                motion_stat = None
                time_support_stat = None

                if gaussians.gaussian_dim == 4:
                    cur_vis = visibility_filter.unsqueeze(1).float()

                    # 1) 当前步 opacity 统计
                    alpha_stat = gaussians.get_opacity.detach() * cur_vis

                    # 2) 当前步运动强度统计
                    _, mean_offset = gaussians.get_current_covariance_and_mean_offset(
                        1.0, viewpoint_cam.timestamp
                    )
                    motion_stat = mean_offset.norm(dim=1, keepdim=True).detach() * cur_vis

                    # 3) 当前步时间尺度统计
                    time_support_stat = gaussians.get_scaling_t.detach() * cur_vis

                    batch_alpha_stat.append(alpha_stat)
                    batch_motion_stat.append(motion_stat)
                    batch_time_support_stat.append(time_support_stat)

                # Loss
                Lstatic_mask = torch.tensor(0.0, device="cuda")
                Ldynamic_mask = torch.tensor(0.0, device="cuda")
                Lsh = torch.tensor(0.0, device="cuda")
                Ll1 = l1_loss(image, gt_image)
                Lssim = 1.0 - ssim(image, gt_image)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
                
                ###### opa mask Loss ######
                if opt.lambda_opa_mask > 0:
                    o = alpha.clamp(1e-6, 1-1e-6)
                    sky = 1 - viewpoint_cam.gt_alpha_mask

                    Lopa_mask = (- sky * torch.log(1 - o)).mean()

                    # lambda_opa_mask = opt.lambda_opa_mask * (1 - 0.99 * min(1, iteration/opt.iterations))
                    lambda_opa_mask = opt.lambda_opa_mask
                    loss = loss + lambda_opa_mask * Lopa_mask
                ###### opa mask Loss ######
                
                ###### rigid loss ######
                if opt.lambda_rigid > 0:
                    k = 20
                    # cur_time = viewpoint_cam.timestamp
                    # _, delta_mean = gaussians.get_current_covariance_and_mean_offset(1.0, cur_time)
                    xyz_mean = gaussians.get_xyz
                    xyz_cur =  xyz_mean #  + delta_mean
                    idx, dist = knn(xyz_cur[None].contiguous().detach(), 
                                    xyz_cur[None].contiguous().detach(), 
                                    k)
                    _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                    weight = torch.exp(-100 * dist)
                    # cur_marginal_t = gaussians.get_marginal_t(cur_time).detach().squeeze(-1)
                    # marginal_weights = cur_marginal_t[idx] * cur_marginal_t[None,:,None]
                    # weight *= marginal_weights
                    
                    # mean_t, cov_t = gaussians.get_t, gaussians.get_cov_t(scaling_modifier=1)
                    # mean_t_nn, cov_t_nn = mean_t[idx], cov_t[idx]
                    # weight *= torch.exp(-0.5*(mean_t[None, :, None]-mean_t_nn)**2/cov_t[None, :, None]/cov_t_nn*(cov_t[None, :, None]+cov_t_nn)).squeeze(-1).detach()
                    vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
                    Lrigid = (weight * vel_dist).sum() / k / xyz_cur.shape[0]
                    loss = loss + opt.lambda_rigid * Lrigid
                ########################
                
                ###### motion loss ######
                if opt.lambda_motion > 0:
                    _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                    Lmotion = velocity.norm(p=2, dim=1).mean()
                    loss = loss + opt.lambda_motion * Lmotion
                ########################

                ##########gs_mask###############
                if opt.use_pruning and (opt.lambda_static_mask > 0 or opt.lambda_dynamic_mask > 0):
                    if hasattr(gaussians, "static_mask_logit") and gaussians.static_mask_logit.numel() > 0:
                        Lstatic_mask = torch.sigmoid(gaussians.static_mask_logit).mean()
                        if iteration > 3500:
                            # 3500~4500 iter warmup
                            warmup = min(1.0, (iteration - 3500) / 1000)
                            current_lambda_static_mask = opt.lambda_static_mask * warmup

                            loss = loss + current_lambda_static_mask * Lstatic_mask

                    if hasattr(gaussians, "dynamic_mask_logit") and gaussians.dynamic_mask_logit.numel() > 0:
                        soft_dyn = torch.sigmoid(gaussians.dynamic_mask_logit).view(-1)
                        
                        st = gaussians.get_scaling_t.detach().view(-1)
                        gate = (st < opt.scale_t_threshold).float()

                        # 时间感知统计量
                        vis = (gaussians.dynamic_vis_denom / (gaussians.denom + 1e-6)).detach().view(-1)
                        motion = (gaussians.dynamic_motion_accum / (gaussians.dynamic_vis_denom + 1e-6)).detach().view(-1)
                        tsup = (gaussians.dynamic_time_support_accum / (gaussians.dynamic_vis_denom + 1e-6)).detach().view(-1)

                        vis_norm = vis / (vis.mean() + 1e-6)
                        motion_norm = motion / (motion.mean() + 1e-6)
                        tsup_norm = tsup / (tsup.mean() + 1e-6)

                        # 时间感知权重：低可见、弱运动、短时支撑 更容易剪
                        # exp 形式权重：统计量越大，权重越小
                        beta_vis = 0.5
                        beta_motion = 1.0
                        beta_tsup = 0.25

                        weight = torch.exp(-beta_vis * vis_norm)
                        weight = weight * torch.exp(-beta_motion * motion_norm)
                        weight = weight * torch.exp(-beta_tsup * tsup_norm)

                        effective_weight = gate * weight
                        Ldynamic_mask = (effective_weight * soft_dyn).sum() / (effective_weight.sum() + 1e-6)

                        if iteration > 3500:
                            warmup = min(1.0, (iteration - 3500) / 500)
                            current_lambda_dynamic_mask = opt.lambda_dynamic_mask * warmup
                            loss = loss + current_lambda_dynamic_mask * Ldynamic_mask
                ###################################

                ############SH Loss################
                if opt.use_sh_adaptive and opt.lambda_sh > 0:
                    sh_weights = torch.tensor([3/15, 5/15, 7/15], device="cuda")
                    total_sh_loss = torch.tensor(0.0, device="cuda")

                    current_lambda_sh = 0.0
                    if iteration >= sh_mask_start_iter:
                        warmup = min(1.0, (iteration - sh_mask_start_iter) / sh_mask_warmup_iters)
                        current_lambda_sh = opt.lambda_sh * warmup

                    if hasattr(gaussians, 'dynamic_sh_mask_logit') and gaussians.dynamic_sh_mask_logit.numel() > 0:
                        soft_dyn = torch.sigmoid(gaussians.dynamic_sh_mask_logit)
                        L_dynamic_sh = (soft_dyn * sh_weights).sum(dim=1).mean()
                        loss = loss + current_lambda_sh * L_dynamic_sh
                        total_sh_loss = total_sh_loss + L_dynamic_sh

                    if hasattr(gaussians, 'static_sh_mask_logit') and gaussians.static_sh_mask_logit.numel() > 0:
                        soft_static = torch.sigmoid(gaussians.static_sh_mask_logit)
                        L_static_sh = (soft_static * sh_weights).sum(dim=1).mean()
                        loss = loss + current_lambda_sh * L_static_sh
                        total_sh_loss = total_sh_loss + L_static_sh

                    Lsh = total_sh_loss
                #######################################

                loss = loss / batch_size
                loss.backward()
                batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)

                static = False
                if len(viewspace_point_tensor_static) > 0:
                    static = True
                if static:
                    batch_point_grad_static.append(torch.norm(viewspace_point_tensor_static.grad[:,:2], dim=-1))
                    batch_radii_static.append(radii_static)
                    batch_visibility_filter_static.append(visibility_filter_static)

            if batch_size > 1:
                visibility_count = torch.stack(batch_visibility_filter,1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii,1).max(1)[0]
                
                batch_alpha_stat_agg = None
                batch_motion_stat_agg = None
                batch_time_support_stat_agg = None

                batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)
                batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)

                if static:
                    visibility_count_static = torch.stack(batch_visibility_filter_static,1).sum(1)
                    visibility_filter_static = visibility_count_static > 0
                    radii_static = torch.stack(batch_radii_static,1).max(1)[0]
                
                    batch_viewspace_point_grad_static = torch.stack(batch_point_grad_static,1).sum(1)
                    batch_viewspace_point_grad_static[visibility_filter_static] = batch_viewspace_point_grad_static[visibility_filter_static] * batch_size / visibility_count_static[visibility_filter_static]
                    batch_viewspace_point_grad_static = batch_viewspace_point_grad_static.unsqueeze(1)
                
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone()[:,0].detach()
                    batch_t_grad[visibility_filter] = batch_t_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                    batch_t_grad = batch_t_grad.unsqueeze(1)
                    
                    alpha_stack = torch.stack(batch_alpha_stat, dim=1)          # [N, B, 1]
                    motion_stack = torch.stack(batch_motion_stat, dim=1)        # [N, B, 1]
                    tsup_stack = torch.stack(batch_time_support_stat, dim=1)    # [N, B, 1]

                    vis_count_unsq = visibility_count.unsqueeze(1).clamp_min(1).float()

                    batch_alpha_stat_agg = alpha_stack.sum(dim=1) / vis_count_unsq
                    batch_motion_stat_agg = motion_stack.sum(dim=1) / vis_count_unsq
                    batch_time_support_stat_agg = tsup_stack.sum(dim=1) / vis_count_unsq
            else:
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone().detach()
            
            iter_end.record()
            loss_dict = {"Ll1": Ll1,
                        "Lssim": Lssim}

            with torch.no_grad():
                psnr_for_log = psnr(image, gt_image).mean().double()
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_l1loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_l1loss_for_log
                ema_ssimloss_for_log = 0.4 * Lssim.item() + 0.6 * ema_ssimloss_for_log
                
                for lambda_name in lambda_all:
                    if opt.__dict__[lambda_name] > 0:
                        ema = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                        vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"] = 0.4 * vars()[f"L{lambda_name.replace('lambda_', '')}"].item() + 0.6*ema
                        loss_dict[lambda_name.replace("lambda_", "L")] = vars()[lambda_name.replace("lambda_", "L")]
                        
                if iteration % 10 == 0:
                    postfix = {"Loss": f"{ema_loss_for_log:.{7}f}",
                                            "PSNR": f"{psnr_for_log:.{2}f}",
                                            "Ll1": f"{ema_l1loss_for_log:.{4}f}",
                                            "Lssim": f"{ema_ssimloss_for_log:.{4}f}",
                                            "points": scene.gaussians.get_xyz.shape[0],
                                            "static": scene.gaussians.get_static_xyz.shape[0]}
                    
                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            ema_loss = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                            postfix[lambda_name.replace("lambda_", "L")] = f"{ema_loss:.{4}f}"
                            
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                test_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_dict)
                if (iteration in testing_iterations):
                    if test_psnr >= best_psnr:
                        best_psnr = test_psnr
                        print("\n[ITER {}] Saving best checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_best.pth")
                        
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter and (opt.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < opt.densify_until_num_points):
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    if static:
                        gaussians.static_max_radii2D[visibility_filter_static] = torch.max(gaussians.static_max_radii2D[visibility_filter_static], radii_static[visibility_filter_static])
                    if batch_size == 1:
                        gaussians.add_densification_stats(
                            viewspace_point_tensor,
                            visibility_filter,
                            batch_t_grad if gaussians.gaussian_dim == 4 else None,
                            alpha_stat=alpha_stat,
                            motion_stat=motion_stat,
                            time_support_stat=time_support_stat,
                        )
                    else:
                        gaussians.add_densification_stats_grad(
                            batch_viewspace_point_grad,
                            visibility_filter,
                            batch_t_grad if gaussians.gaussian_dim == 4 else None,
                            alpha_stat=batch_alpha_stat_agg if gaussians.gaussian_dim == 4 else None,
                            motion_stat=batch_motion_stat_agg if gaussians.gaussian_dim == 4 else None,
                            time_support_stat=batch_time_support_stat_agg if gaussians.gaussian_dim == 4 else None,
                        )
                        if static:
                            gaussians.add_densification_stats_grad_static(
                                batch_viewspace_point_grad_static,
                                visibility_filter_static
                            )

                    if iteration > opt.densify_from_iter: 
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        if iteration % opt.densification_interval == 0: 
                            gaussians.densify_and_prune(opt.densify_grad_threshold, opt.thresh_opa_prune, scene.cameras_extent, size_threshold, opt.densify_grad_t_threshold)
                            gaussians.dynamic2static(opt.scale_t_threshold)
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
                        
                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if pipe.env_map_res and iteration < pipe.env_optimize_until:
                        env_map_optimizer.step()
                        env_map_optimizer.zero_grad(set_to_none = True)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_dict=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        if loss_dict is not None:
            if "Lrigid" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/rigid_loss', loss_dict['Lrigid'].item(), iteration)
            if "Ldepth" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/depth_loss', loss_dict['Ldepth'].item(), iteration)
            if "Ltv" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/tv_loss', loss_dict['Ltv'].item(), iteration)
            if "Lopa" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/opa_loss', loss_dict['Lopa'].item(), iteration)
            if "Lptsopa" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/pts_opa_loss', loss_dict['Lptsopa'].item(), iteration)
            if "Lsmooth" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/smooth_loss', loss_dict['Lsmooth'].item(), iteration)
            if "Llaplacian" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/laplacian_loss', loss_dict['Llaplacian'].item(), iteration)


        tb_writer.add_scalar('gpu/memory_allocated_MB', torch.cuda.memory_allocated() / 1e6, iteration)
        tb_writer.add_scalar('gpu/memory_reserved_MB', torch.cuda.memory_reserved() / 1e6, iteration)

    psnr_test_iter = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        validation_configs = ({'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
                              {'name': 'test', 'cameras' : [scene.getTestCameras()[idx] for idx in range(len(scene.getTestCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                msssim_test = 0.0
                for idx, batch_data in enumerate(tqdm(config['cameras'])):
                    gt_image, viewpoint = batch_data
                    gt_image = gt_image.cuda()
                    viewpoint = viewpoint.cuda()
                    
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    
                    depth = easy_cmap(render_pkg['depth'][0])
                    alpha = torch.clamp(render_pkg['alpha'], 0.0, 1.0).repeat(3,1,1)
                    image_4d = torch.clamp(render_pkg["render_4d"], 0.0, 1.0)
                    image_3d = torch.clamp(render_pkg["render_3d"], 0.0, 1.0)

                    if tb_writer and (idx < 5):
                        grid = [gt_image, image, image_4d, image_3d]
                        grid = make_grid(grid, nrow=2)
                        tb_writer.add_images(config['name'] + "_view_{}/gt_vs_render".format(viewpoint.image_name), grid[None], global_step=iteration)
                            
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    msssim_test += msssim(image[None].cpu(), gt_image[None].cpu())
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 
                ssim_test /= len(config['cameras'])     
                msssim_test /= len(config['cameras'])        
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - msssim', msssim_test, iteration)
                if config['name'] == 'test':
                    psnr_test_iter = psnr_test.item()
                    
    torch.cuda.empty_cache()
    return psnr_test_iter


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[6_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[6_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--val", action="store_true", default=False)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
        
    cfg = OmegaConf.load(args.config)

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)
        
    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [i for i in range(0,args.iterations,500)]
    
    setup_seed(args.seed)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if args.val == False:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from,
                args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.batch_size)

    else:
        validation(lp.extract(args), op.extract(args), pp.extract(args),args.start_checkpoint,args.gaussian_dim, 
                   args.time_duration,args.rot_4d, args.force_sh_3d, args.num_pts, args.num_pts_ratio)
        

    print("\nComplete.")
