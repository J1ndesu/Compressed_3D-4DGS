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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_rotation_4d, build_scaling_rotation_4d, rotation_matrix_to_rotation_3d
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import sh_channels_4d
from utils.compression_utils import get_ste_mask, get_sh_masks

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L.transpose(1, 2) @ L
            symm = strip_symmetric(actual_covariance)
            return symm
        
        def build_covariance_from_scaling_rotation_4d(scaling, scaling_modifier, rotation_l, rotation_r, dt=0.0):
            L = build_scaling_rotation_4d(scaling_modifier * scaling, rotation_l, rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            cov_11 = actual_covariance[:,:3,:3]
            cov_12 = actual_covariance[:,0:3,3:4]
            cov_t = actual_covariance[:,3:4,3:4]
            current_covariance = cov_11 - cov_12 @ cov_12.transpose(1, 2) / cov_t
            symm = strip_symmetric(current_covariance)
            if dt.shape[1] > 1:
                mean_offset = (cov_12.squeeze(-1) / cov_t.squeeze(-1))[:, None, :] * dt[..., None]
                mean_offset = mean_offset[..., None]  # [num_pts, num_time, 3, 1]
            else:
                mean_offset = cov_12.squeeze(-1) / cov_t.squeeze(-1) * dt
            return symm, mean_offset.squeeze(-1)
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        if not self.rot_4d:
            self.covariance_activation = build_covariance_from_scaling_rotation
        else:
            self.covariance_activation = build_covariance_from_scaling_rotation_4d

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, gaussian_dim : int = 3, time_duration: list = [-0.5, 0.5], rot_4d: bool = False, force_sh_3d: bool = False, sh_degree_t : int = 0):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        self.gaussian_dim = gaussian_dim
        self._t = torch.empty(0)
        self._scaling_t = torch.empty(0)
        self.time_duration = time_duration
        self.rot_4d = rot_4d
        self._rotation_r = torch.empty(0)
        self.force_sh_3d = force_sh_3d
        self.t_gradient_accum = torch.empty(0)

        self.dynamic_vis_denom = torch.empty(0)
        self.dynamic_alpha_accum = torch.empty(0)
        self.dynamic_motion_accum = torch.empty(0)
        self.dynamic_time_support_accum = torch.empty(0)

        if self.rot_4d or self.force_sh_3d:
            assert self.gaussian_dim == 4
        self.env_map = torch.empty(0)
        
        self.active_sh_degree_t = 0
        self.max_sh_degree_t = sh_degree_t

        self.static_xyz = torch.empty(0, device="cuda")
        self.static_features_dc = torch.empty(0, device="cuda")
        self.static_features_rest = torch.empty(0, device="cuda")
        self.static_scaling = torch.empty(0, device="cuda")
        self.static_rotation = torch.empty(0, device="cuda")
        self.static_opacity = torch.empty(0, device="cuda")
        self.static_max_radii2D = torch.empty(0)
        self.static_denom = torch.empty(0)
        self.static_xyz_gradient_accum = torch.empty(0)
        
        self.use_pruning = False
        self.use_sh_adaptive = False
        self.enable_sh_mask = False

        self.setup_functions()

    def capture(self):
        if self.gaussian_dim == 3:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        elif self.gaussian_dim == 4:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.t_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self._t,
                self._scaling_t,
                self._rotation_r,
                self.rot_4d,
                self.env_map,
                self.active_sh_degree_t,
                self.static_xyz,
                self.static_features_dc,
                self.static_features_rest,
                self.static_scaling,
                self.static_rotation,
                self.static_opacity,
                self.static_max_radii2D,
                self.static_xyz_gradient_accum,
                self.static_denom,
                self.dynamic_mask_logit,
                self.static_mask_logit,
                self.dynamic_vis_denom,
                self.dynamic_alpha_accum,
                self.dynamic_motion_accum,
                self.dynamic_time_support_accum,
                self.dynamic_sh_mask_logit,
                self.static_sh_mask_logit
            )
    
    def restore(self, model_args, training_args):
        if self.gaussian_dim == 3:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        elif self.gaussian_dim == 4:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            t_gradient_accum,
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            self._t,
            self._scaling_t,
            self._rotation_r,
            self.rot_4d,
            self.env_map,
            self.active_sh_degree_t,
            self.static_xyz,
            self.static_features_dc,
            self.static_features_rest,
            self.static_scaling,
            self.static_rotation,
            self.static_opacity,
            self.static_max_radii2D,
            self.static_xyz_gradient_accum,
            self.static_denom,
            self.dynamic_mask_logit,
            self.static_mask_logit,
            self.dynamic_vis_denom,
            self.dynamic_alpha_accum,
            self.dynamic_motion_accum,
            self.dynamic_time_support_accum,
            self.dynamic_sh_mask_logit,
            self.static_sh_mask_logit) = model_args

        self.xyz_gradient_accum = xyz_gradient_accum
        self.t_gradient_accum = t_gradient_accum
        self.denom = denom
        if training_args is not None:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)
            self.enable_sh_mask = False
        
        if training_args is None:
            self.use_pruning = (
                (hasattr(self, "dynamic_mask_logit") and self.dynamic_mask_logit is not None and self.dynamic_mask_logit.numel() > 0) or
                (hasattr(self, "static_mask_logit") and self.static_mask_logit is not None and self.static_mask_logit.numel() > 0)
            )
            self.use_sh_adaptive = (
                (hasattr(self, "dynamic_sh_mask_logit") and self.dynamic_sh_mask_logit is not None and self.dynamic_sh_mask_logit.numel() > 0) or
                (hasattr(self, "static_sh_mask_logit") and self.static_sh_mask_logit is not None and self.static_sh_mask_logit.numel() > 0)
            )
            self.enable_sh_mask = self.use_sh_adaptive

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t)
    
    @property
    def get_scaling_xyzt(self):
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_t], dim = 1))
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_rotation_r(self):
        return self.rotation_activation(self._rotation_r)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_t(self):
        return self._t
    
    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim = 1)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_static_xyz(self):
        return self.static_xyz
    
    @property
    def get_static_features(self):
        features_dc = self.static_features_dc
        features_rest = self.static_features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_static_opacity(self):
        return self.opacity_activation(self.static_opacity)
    
    @property
    def get_static_scaling(self):
        return self.scaling_activation(self.static_scaling)
    
    @property
    def get_static_rotation(self):
        if len(self.static_rotation) == 0:
            return self.static_rotation
        return self.rotation_activation(self.static_rotation)
    
    @property
    def get_max_sh_channels(self):
        if self.gaussian_dim == 3 or self.force_sh_3d:
            return (self.max_sh_degree+1)**2
        elif self.gaussian_dim == 4 and self.max_sh_degree_t == 0:
            return sh_channels_4d[self.max_sh_degree]
        elif self.gaussian_dim == 4 and self.max_sh_degree_t > 0:
            return (self.max_sh_degree+1)**2 * (self.max_sh_degree_t + 1)
    
    def get_cov_t(self, scaling_modifier = 1):
        if self.rot_4d:
            L = build_scaling_rotation_4d(scaling_modifier * self.get_scaling_xyzt, self._rotation, self._rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance[:,3,3].unsqueeze(1)
        else:
            return self.get_scaling_t * scaling_modifier

    def get_marginal_t(self, timestamp, scaling_modifier = 1): # Standard
        sigma = self.get_cov_t(scaling_modifier)
        return torch.exp(-0.5*(self.get_t-timestamp)**2/sigma) # / torch.sqrt(2*torch.pi*sigma)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def get_current_covariance_and_mean_offset(self, scaling_modifier = 1, timestamp = 0.0):
        return self.covariance_activation(self.get_scaling_xyzt, scaling_modifier, 
                                                              self._rotation, 
                                                              self._rotation_r,
                                                              dt = timestamp - self.get_t)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))

        if self.active_sh_degree > 0:
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
               l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        if self.active_sh_degree != 0:
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        # TODO: may need to add empty shs for SIBR_viewer?
        if self.active_sh_degree > 0:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        

    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        elif self.max_sh_degree_t and self.active_sh_degree_t < self.max_sh_degree_t:
            self.active_sh_degree_t += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        if self.gaussian_dim == 4:
            if pcd.time is None:
                fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device="cuda") * 1.2 - 0.1) * (self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                fused_times = torch.from_numpy(pcd.time).cuda().float()
            
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        if self.gaussian_dim == 4:
            # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
            dist_t = torch.zeros_like(fused_times, device="cuda") + (self.time_duration[1] - self.time_duration[0]) / 5
            scales_t = torch.log(torch.sqrt(dist_t))
            if self.rot_4d:
                rots_r = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                rots_r[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            if self.rot_4d:
                self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))


    def create_from_pth(self, path, spatial_lr_scale):
        assert self.gaussian_dim == 4 and self.rot_4d
        self.spatial_lr_scale = spatial_lr_scale
        init_4d_gaussian = torch.load(path)
        fused_point_cloud = init_4d_gaussian['xyz'].cuda()
        features_dc = init_4d_gaussian['features_dc'].cuda()
        features_rest = init_4d_gaussian['features_rest'].cuda()
        fused_times = init_4d_gaussian['t'].cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = init_4d_gaussian['scaling'].cuda()
        rots = init_4d_gaussian['rotation'].cuda()
        scales_t = init_4d_gaussian['scaling_t'].cuda()
        rots_r = init_4d_gaussian['rotation_r'].cuda()

        opacities = init_4d_gaussian['opacity'].cuda()

        if init_4d_gaussian['static_xyz'] is not None:
            static_xyz = init_4d_gaussian['static_xyz'].cuda()
            static_features_dc = init_4d_gaussian['static_features_dc'].cuda()
            static_features_rest = init_4d_gaussian['static_features_rest'].cuda()
            static_scaling = init_4d_gaussian['static_scaling'].cuda()
            static_rotation = init_4d_gaussian['static_rotation'].cuda()
            static_opacity = init_4d_gaussian['static_opacity'].cuda()

            self.static_xyz = nn.Parameter(static_xyz.requires_grad_(True))
            self.static_features_dc = nn.Parameter(static_features_dc.requires_grad_(True))
            self.static_features_rest = nn.Parameter(static_features_rest.requires_grad_(True))
            self.static_scaling = nn.Parameter(static_scaling.requires_grad_(True))
            self.static_rotation = nn.Parameter(static_rotation.requires_grad_(True))
            self.static_opacity = nn.Parameter(static_opacity.requires_grad_(True))
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.transpose(1, 2).requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        self._t = nn.Parameter(fused_times.requires_grad_(True))
        self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
        self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

    def training_setup(self, training_args):
        self.use_pruning = training_args.use_pruning
        self.use_sh_adaptive = training_args.use_sh_adaptive
        self.use_vq = training_args.use_vq
        self.enable_sh_mask = False

        self.static_mask_lr = training_args.static_mask_lr
        self.dynamic_mask_lr = training_args.dynamic_mask_lr
        self.sh_mask_lr = training_args.sh_mask_lr
        self.codebook_lr = training_args.codebook_lr
      
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.gaussian_dim == 4: # TODO: tune time_lr_scale
            if training_args.position_t_lr_init < 0:
                training_args.position_t_lr_init = training_args.position_lr_init
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

            self.dynamic_vis_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.dynamic_alpha_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.dynamic_motion_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.dynamic_time_support_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

            l.append({'params': [self._t], 'lr': training_args.position_t_lr_init * self.spatial_lr_scale, "name": "t"})
            l.append({'params': [self._scaling_t], 'lr': training_args.scaling_lr, "name": "scaling_t"})
            if self.rot_4d:
                l.append({'params': [self._rotation_r], 'lr': training_args.rotation_lr, "name": "rotation_r"})

            l.append({'params': [self.static_xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "static_xyz"}),
            l.append({'params': [self.static_features_dc], 'lr': training_args.feature_lr, "name": "static_f_dc"}),
            l.append({'params': [self.static_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "static_f_rest"}),
            l.append({'params': [self.static_opacity], 'lr': training_args.opacity_lr, "name": "static_opacity"}),
            l.append({'params': [self.static_scaling], 'lr': training_args.scaling_lr, "name": "static_scaling"}),
            l.append({'params': [self.static_rotation], 'lr': training_args.rotation_lr, "name": "static_rotation"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        dyn_num = self._xyz.shape[0]
        if self.use_pruning:
            self.static_mask_lr = training_args.static_mask_lr
            self.dynamic_mask_lr = training_args.dynamic_mask_lr
            self.lambda_static_mask = training_args.lambda_static_mask
            self.lambda_dynamic_mask = training_args.lambda_dynamic_mask
            self.phi_threshold = training_args.phi_threshold
            self.static_mask_logit = torch.empty((0, 1), device="cuda", dtype=torch.float32)            
            dyn_init = torch.full((dyn_num, 1), 0.0, device="cuda", dtype=torch.float32)
            self.dynamic_mask_logit = nn.Parameter(dyn_init.requires_grad_(True))

            self.optimizer.add_param_group({
                'params': [self.dynamic_mask_logit],
                'lr': self.dynamic_mask_lr,
                "name": "dynamic_mask_logit"
            })
        else:
            self.static_mask_logit = torch.empty((0, 1), device="cuda", dtype=torch.float32)
            self.dynamic_mask_logit = torch.empty((0, 1), device="cuda", dtype=torch.float32)

        if self.use_sh_adaptive:
            self.sh_mask_lr = training_args.sh_mask_lr
            self.lambda_sh = training_args.lambda_sh
            self.phi_prune_sh = training_args.phi_prune_sh
            self.static_sh_mask_logit = torch.empty((0, 3), device="cuda", dtype=torch.float32)
            dynamic_sh_mask_init = torch.full((dyn_num, 3), 0.0, device="cuda", dtype=torch.float32)
            self.dynamic_sh_mask_logit = nn.Parameter(dynamic_sh_mask_init.requires_grad_(True))

            self.optimizer.add_param_group({
                'params': [self.dynamic_sh_mask_logit],
                'lr': self.sh_mask_lr,
                "name": "dynamic_sh_mask_logit"
            })
        else:
            self.dynamic_sh_mask_logit = torch.empty((0, 3), device="cuda", dtype=torch.float32)
            self.static_sh_mask_logit = torch.empty((0, 3), device="cuda", dtype=torch.float32)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            # if param_group["name"] == "t" and self.gaussian_dim == 4:
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        if len(self.static_opacity) != 0:
            static_opacities_new = inverse_sigmoid(torch.min(self.get_static_opacity, torch.ones_like(self.get_static_opacity)*0.01))
            optimizable_tensors = self.replace_tensor_to_optimizer(static_opacities_new, "static_opacity")
            self.static_opacity = optimizable_tensors["static_opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask, static=False):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if static and 'static' not in group["name"]:
                continue
            elif not static and 'static' in group["name"]:
                continue

            params = group["params"][0]
        
            if params.shape[0] != mask.shape[0]:
                # print(f"Skipping {group['name']} due to shape mismatch")
                optimizable_tensors[group["name"]] = params
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        
        if self.optimizer is None:
            self._xyz = nn.Parameter(self._xyz[valid_points_mask].requires_grad_(True))
            self._features_dc = nn.Parameter(self._features_dc[valid_points_mask].requires_grad_(True))
            self._features_rest = nn.Parameter(self._features_rest[valid_points_mask].requires_grad_(True))
            self._opacity = nn.Parameter(self._opacity[valid_points_mask].requires_grad_(True))
            self._scaling = nn.Parameter(self._scaling[valid_points_mask].requires_grad_(True))
            self._rotation = nn.Parameter(self._rotation[valid_points_mask].requires_grad_(True))

            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]

            if self.gaussian_dim == 4:
                self._t = nn.Parameter(self._t[valid_points_mask].requires_grad_(True))
                self._scaling_t = nn.Parameter(self._scaling_t[valid_points_mask].requires_grad_(True))
                if self.rot_4d:
                    self._rotation_r = nn.Parameter(self._rotation_r[valid_points_mask].requires_grad_(True))
                self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]

                if hasattr(self, "dynamic_vis_denom"):
                    self.dynamic_vis_denom = self.dynamic_vis_denom[valid_points_mask]
                if hasattr(self, "dynamic_alpha_accum"):
                    self.dynamic_alpha_accum = self.dynamic_alpha_accum[valid_points_mask]
                if hasattr(self, "dynamic_motion_accum"):
                    self.dynamic_motion_accum = self.dynamic_motion_accum[valid_points_mask]
                if hasattr(self, "dynamic_time_support_accum"):
                    self.dynamic_time_support_accum = self.dynamic_time_support_accum[valid_points_mask]

            if self.use_pruning and hasattr(self, "dynamic_mask_logit") and self.dynamic_mask_logit is not None and self.dynamic_mask_logit.numel() > 0:
                self.dynamic_mask_logit = nn.Parameter(self.dynamic_mask_logit[valid_points_mask].requires_grad_(True))
            
            if self.use_sh_adaptive and hasattr(self, "dynamic_sh_mask_logit") and self.dynamic_sh_mask_logit is not None and self.dynamic_sh_mask_logit.numel() > 0:
                self.dynamic_sh_mask_logit = nn.Parameter(self.dynamic_sh_mask_logit[valid_points_mask].requires_grad_(True))

            return
        
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]

            self.dynamic_vis_denom = self.dynamic_vis_denom[valid_points_mask]
            self.dynamic_alpha_accum = self.dynamic_alpha_accum[valid_points_mask]
            self.dynamic_motion_accum = self.dynamic_motion_accum[valid_points_mask]
            self.dynamic_time_support_accum = self.dynamic_time_support_accum[valid_points_mask]

        if "dynamic_mask_logit" in optimizable_tensors:
            self.dynamic_mask_logit = optimizable_tensors["dynamic_mask_logit"]

        if "dynamic_sh_mask_logit" in optimizable_tensors:
            self.dynamic_sh_mask_logit = optimizable_tensors["dynamic_sh_mask_logit"]

    def prune_static_points(self, mask):
        valid_points_mask = ~mask
        
        if self.optimizer is None:
            self.static_xyz = nn.Parameter(self.static_xyz[valid_points_mask].requires_grad_(True))
            self.static_features_dc = nn.Parameter(self.static_features_dc[valid_points_mask].requires_grad_(True))
            self.static_features_rest = nn.Parameter(self.static_features_rest[valid_points_mask].requires_grad_(True))
            self.static_opacity = nn.Parameter(self.static_opacity[valid_points_mask].requires_grad_(True))
            self.static_scaling = nn.Parameter(self.static_scaling[valid_points_mask].requires_grad_(True))
            self.static_rotation = nn.Parameter(self.static_rotation[valid_points_mask].requires_grad_(True))

            self.static_xyz_gradient_accum = self.static_xyz_gradient_accum[valid_points_mask]
            self.static_denom = self.static_denom[valid_points_mask]
            self.static_max_radii2D = self.static_max_radii2D[valid_points_mask]

            if self.use_pruning and hasattr(self, "static_mask_logit") and self.static_mask_logit is not None and self.static_mask_logit.numel() > 0:
                self.static_mask_logit = nn.Parameter(self.static_mask_logit[valid_points_mask].requires_grad_(True))

            if self.use_sh_adaptive and hasattr(self, "static_sh_mask_logit") and self.static_sh_mask_logit is not None and self.static_sh_mask_logit.numel() > 0:
                self.static_sh_mask_logit = nn.Parameter(self.static_sh_mask_logit[valid_points_mask].requires_grad_(True))

            return
        
        optimizable_tensors = self._prune_optimizer(valid_points_mask, static=True)

        self.static_xyz = optimizable_tensors["static_xyz"]
        self.static_features_dc = optimizable_tensors["static_f_dc"]
        self.static_features_rest = optimizable_tensors["static_f_rest"]
        self.static_opacity = optimizable_tensors["static_opacity"]
        self.static_scaling = optimizable_tensors["static_scaling"]
        self.static_rotation = optimizable_tensors["static_rotation"]

        self.static_xyz_gradient_accum = self.static_xyz_gradient_accum[valid_points_mask]
        self.static_denom = self.static_denom[valid_points_mask]
        self.static_max_radii2D = self.static_max_radii2D[valid_points_mask]

        if "static_mask_logit" in optimizable_tensors:
            self.static_mask_logit = optimizable_tensors["static_mask_logit"]
        
        if "static_sh_mask_logit" in optimizable_tensors:
            self.static_sh_mask_logit = optimizable_tensors["static_sh_mask_logit"]
        

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            try:
                extension_tensor = tensors_dict[group["name"]]
            except:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r, new_dynamic_mask_logit = None, new_dynamic_sh_mask_logit = None):

        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        }

        if self.gaussian_dim == 4:
            d["t"] = new_t
            d["scaling_t"] = new_scaling_t
            if self.rot_4d:
                d["rotation_r"] = new_rotation_r

        if self.use_pruning and new_dynamic_mask_logit is not None:
            group_names = [group["name"] for group in self.optimizer.param_groups]

            if "dynamic_mask_logit" not in group_names:
                self.dynamic_mask_logit = nn.Parameter(new_dynamic_mask_logit.requires_grad_(True))
                self.optimizer.add_param_group({
                    'params': [self.dynamic_mask_logit],
                    'lr': self.dynamic_mask_lr,
                    "name": "dynamic_mask_logit"
                })
            else:
                d["dynamic_mask_logit"] = new_dynamic_mask_logit
        
        if self.use_sh_adaptive and new_dynamic_sh_mask_logit is not None:
            group_names = [group["name"] for group in self.optimizer.param_groups]

            if "dynamic_sh_mask_logit" not in group_names:
                self.dynamic_sh_mask_logit = nn.Parameter(new_dynamic_sh_mask_logit.requires_grad_(True))
                self.optimizer.add_param_group({
                    'params': [self.dynamic_sh_mask_logit],
                    'lr': self.sh_mask_lr,
                    "name": "dynamic_sh_mask_logit"
                })
            else:
                d["dynamic_sh_mask_logit"] = new_dynamic_sh_mask_logit

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if "dynamic_mask_logit" in optimizable_tensors:
            self.dynamic_mask_logit = optimizable_tensors["dynamic_mask_logit"]

        if "dynamic_sh_mask_logit" in optimizable_tensors:
            self.dynamic_sh_mask_logit = optimizable_tensors["dynamic_sh_mask_logit"]

        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            
            self.dynamic_vis_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.dynamic_alpha_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.dynamic_motion_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.dynamic_time_support_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densification_postfix_static(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_static_mask_logit = None, new_static_sh_mask_logit = None):
        d = {"static_xyz": new_xyz,
        "static_f_dc": new_features_dc,
        "static_f_rest": new_features_rest,
        "static_opacity": new_opacities,
        "static_scaling" : new_scaling,
        "static_rotation" : new_rotation,
        }

        if self.use_pruning and new_static_mask_logit is not None:
            group_names = [group["name"] for group in self.optimizer.param_groups]          
            if "static_mask_logit" not in group_names:
                self.static_mask_logit = nn.Parameter(new_static_mask_logit.requires_grad_(True))
                self.optimizer.add_param_group({
                    'params': [self.static_mask_logit], 
                    'lr': self.static_mask_lr, 
                    "name": "static_mask_logit"
                })
            else:
                d["static_mask_logit"] = new_static_mask_logit
        
        if self.use_sh_adaptive and new_static_sh_mask_logit is not None:
            group_names = [group["name"] for group in self.optimizer.param_groups]          
            if "static_sh_mask_logit" not in group_names:
                self.static_sh_mask_logit = nn.Parameter(new_static_sh_mask_logit.requires_grad_(True))
                self.optimizer.add_param_group({
                    'params': [self.static_sh_mask_logit], 
                    'lr': self.sh_mask_lr, 
                    "name": "static_sh_mask_logit"
                })
            else:
                d["static_sh_mask_logit"] = new_static_sh_mask_logit

        optimizable_tensors = self.cat_tensors_to_optimizer(d)      
        self.static_xyz = optimizable_tensors["static_xyz"]
        self.static_features_dc = optimizable_tensors["static_f_dc"]
        self.static_features_rest = optimizable_tensors["static_f_rest"]
        self.static_opacity = optimizable_tensors["static_opacity"]
        self.static_scaling = optimizable_tensors["static_scaling"]
        self.static_rotation = optimizable_tensors["static_rotation"]
    
        if "static_mask_logit" in optimizable_tensors:
            self.static_mask_logit = optimizable_tensors["static_mask_logit"]

        if "static_sh_mask_logit" in optimizable_tensors:
            self.static_sh_mask_logit = optimizable_tensors["static_sh_mask_logit"]

        self.static_max_radii2D = torch.zeros((self.static_xyz.shape[0]), device="cuda")
        self.static_denom = torch.zeros((self.static_xyz.shape[0], 1), device="cuda")
        self.static_xyz_gradient_accum = torch.zeros((self.static_xyz.shape[0], 1), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        #if not tsplit:
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # print(f"num_to_densify_pos: {torch.where(padded_grad >= grad_threshold, True, False).sum()}, num_to_split_pos: {selected_pts_mask.sum()}")

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N)) 
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1) 
        
        if not self.rot_4d:
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_t = None
            new_scaling_t = None
            new_rotation_r = None
            if self.gaussian_dim == 4:
                stds_t = self.get_scaling_t[selected_pts_mask].repeat(N,1)
                means_t = torch.zeros((stds_t.size(0), 1),device="cuda")
                samples_t = torch.normal(mean=means_t, std=stds_t)
                new_t = samples_t + self.get_t[selected_pts_mask].repeat(N, 1)
                new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
        else:
            stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 4),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation_4d(self._rotation[selected_pts_mask], self._rotation_r[selected_pts_mask]).repeat(N,1,1)
            new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyzt[selected_pts_mask].repeat(N, 1) 

            new_xyz = new_xyzt[...,0:3] 
            new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
            new_t = new_xyzt[...,3:4]
            new_rotation_r = self._rotation_r[selected_pts_mask].repeat(N,1)

        new_dynamic_mask_logit = None
        if self.use_pruning:
            if hasattr(self, "dynamic_mask_logit") and self.dynamic_mask_logit.numel() > 0:
                new_dynamic_mask_logit = self.dynamic_mask_logit[selected_pts_mask]
                if N > 1:
                    new_dynamic_mask_logit = new_dynamic_mask_logit.repeat(N, 1)
            else:
                new_dynamic_mask_logit = torch.full((N * selected_pts_mask.sum(), 1), 0.0, device="cuda")

        new_dynaminc_sh_mask_logit = None
        if self.use_sh_adaptive:
            if hasattr(self, "dynamic_sh_mask_logit") and self.dynamic_sh_mask_logit.numel() > 0:
                new_dynaminc_sh_mask_logit = self.dynamic_sh_mask_logit[selected_pts_mask]
                if N > 1:
                    new_dynaminc_sh_mask_logit = new_dynaminc_sh_mask_logit.repeat(N, 1)
            else:            
                new_dynaminc_sh_mask_logit = torch.full((N * selected_pts_mask.sum(), 3), 0.0, device="cuda")

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r, new_dynamic_mask_logit, new_dynaminc_sh_mask_logit)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # print(f"num_to_densify_pos: {torch.where(grads >= grad_threshold, True, False).sum()}, num_to_clone_pos: {selected_pts_mask.sum()}")
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]

        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        
        new_rotation = self._rotation[selected_pts_mask]
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            if self.rot_4d:
                new_rotation_r = self._rotation_r[selected_pts_mask]

        new_dynamic_mask_logit = None
        if self.use_pruning:
            if hasattr(self, "dynamic_mask_logit") and self.dynamic_mask_logit.numel() > 0:
                new_dynamic_mask_logit = self.dynamic_mask_logit[selected_pts_mask]
            else:
                new_dynamic_mask_logit = torch.full((selected_pts_mask.sum(), 1), 0.0, device="cuda") #0.5

        new_dynamic_sh_mask_logit = None
        if self.use_sh_adaptive:
            if hasattr(self, "dynamic_sh_mask_logit") and self.dynamic_sh_mask_logit.numel() > 0:
                new_dynamic_sh_mask_logit = self.dynamic_sh_mask_logit[selected_pts_mask]
            else:            
                new_dynamic_sh_mask_logit = torch.full((selected_pts_mask.sum(), 3), 0.0, device="cuda") #0.5

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r, new_dynamic_mask_logit, new_dynamic_sh_mask_logit)

    def densify_and_split_static(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_static_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_static_scaling, dim=1).values > self.percent_dense*scene_extent)

        new_scaling = self.scaling_inverse_activation(self.get_static_scaling[selected_pts_mask].repeat(N,1) / (0.8*N)) 
        new_rotation = self.static_rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self.static_features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self.static_features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self.static_opacity[selected_pts_mask].repeat(N,1) 

        stds = self.get_static_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.static_rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_static_xyz[selected_pts_mask].repeat(N, 1)

        new_mask_logit = None
        if self.use_pruning:
            if hasattr(self, "static_mask_logit") and self.static_mask_logit.numel() > 0:
                new_mask_logit = self.static_mask_logit[selected_pts_mask]
                if N > 1:
                    new_mask_logit = new_mask_logit.repeat(N, 1)
            else:
                new_mask_logit = torch.full((N * selected_pts_mask.sum(), 1), 0.0, device="cuda")#0.5

        new_static_sh_mask_logit = None
        if self.use_sh_adaptive:
            if hasattr(self, "static_sh_mask_logit") and self.static_sh_mask_logit.numel() > 0:
                new_static_sh_mask_logit = self.static_sh_mask_logit[selected_pts_mask]
                if N > 1:
                    new_static_sh_mask_logit = new_static_sh_mask_logit.repeat(N, 1)
            else:            
                new_static_sh_mask_logit = torch.full((N * selected_pts_mask.sum(), 3), 0.0, device="cuda")#0.5
        
        self.densification_postfix_static(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_mask_logit, new_static_sh_mask_logit)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_static_points(prune_filter)

    def densify_and_clone_static(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_static_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self.static_xyz[selected_pts_mask]
        new_features_dc = self.static_features_dc[selected_pts_mask]
        new_features_rest = self.static_features_rest[selected_pts_mask]
        new_opacities = self.static_opacity[selected_pts_mask]
        new_scaling = self.static_scaling[selected_pts_mask]
        new_rotation = self.static_rotation[selected_pts_mask]

        new_mask_logit = None
        if self.use_pruning:
            if hasattr(self, "static_mask_logit") and self.static_mask_logit.numel() > 0:
                new_mask_logit = self.static_mask_logit[selected_pts_mask]
            else:
                new_mask_logit = torch.full((selected_pts_mask.sum(), 1), 0.0, device="cuda") #0.5

        new_static_sh_mask_logit = None
        if self.use_sh_adaptive:
            if hasattr(self, "static_sh_mask_logit") and self.static_sh_mask_logit.numel() > 0:
                new_static_sh_mask_logit = self.static_sh_mask_logit[selected_pts_mask]
            else:            
                new_static_sh_mask_logit = torch.full((selected_pts_mask.sum(), 3), 0.0, device="cuda") #0.5
        
        self.densification_postfix_static(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_mask_logit, new_static_sh_mask_logit)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_t=None, prune_only=False, dynamic_only=False):
        if not prune_only:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            # if self.gaussian_dim == 4:
            #     grads_t = self.t_gradient_accum / self.denom
            #     grads_t[grads_t.isnan()] = 0.0
            # else:
            #     grads_t = None

            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)
            if len(self.static_xyz) != 0 and not dynamic_only:
                static_grads = self.static_xyz_gradient_accum / self.static_denom
                static_grads[static_grads.isnan()] = 0.0
                self.densify_and_clone_static(static_grads, max_grad, extent)
                self.densify_and_split_static(static_grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if len(self.static_xyz) != 0 and not dynamic_only:
            prune_static_mask = (self.get_static_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

            if len(self.static_xyz) != 0 and not dynamic_only:
                big_points_vs_static = self.static_max_radii2D > max_screen_size
                big_points_ws_static = self.get_static_scaling.max(dim=1).values > 0.1 * extent
                prune_static_mask = torch.logical_or(torch.logical_or(prune_static_mask, big_points_vs_static), big_points_ws_static)
        self.prune_points(prune_mask)
        if len(self.static_xyz) != 0 and not dynamic_only:
            self.prune_static_points(prune_static_mask) 

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, avg_t_grad=None, alpha_stat=None, motion_stat=None, time_support_stat=None):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]

            self.dynamic_vis_denom[update_filter] += 1

            if alpha_stat is not None:
                self.dynamic_alpha_accum[update_filter] += alpha_stat[update_filter]

            if motion_stat is not None:
                self.dynamic_motion_accum[update_filter] += motion_stat[update_filter]

            if time_support_stat is not None:
                self.dynamic_time_support_accum[update_filter] += time_support_stat[update_filter]
        
    def add_densification_stats_grad(self, viewspace_point_grad, update_filter, avg_t_grad=None, alpha_stat=None, motion_stat=None, time_support_stat=None):
        self.xyz_gradient_accum[update_filter] += viewspace_point_grad[update_filter]
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]

        self.dynamic_vis_denom[update_filter] += 1

        if alpha_stat is not None:
            self.dynamic_alpha_accum[update_filter] += alpha_stat[update_filter]

        if motion_stat is not None:
            self.dynamic_motion_accum[update_filter] += motion_stat[update_filter]

        if time_support_stat is not None:
            self.dynamic_time_support_accum[update_filter] += time_support_stat[update_filter]

    def add_densification_stats_grad_static(self, viewspace_point_grad, update_filter):
        self.static_xyz_gradient_accum[update_filter] += viewspace_point_grad[update_filter]
        self.static_denom[update_filter] += 1


    def dynamic2static(self, scale_threshold=3):
        static_mask = (self.get_scaling_t > scale_threshold).squeeze()
        if static_mask.sum() == 0:
            return
        new_xyz = self._xyz[static_mask]
        new_features_dc = self._features_dc[static_mask]
        new_features_rest = self._features_rest[static_mask]
        new_opacities = self._opacity[static_mask]
        r_4d = build_rotation_4d(self._rotation[static_mask], self._rotation_r[static_mask])
        r_3d = r_4d[:,:3,:3]
        new_rotation = rotation_matrix_to_rotation_3d(r_3d)
        new_scaling = self._scaling[static_mask]

        new_static_mask_logit = None
        if self.use_pruning:
            new_static_mask_logit = torch.full((static_mask.sum(), 1), 0.0, device="cuda")

        new_static_sh_mask_logit = None
        if self.use_sh_adaptive:
            new_static_sh_mask_logit = torch.full((static_mask.sum(), 3), 0.0, device="cuda")

        self.prune_points(static_mask)

        self.densification_postfix_static(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_static_mask_logit, new_static_sh_mask_logit)

    def prune_low_mask_points(self, threshold=0.1):
        """
        对静态点进行硬剪枝（只剪静态 3D 池）
        """
        if self.use_pruning and (hasattr(self, "static_mask_logit") and self.static_mask_logit is not None and self.static_mask_logit.numel() > 0 and hasattr(self, "static_xyz") and self.static_xyz.shape[0] > 0):
            sm = torch.sigmoid(self.static_mask_logit)
            keep_static_mask = (sm > threshold).squeeze()
            if keep_static_mask.dim() == 0:
                keep_static_mask = keep_static_mask.unsqueeze(0)
            if not keep_static_mask.all():
                prune_static_mask = ~keep_static_mask
                self.prune_static_points(prune_static_mask)
        
        if self.use_pruning and hasattr(self, "static_mask_logit") and hasattr(self, "static_xyz"):
            new_static = self.static_xyz.shape[0]

            if self.static_mask_logit.numel() == 0 and new_static > 0:
                pad = torch.full((new_static, 1), 0.0, device="cuda")
                self.static_mask_logit = nn.Parameter(pad)

            elif self.static_mask_logit.shape[0] != new_static:
                diff = new_static - self.static_mask_logit.shape[0]
                if diff > 0:
                    pad = torch.full((diff, 1), 0.0, device="cuda", dtype=self.static_mask_logit.dtype)
                    self.static_mask_logit = nn.Parameter(torch.cat([self.static_mask_logit, pad], dim=0))
                else:
                    self.static_mask_logit = nn.Parameter(self.static_mask_logit[:new_static])
        
        if self.use_sh_adaptive and hasattr(self, "static_sh_mask_logit") and hasattr(self, "static_xyz"):
            new_static = self.static_xyz.shape[0]

            if self.static_sh_mask_logit.numel() == 0 and new_static > 0:
                pad = torch.full((new_static, 3), 0.0, device="cuda")
                self.static_sh_mask_logit = nn.Parameter(pad)

            elif self.static_sh_mask_logit.shape[0] != new_static:
                diff = new_static - self.static_sh_mask_logit.shape[0]
                if diff > 0:
                    pad = torch.full((diff, 3), 0.0, device="cuda", dtype=self.static_sh_mask_logit.dtype)
                    self.static_sh_mask_logit = nn.Parameter(torch.cat([self.static_sh_mask_logit, pad], dim=0))
                else:
                    self.static_sh_mask_logit = nn.Parameter(self.static_sh_mask_logit[:new_static])

    
    def prune_low_dynamic_mask_points(self, threshold=0.1):
        """
        对动态点进行硬剪枝（只剪动态 4D 池）
        """
        if self.use_pruning and (hasattr(self, "dynamic_mask_logit") and self.dynamic_mask_logit is not None and self.dynamic_mask_logit.numel() > 0 and hasattr(self, "_xyz") and self._xyz.shape[0] > 0):
            dm = torch.sigmoid(self.dynamic_mask_logit)
            keep_dynamic_mask = (dm > threshold).squeeze()
            if keep_dynamic_mask.dim() == 0:
                keep_dynamic_mask = keep_dynamic_mask.unsqueeze(0)

            if not keep_dynamic_mask.all():
                prune_dynamic_mask = ~keep_dynamic_mask
                self.prune_points(prune_dynamic_mask)

        if self.use_pruning and hasattr(self, "dynamic_mask_logit") and hasattr(self, "_xyz"):
            new_dynamic = self._xyz.shape[0]

            if self.dynamic_mask_logit.numel() == 0 and new_dynamic > 0:
                pad = torch.full((new_dynamic, 1), 0.0, device="cuda")
                self.dynamic_mask_logit = nn.Parameter(pad)

            elif self.dynamic_mask_logit.shape[0] != new_dynamic:
                diff = new_dynamic - self.dynamic_mask_logit.shape[0]
                if diff > 0:
                    pad = torch.full((diff, 1), 0.0, device="cuda", dtype=self.dynamic_mask_logit.dtype)
                    self.dynamic_mask_logit = nn.Parameter(torch.cat([self.dynamic_mask_logit, pad], dim=0))
                else:
                    self.dynamic_mask_logit = nn.Parameter(self.dynamic_mask_logit[:new_dynamic])
        
        if self.use_sh_adaptive and hasattr(self, "dynamic_sh_mask_logit") and hasattr(self, "_xyz"):
            new_dynamic = self._xyz.shape[0]

            if self.dynamic_sh_mask_logit.numel() == 0 and new_dynamic > 0:
                pad = torch.full((new_dynamic, 3), 0.0, device="cuda")
                self.dynamic_sh_mask_logit = nn.Parameter(pad)

            elif self.dynamic_sh_mask_logit.shape[0] != new_dynamic:
                diff = new_dynamic - self.dynamic_sh_mask_logit.shape[0]
                if diff > 0:
                    pad = torch.full((diff, 3), 0.0, device="cuda", dtype=self.dynamic_sh_mask_logit.dtype)
                    self.dynamic_sh_mask_logit = nn.Parameter(torch.cat([self.dynamic_sh_mask_logit, pad], dim=0))
                else:
                    self.dynamic_sh_mask_logit = nn.Parameter(self.dynamic_sh_mask_logit[:new_dynamic])

    @torch.no_grad()
    def hard_prune_sh(self, threshold=0.1):
        """
        Independent hard prune for SH coefficients.
        Keep tensor shape unchanged.
        Only zero corresponding SH bands independently.
        """

        def prune_branch(features_rest, sh_mask_logit, branch_name="dynamic"):
            if features_rest is None or features_rest.numel() == 0:
                return features_rest

            if sh_mask_logit is None or sh_mask_logit.numel() == 0:
                print(f"[{branch_name}] No SH mask found, skip.")
                return features_rest

            soft_sh = torch.sigmoid(sh_mask_logit)

            # independent gating
            m1 = (soft_sh[:, 0] > threshold).to(features_rest.dtype)[:, None, None]  # l=1
            m2 = (soft_sh[:, 1] > threshold).to(features_rest.dtype)[:, None, None]  # l=2
            m3 = (soft_sh[:, 2] > threshold).to(features_rest.dtype)[:, None, None]  # l=3

            spatial_rest = features_rest[:, :15, :].clone()

            spatial_rest[:, 0:3,  :] *= m1   # l=1
            spatial_rest[:, 3:8,  :] *= m2   # l=2
            spatial_rest[:, 8:15, :] *= m3   # l=3

            if features_rest.shape[1] > 15:
                tail = features_rest[:, 15:, :].clone()
                features_rest = torch.cat([spatial_rest, tail], dim=1)
            else:
                features_rest = spatial_rest
            return features_rest

        if hasattr(self, "_features_rest"):
            self._features_rest = prune_branch(
                self._features_rest,
                getattr(self, "dynamic_sh_mask_logit", None),
                "dynamic"
            )

        if hasattr(self, "static_features_rest"):
            self.static_features_rest = prune_branch(
                self.static_features_rest,
                getattr(self, "static_sh_mask_logit", None),
                "static"
            )

    @torch.no_grad()
    def compute_model_size(
        self,
        include_dynamic=True,
        include_static=True,
        include_temporal=True,
        include_masks=False,
        include_sh_mask_logits=False,
        include_env_map=False,
        include_training_stats=False,
        count_mode="dense",              # "dense" or "effective"
        phi_threshold_dynamic=None,      # e.g. 0.1
        phi_threshold_static=None,       # e.g. 0.1
        sh_threshold=None,               # e.g. 0.1
        override_dtype_bytes=None,       # None -> use tensor.element_size(); 2 -> fp16; 4 -> fp32
        verbose=False,
    ):
        """
        统计模型大小（默认不含 optimizer state）。

        两种模式：
        1) dense:
        - 按当前 tensor shape 直接统计
        - 适合原始 checkpoint / 原始 dense 存储大小

        2) effective:
        - 可结合 phi_threshold_* 和 sh_threshold，只统计“有效”参数
        - 适合做 RDO / ablation 的 rate 估计
        - 注意：SH hard prune 只置零不改 shape，因此 dense 模式下 SH 不会变小，effective 模式才会体现 SH 压缩收益
        """
        assert count_mode in ["dense", "effective"]

        def _is_valid_tensor(x):
            return torch.is_tensor(x) and x is not None and x.numel() > 0

        def _elem_bytes(x):
            if override_dtype_bytes is not None:
                return int(override_dtype_bytes)
            return int(x.element_size())

        def _tensor_bytes(x, row_mask=None):
            if not _is_valid_tensor(x):
                return 0
            t = x.detach()
            if row_mask is not None:
                if t.shape[0] != row_mask.shape[0]:
                    raise ValueError(
                        f"Row mask shape mismatch: tensor first dim = {t.shape[0]}, mask = {row_mask.shape[0]}"
                    )
                t = t[row_mask]
            return int(t.numel()) * _elem_bytes(t)

        def _make_keep_mask(mask_logit, num_points, threshold, device):
            keep = torch.ones(num_points, dtype=torch.bool, device=device)
            if threshold is None:
                return keep
            if _is_valid_tensor(mask_logit) and mask_logit.shape[0] == num_points:
                soft = torch.sigmoid(mask_logit.detach()).view(-1)
                keep = soft > threshold
            return keep

        def _features_rest_bytes(features_rest, sh_mask_logit=None, point_keep_mask=None, sh_thr=None):
            """
            对 features_rest 做两种统计：
            - dense: 直接按 tensor shape 计
            - effective:
                * 若不给 sh_thr：仍按保留下来的点的 dense shape 计
                * 若给 sh_thr：只统计有效 SH bands（l1/l2/l3）+ tail
            """
            if not _is_valid_tensor(features_rest):
                return 0

            t = features_rest.detach()
            if point_keep_mask is not None:
                t = t[point_keep_mask]

            if count_mode == "dense" or sh_thr is None:
                return int(t.numel()) * _elem_bytes(t)

            # effective 模式下，按 SH band 实际有效通道数计算
            # 约定前 15 个通道对应 spatial SH 的 l1/l2/l3: 3 / 5 / 7
            # 这和你 hard_prune_sh() / apply_sh_masks() 的写法一致
            if t.numel() == 0:
                return 0

            n_pts = t.shape[0]
            n_ch = t.shape[1]
            n_rgb = t.shape[2]
            bytes_per_elem = _elem_bytes(t)

            if not _is_valid_tensor(sh_mask_logit):
                return int(t.numel()) * bytes_per_elem

            sh = sh_mask_logit.detach()
            if point_keep_mask is not None:
                sh = sh[point_keep_mask]

            if sh.shape[0] != n_pts:
                return int(t.numel()) * bytes_per_elem

            # 前 15 个 spatial channels
            spatial_ch = min(n_ch, 15)
            tail_ch = max(n_ch - 15, 0)

            # m1/m2/m3: [N]
            m1 = (torch.sigmoid(sh[:, 0]) > sh_thr)
            m2 = (torch.sigmoid(sh[:, 1]) > sh_thr)
            m3 = (torch.sigmoid(sh[:, 2]) > sh_thr)

            # l1/l2/l3 分别占 3/5/7 个通道
            kept_spatial_ch = 0
            if spatial_ch > 0:
                # 只在前 15 通道范围内计数
                ch_l1 = min(max(spatial_ch - 0, 0), 3)
                ch_l2 = min(max(spatial_ch - 3, 0), 5)
                ch_l3 = min(max(spatial_ch - 8, 0), 7)

                kept_spatial_ch += int(m1.sum().item()) * ch_l1
                kept_spatial_ch += int(m2.sum().item()) * ch_l2
                kept_spatial_ch += int(m3.sum().item()) * ch_l3

            kept_tail_ch = tail_ch * n_pts
            kept_total_values = (kept_spatial_ch + kept_tail_ch) * n_rgb
            return int(kept_total_values) * bytes_per_elem

        breakdown = {}
        total_bytes = 0

        # ---------------- Dynamic keep mask ----------------
        dyn_keep = None
        if include_dynamic and _is_valid_tensor(self._xyz):
            dyn_keep = _make_keep_mask(
                getattr(self, "dynamic_mask_logit", None),
                self._xyz.shape[0],
                phi_threshold_dynamic if count_mode == "effective" else None,
                self._xyz.device,
            )

        # ---------------- Static keep mask ----------------
        sta_keep = None
        if include_static and _is_valid_tensor(self.static_xyz):
            sta_keep = _make_keep_mask(
                getattr(self, "static_mask_logit", None),
                self.static_xyz.shape[0],
                phi_threshold_static if count_mode == "effective" else None,
                self.static_xyz.device,
            )

        # ---------------- Dynamic branch ----------------
        if include_dynamic:
            breakdown["dynamic_xyz"] = _tensor_bytes(getattr(self, "_xyz", None), dyn_keep)
            breakdown["dynamic_features_dc"] = _tensor_bytes(getattr(self, "_features_dc", None), dyn_keep)
            breakdown["dynamic_features_rest"] = _features_rest_bytes(
                getattr(self, "_features_rest", None),
                getattr(self, "dynamic_sh_mask_logit", None),
                dyn_keep,
                sh_threshold if count_mode == "effective" else None,
            )
            breakdown["dynamic_opacity"] = _tensor_bytes(getattr(self, "_opacity", None), dyn_keep)
            breakdown["dynamic_scaling"] = _tensor_bytes(getattr(self, "_scaling", None), dyn_keep)
            breakdown["dynamic_rotation"] = _tensor_bytes(getattr(self, "_rotation", None), dyn_keep)

            if include_temporal and self.gaussian_dim == 4:
                breakdown["dynamic_t"] = _tensor_bytes(getattr(self, "_t", None), dyn_keep)
                breakdown["dynamic_scaling_t"] = _tensor_bytes(getattr(self, "_scaling_t", None), dyn_keep)
                breakdown["dynamic_rotation_r"] = _tensor_bytes(
                    getattr(self, "_rotation_r", None), dyn_keep if self.rot_4d else None
                )

        # ---------------- Static branch ----------------
        if include_static:
            breakdown["static_xyz"] = _tensor_bytes(getattr(self, "static_xyz", None), sta_keep)
            breakdown["static_features_dc"] = _tensor_bytes(getattr(self, "static_features_dc", None), sta_keep)
            breakdown["static_features_rest"] = _features_rest_bytes(
                getattr(self, "static_features_rest", None),
                getattr(self, "static_sh_mask_logit", None),
                sta_keep,
                sh_threshold if count_mode == "effective" else None,
            )
            breakdown["static_opacity"] = _tensor_bytes(getattr(self, "static_opacity", None), sta_keep)
            breakdown["static_scaling"] = _tensor_bytes(getattr(self, "static_scaling", None), sta_keep)
            breakdown["static_rotation"] = _tensor_bytes(getattr(self, "static_rotation", None), sta_keep)

        # ---------------- Mask logits ----------------
        if include_masks:
            breakdown["dynamic_mask_logit"] = _tensor_bytes(getattr(self, "dynamic_mask_logit", None))
            breakdown["static_mask_logit"] = _tensor_bytes(getattr(self, "static_mask_logit", None))

        if include_sh_mask_logits:
            breakdown["dynamic_sh_mask_logit"] = _tensor_bytes(getattr(self, "dynamic_sh_mask_logit", None))
            breakdown["static_sh_mask_logit"] = _tensor_bytes(getattr(self, "static_sh_mask_logit", None))

        # ---------------- Env map ----------------
        if include_env_map:
            breakdown["env_map"] = _tensor_bytes(getattr(self, "env_map", None))

        # ---------------- Training-only statistics ----------------
        if include_training_stats:
            breakdown["xyz_gradient_accum"] = _tensor_bytes(getattr(self, "xyz_gradient_accum", None))
            breakdown["denom"] = _tensor_bytes(getattr(self, "denom", None))
            breakdown["max_radii2D"] = _tensor_bytes(getattr(self, "max_radii2D", None))
            breakdown["t_gradient_accum"] = _tensor_bytes(getattr(self, "t_gradient_accum", None))
            breakdown["dynamic_vis_denom"] = _tensor_bytes(getattr(self, "dynamic_vis_denom", None))
            breakdown["dynamic_alpha_accum"] = _tensor_bytes(getattr(self, "dynamic_alpha_accum", None))
            breakdown["dynamic_motion_accum"] = _tensor_bytes(getattr(self, "dynamic_motion_accum", None))
            breakdown["dynamic_time_support_accum"] = _tensor_bytes(getattr(self, "dynamic_time_support_accum", None))
            breakdown["static_max_radii2D"] = _tensor_bytes(getattr(self, "static_max_radii2D", None))
            breakdown["static_xyz_gradient_accum"] = _tensor_bytes(getattr(self, "static_xyz_gradient_accum", None))
            breakdown["static_denom"] = _tensor_bytes(getattr(self, "static_denom", None))

        total_bytes = sum(breakdown.values())
        total_mb = total_bytes / 1024.0 / 1024.0

        result = {
            "total_bytes": int(total_bytes),
            "total_mb": float(total_mb),
            "breakdown_bytes": breakdown,
            "breakdown_mb": {k: v / 1024.0 / 1024.0 for k, v in breakdown.items()},
            "num_dynamic_points": int(self._xyz.shape[0]) if _is_valid_tensor(getattr(self, "_xyz", None)) else 0,
            "num_static_points": int(self.static_xyz.shape[0]) if _is_valid_tensor(getattr(self, "static_xyz", None)) else 0,
            "num_dynamic_kept": int(dyn_keep.sum().item()) if dyn_keep is not None else 0,
            "num_static_kept": int(sta_keep.sum().item()) if sta_keep is not None else 0,
            "count_mode": count_mode,
        }

        if verbose:
            print("\n================ Model Size ================")
            print(f"count_mode      : {count_mode}")
            print(f"total_bytes     : {result['total_bytes']}")
            print(f"total_mb        : {result['total_mb']:.4f} MB")
            print(f"dynamic kept    : {result['num_dynamic_kept']} / {result['num_dynamic_points']}")
            print(f"static kept     : {result['num_static_kept']} / {result['num_static_points']}")
            for k, v in result["breakdown_mb"].items():
                if v > 0:
                    print(f"{k:24s}: {v:.4f} MB")
            print("===========================================\n")

        return result