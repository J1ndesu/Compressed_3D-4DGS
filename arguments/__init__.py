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

"""Argument groups and defaults; main.py subsequently applies YAML overrides."""

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    """Register subclass defaults as CLI options and extract a matching namespace.

    A leading underscore adds a one-letter alias. Boolean options use
    store_true; set them to false in YAML when the configuration enables them.
    """
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.extension = ".png"
        self.num_extra_pts = 0
        self.loaded_pth = ""
        self.frame_ratio = 1
        self.dataloader = False
        self.from3dgs = ""
        self.start_timestamp = 0
        self.end_timestamp = -1
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.env_map_res = 0
        self.env_optimize_until = 1000000000
        self.env_optimize_from = 0
        self.eval_shfs_4d = False
        self.opa_threshold = 0.05
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    """Defaults for reconstruction, densification, and compression training."""
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_t_lr_init = -1.0
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.thresh_opa_prune = 0.005
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.densify_grad_t_threshold = 0.0002 / 40
        self.densify_until_num_points = -1
        self.final_prune_from_iter = -1
        self.sh_increase_interval = 1000
        self.lambda_opa_mask = 0.0
        self.lambda_rigid = 0.0
        self.lambda_motion = 0.0
        self.scale_t_threshold =3.0

        # Compression flags, penalties, and thresholds.
        self.use_pruning = False  # Enable learned Gaussian opacity gates.
        self.use_sh_adaptive = False  # Enable independent spatial SH band gates.
        self.use_vq = False  # Reserved flag; no vector-quantization pipeline is implemented.
        
        self.lambda_static_mask = 0.0002  # Static Gaussian sparsity weight.
        self.lambda_dynamic_mask = 0.0005  # Time-weighted dynamic Gaussian sparsity weight.
        self.lambda_sh = 0.0005  # SH band sparsity weight.
        self.phi_threshold = 0.1  # Strict sigmoid threshold for training-time Gaussian gates.
        self.phi_prune_dynamic = 0.1  # Hard-pruning threshold for dynamic points in validation.
        self.phi_prune_static = 0.1  # Hard-pruning threshold for static points in validation.
        self.phi_prune_sh = 0.1  # SH band threshold for rendering and validation hard pruning.
        
        self.gs_mask_start_iter = 3500
        self.gs_mask_warmup_iters_static = 1000
        self.gs_mask_warmup_iters_dynamic = 500

        self.sh_mask_start_iter = 4000
        self.sh_mask_warmup_iters = 1000
        
        # Reserved vector-quantization parameters.
        self.lambda_vqr = 0.0  # Reserved rotation VQ weight; keep zero.
        self.lambda_vqs = 0.0  # Reserved scale VQ weight; keep zero.
        self.lambda_vqc = 0.0  # Reserved color VQ weight; keep zero.

        self.static_mask_lr = 0.002  # Learning rate for static Gaussian mask logits.
        self.dynamic_mask_lr = 0.005  # Learning rate for dynamic Gaussian mask logits.
        self.sh_mask_lr = 0.05  # Learning rate for SH mask logits.
        self.codebook_lr = 0.0  # Reserved VQ codebook learning rate; keep zero.

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    """Merge saved cfg_args with non-None CLI values.

    This legacy helper is not the YAML merge used by main.py.
    """
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
