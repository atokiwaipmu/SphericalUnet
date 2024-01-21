
import os
import numpy as np
from glob import glob

def set_data_params(params=None, 
                    n_maps=None, 
                    order=2, 
                    transform_type="sigmoid"):
    if params is None:
        params = {}
    if "data" not in params.keys():
        params["data"] = {}
    params["data"]["HR_dir"]: str = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/data/nc256_smoothed/"
    params["data"]["LR_dir"]: str = "//gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/data/nc128_smoothed/"
    params["data"]["n_maps"]: int = n_maps if n_maps is not None else len(glob(params["data"]["LR_dir"] + "*.fits"))
    params["data"]["nside"]: int = 512
    params["data"]["order"]: int = order
    params["data"]["transform_type"]: str = transform_type
    params["data"]["upsample_scale"]: float = 2.0
    return params

def set_architecture_params(params=None, 
                            model="unet", 
                            norm_type="batch",
                            act_type="mish",
                            use_conv=False,
                            num_resblocks=[2, 2, 2, 2]):
    if params is None:
        params = {}
    if "architecture" not in params.keys():
        params["architecture"] = {}
    params["architecture"]["model"]: str = model
    params["architecture"]["mults"] = [1, 2, 4, 8]
    params["architecture"]["skip_factor"]: float = 1/np.sqrt(2)
    params["architecture"]["kernel_size"]: int = 20 
    params["architecture"]["dim_in"]: int = 1 
    params["architecture"]["dim_out"]: int = 1
    params["architecture"]["inner_dim"]: int = 64
    params["architecture"]["norm_type"]: str = norm_type
    params["architecture"]["act_type"]: str = act_type
    params["architecture"]["use_conv"]: bool = use_conv
    params["architecture"]["num_blocks"]: list = num_resblocks
    return params

def set_train_params(params=None, base_dir=None, target="HR", batch_size=4):
    if params is None:
        params = {}
    if "train" not in params.keys():
        params["train"] = {}
    params["train"]['target']: str = target
    params["train"]['train_rate']: float = 0.825
    params["train"]['batch_size']: int = batch_size
    params["train"]['learning_rate'] = 10**-4
    params["train"]['n_epochs']: int = 100
    params["train"]['gamma']: float = 0.9999
    params["train"]['save_dir']: str = f"{base_dir}/ckpt_logs/{params['architecture']['model']}/"
    os.makedirs(params["train"]['save_dir'], exist_ok=True)
    params["train"]['log_name']: str = f"{params['train']['target']}_{params['data']['transform_type']}_{params['data']['order']}"
    params["train"]['patience']: int = 10
    params["train"]['save_top_k']: int = 3
    params["train"]['early_stop']: bool = True
    return params

def set_params(
        base_dir="/gpfs02/work/akira.tokiwa/gpgpu/Github/SphericalUnet",
        n_maps=None,
        order=2,
        transform_type="smoothed",
        model="unet",
        norm_type="batch",
        act_type="mish",
        target="HR", 
        batch_size=4
        ):
    params = {}
    params = set_data_params(params, n_maps=n_maps, order=order, transform_type=transform_type)
    params = set_architecture_params(params, model=model, norm_type=norm_type, act_type=act_type)
    params = set_train_params(params, base_dir=base_dir, target=target, batch_size=batch_size)
    return params