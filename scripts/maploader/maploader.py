import os
from glob import glob
import numpy as np
import torch
import healpy as hp
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Normalize

# Constants
PIXEL_AREA_MULTIPLIER = 12  # The pixel area is defined by 12*order^2


class Transforms():
    def __init__(self, transform_type='log2linear', range_min=None, range_max=None):
        self.transform_type = transform_type
        self.transform, self.inverse_transform = None, None
        self.range_min, self.range_max = range_min, range_max
        self.set_transforms()

    def set_transforms(self):
        if self.transform_type == 'minmax':
            self.transform = Compose([lambda t: (t - self.range_min) / (self.range_max - self.range_min) * 2 - 1])
            self.inverse_transform = Compose([lambda t: (t + 1) / 2 * (self.range_max - self.range_min) + self.range_min])
        elif self.transform_type == 'sigmoid':
            self.transform = Compose([lambda t: (torch.sigmoid(t)-0.5)*2]) # [-inf, inf] -> [0, 1] -> [-1, 1]
            self.inverse_transform = Compose([lambda t: torch.logit((t/2+0.5))])
        elif self.transform_type == 'both':
            self.transform = Compose([lambda t: torch.sigmoid(t), lambda t: (t - self.range_min) / (self.range_max - self.range_min) * 2 - 1])
            self.inverse_transform = Compose([lambda t: (t + 1) / 2 * (self.range_max - self.range_min) + self.range_min, lambda t: torch.logit((t))])
        elif self.transform_type == 'log2linear':
            self.transform = Compose([lambda t: 10**t - 1])
            self.inverse_transform = Compose([lambda t: torch.log10(t + 1)])
        elif self.transform_type == 'smoothed':
            # minus min, linear to log, then sigmoid
            self.transform = Compose([lambda t: (t - self.range_min), lambda t: torch.log10(t + 1), lambda t: (torch.sigmoid(t)-0.5)*2])
            self.inverse_transform = Compose([lambda t: torch.logit((t/2+0.5)), lambda t: 10**t - 1, lambda t: (t + self.range_min)])
        else: # no transform
            self.transform = Compose([lambda t:t])
            self.inverse_transform = Compose([lambda t:t])

def hp_split(img, order, nest=True):
    """
    Function to split the image into multiple images based on the given order.
    """
    npix = len(img)
    nsample = PIXEL_AREA_MULTIPLIER * order**2
    
    if npix < nsample:
        raise ValueError('Order not compatible with data.')
    
    if not nest:
        raise NotImplementedError('Implement the change of coordinate.')
    
    return img.reshape([nsample, npix // nsample])

class MapDataset(data.Dataset):
    """
    Class for the map dataset.
    """
    def __init__(self, mapdir, n_maps, nside, order=2):
        self.nside = nside
        self.n_maps = n_maps
        self.order = order
        self.npix = hp.nside2npix(self.nside)
        self.data_shape = (self.n_maps, self.npix, 1)
        self.maps = sorted(glob(f'{mapdir}*.fits'))[:self.n_maps]
        self.patch_flag = False

    def get_numpymap(self):
        map_stacked = np.vstack([hp.read_map(dmap) for dmap in self.maps])
        return map_stacked
    
    def maps2patches(self, map_stacked):
        map_patches = np.vstack([hp_split(el, order=self.order) for el in map_stacked])
        self.patch_flag = True
        return map_patches

    def get_tensormap(self, map_stacked):
        if self.patch_flag:
            self.data_shape = (self.n_maps*self.order**2*PIXEL_AREA_MULTIPLIER, self.npix//(self.order**2*PIXEL_AREA_MULTIPLIER), 1)
        tensor_map = ToTensor()(map_stacked).view(*self.data_shape).float()
        return tensor_map    
    
def get_data(map_dir, n_map, nside, order=2, issplit=False, stdout=True):
    dataset = MapDataset(map_dir, n_map, nside, order)
    data_loaded_np = dataset.get_numpymap()
    if issplit:
        data_loaded_np = dataset.maps2patches(data_loaded_np)
    data_loaded = dataset.get_tensormap(data_loaded_np)
    if stdout:
        print("data loaded from {}.  Number of maps: {}".format(map_dir, n_map))
        print("data nside: {}, divided into {} patches, each patch has {} pixels.".format(nside, 12 * order**2, hp.nside2npix(nside)//(12 * order**2)))
    return data_loaded

def get_normalized_data(data_loaded, transform_type='minmax', stdout=True):
    range_min, range_max = data_loaded.min().clone().detach(), data_loaded.max().clone().detach()
    transforms = Transforms(transform_type, range_min, range_max)
    data_normalized = transforms.transform(data_loaded)
    if stdout:
        print("data normalized to [{},{}] by {} transform.".format(data_normalized.min(), data_normalized.max(), transform_type))
    return data_normalized, transforms

def get_loaders(data_input, data_condition, rate_train, batch_size, stdout=True):
    """
    Function to get the loaders for training and validation datasets.
    """
    combined_dataset = data.TensorDataset(data_input, data_condition)
    len_train = int(rate_train * len(data_input))
    len_val = len(data_input) - len_train
    train, val = data.random_split(combined_dataset, [len_train, len_val])
    loaders = {x: data.DataLoader(ds, batch_size=batch_size, shuffle=x=='train', num_workers=os.cpu_count()) for x, ds in zip(('train', 'val'), (train, val))}
    if stdout:
        print("train:validation = {}:{}, batch_size: {}".format(len(loaders['train']), len(loaders['val']), batch_size))
    
    return loaders['train'], loaders['val']

def get_data_from_params(params, stdout=True):
    lr = get_data(params["data"]["LR_dir"], params["data"]["n_maps"], params["data"]["nside"], params["data"]["order"], issplit=True, stdout=stdout)
    hr = get_data(params["data"]["HR_dir"], params["data"]["n_maps"], params["data"]["nside"], params["data"]["order"], issplit=True, stdout=stdout)
    return lr, hr

def get_normalized_from_params(lr, hr, params, stdout=True):
    lr, transforms_lr = get_normalized_data(lr, transform_type=params["data"]["transform_type"], stdout=stdout)
    hr, transforms_hr = get_normalized_data(hr, transform_type=params["data"]["transform_type"], stdout=stdout)

    if params["train"]['target'] == 'difference':
        diff = hr - lr
        if stdout:
            print("Difference data calculated from HR - LR. min: {}, max: {}".format(diff.min(), diff.max()))
        data_input, data_condition = diff, lr
        return data_input, data_condition, transforms_lr, transforms_hr
    elif params["train"]['target'] == 'HR':
        data_input, data_condition = hr, lr
        return data_input, data_condition, transforms_lr, transforms_hr
    else:
        raise ValueError("target must be 'difference' or 'HR'")

def get_loaders_from_params(params, stdout=True):
    lr, hr = get_data_from_params(params, stdout=stdout)
    data_input, data_condition, _, _ = get_normalized_from_params(lr, hr, params, stdout=stdout)
    train_loader, val_loader = get_loaders(data_input, data_condition, params["train"]['train_rate'], params["train"]['batch_size'], stdout=stdout)
    return train_loader, val_loader

def get_condition_from_params(params, stdout=True):
    lr = get_data(params["data"]["LR_dir"], params["data"]["n_maps"], params["data"]["nside"], params["data"]["order"], issplit=True, stdout=stdout)   
    data_condition, transforms_lr= get_normalized_data(lr, transform_type=params["data"]["transform_type"], stdout=stdout)
    return lr, data_condition, transforms_lr
