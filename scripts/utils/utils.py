
import numpy as np
import torch
from inspect import isfunction

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    shape = (batch_size, *((1,) * (len(x_shape) - 1)))
    return out.reshape(shape).to(t.device)