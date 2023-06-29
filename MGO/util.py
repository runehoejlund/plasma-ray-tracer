''' Utility Functions
'''
import numpy as np
from skimage.segmentation import flood

def inner_product(a, b):
    return np.einsum('...i,...i->...', a, b)

def transpose(A):
    '''return transpose of only the last two dimensions.'''
    return np.moveaxis(A, -1, -2)

def eye(shape):
    eye = np.zeros(shape)
    eye[..., :, :] = np.eye(shape[-1])
    return eye

def diag(v):
    _eye = eye((*v.shape, v.shape[-1]))
    return _eye * np.stack([v]*v.shape[-1], axis=-1)

def sgn_mask_from_seed(a, seed):
    '''returns boolean array mask for the "sign branch" connected to the seed index.
    A "sign branch" is here defined as a connected region where the sign of the array `a` is constant.
    '''
    return flood(np.signbit(a), seed)

def get_masks_of_const_sgn(a, ND = 3):
    '''returns list of boolean arrays with masks which
    subdivide the values of the array, `a`,
    into regions where the sign is constant.'''
    seed = (0, ) * ND
    regions_remaining = True
    masks = []
    while regions_remaining:
        masks.append(sgn_mask_from_seed(a, seed))
        remaining_cells = np.argwhere(np.any(masks, axis=0) == False)
        if len(remaining_cells) > 0:
            seed = tuple(remaining_cells[0])
        else:
            regions_remaining = False
    return masks

# def continuous_angle(z, axis=0):
#     z0 = np.take(z, 0, axis=axis)
#     arg0 = np.angle(z0)
#     return arg0 + np.cumsum(np.imag(np.diff(z, prepend=z0)/z))

# def continuous_sqrt(z, axis=0):
#     return np.sqrt(np.abs(z)) * np.exp(1j*continuous_angle(z)/2)

def continuous_angle_of_reals(x, axis=0):
    x0 = np.take(x, 0, axis=axis)
    sgn = np.sign(x)
    return np.angle(x) + 2*np.pi*np.cumsum(np.heaviside(np.diff(sgn, prepend=sgn[0]), 0))

def continuous_sqrt_of_reals(x, axis=0):
    return np.sqrt(np.abs(x)) * np.exp(1j*continuous_angle_of_reals(x)/2)