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

def neighbourhood(i, N, N_neighbours=1):
    '''returns slice corresponding to neighbourhood of i
        and index of i in sliced array.'''
    if i < 0 or i >= N:
        raise ValueError('index for neighbourhood is out of bounds')
    if i - N_neighbours < 0:
        return slice(min(1+2*N_neighbours, N)), i
    if i + N_neighbours >= N:
        return slice(max(0, N-(1+2*N_neighbours)), N), - (N-i)
    return slice(i - N_neighbours, i + N_neighbours + 1), N_neighbours

def continuous_angle(z, axis=0):
    if np.all(np.isclose(np.imag(z), 0)):
        return continuous_angle_of_reals(z, axis=axis)
    return np.unwrap(np.angle(z), discont=np.pi)

# def continuous_sqrt(z, axis=0):
#     return np.sqrt(np.abs(z)) * np.exp(1j*continuous_angle(z)/2)

def sgn_diff(x, axis=0):
    '''returns y: An array with sign of sign changes in x.
        y is +1 whenever the sign of x changes from -1 to +1
        and y is -1 whenever the sign of x changes from +1 to -1.
        At all other places, where there is no sign change, y is 0.'''
    sgn = np.sign(x)
    sgn0 = np.take(sgn, 0, axis=axis)[np.newaxis, ...]
    y = np.sign(np.diff(sgn, prepend=sgn0, axis=axis))
    return y

def continuous_angle_of_reals(x, axis=0):
    return np.angle(x) + 2*np.pi*np.cumsum(np.heaviside(sgn_diff(x, axis=axis), 0), axis=axis)

def continuous_sqrt_of_reals(x, axis=0):
    return np.sqrt(np.abs(x)) * np.exp(1j*continuous_angle_of_reals(x, axis=axis)/2)