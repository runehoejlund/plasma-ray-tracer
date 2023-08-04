'''
This file contains a function, `grad`, which allow us to
differentiate any function defined on a grid through
finite differences. Note, the `grad` function allows for arbitrary
dimensions in the input and output, i.e. f : R^n -> R^m.
'''
import numpy as np
import util as ut

def _dtake(a, index, axis):
    '''numpy take a single index but keep dimensions.'''
    dims = [1 if i == axis else s for i, s in enumerate(a.shape)]
    return np.take(a=a, indices=index, axis=axis).reshape(dims)
    
def _take_many(f, indeces, axis):
    return (*[_dtake(f, i, axis) for i in indeces], )

def pad_left(f, axis):
    ''' makes left padding for calculating three-point
    difference estimate for the beginning point.
    '''
    f0, f1, f2 = _take_many(f, [0, 1, 2], axis)
    f_pad = np.concatenate([3*(f0 - f1) + f2, f], axis=axis)
    return f_pad

def pad_right(f, axis):
    ''' makes right padding for calculating three-point
    difference estimate for the end point.
    '''
    fm1, fm2, fm3 = _take_many(f, [-1, -2, -3], axis)
    f_pad = np.concatenate([f, 3*(fm1 - fm2) + fm3], axis=axis)
    return f_pad

def grad(f, *args, axes=None, cropped_axes=[]):
    '''
    Returns finite difference derivative of N-D array, f, with respect to arguments, *args.

    **Example:**
    Say we have a function, f : R^2 -> R^2 of
    two variables returning 2 values.
    I.e. f(x1, x2) = (f1(x1, x2), f2(x1, x2)).
    The function is defined on a grid of two
    input arrays with shapes x1 ~ (n1, ), x2 ~ (n2, )
    and therefore f has shape f ~ (n1, n2, 2).
    (The grid can be made through `X1, X2 = np.meshgrid(x1, x2, indexing='ij')` )
    
    Using `grad(f, x1, x2)` returns an array of shape ~ (n1, n2, 2, 2)
    containing the estimated partial derivatives
    of f1 and f2 with respect to x1 and x2.
    Thus, `grad(f, x1, x2)[:, :, 0, 0]` corresponds to ∂f_1/∂x_1
    and `grad(f, x1, x2)[:, :, 0, 1]` corresponds to ∂f_1/∂x_2.

    You may also use `grad(f, x2, axes=[1])`
    to only get ∂f_1/∂x_2 and ∂f_2/∂x_2 ... and so on.
    '''
    _f = f.squeeze()
    if axes is None:
        axes = np.arange(len(args))
    
    assert set(cropped_axes) <= set(axes), 'Error: Axes to crop must be contained in axes to differentiate along. Received `cropped_axes` = ' + str(cropped_axes) + ', but `axes` = ' + str(axes)
    shape = (s - 2 if i in cropped_axes else s for i, s in enumerate(_f.shape))
    grad_f = np.zeros((*shape, len(args)), dtype=_f.dtype)
    
    for i, (axis, xi) in enumerate(zip(axes, args)):
        # Handle edge cases
        if len(xi) == 1:
            # If we only have 1 point, assume function is constant.
            grad_f[..., i] = 0
            continue
        
        h = np.diff(xi[:2])[0]
        if h == 0:
            # If h is close to 0, set gradient to 0
            grad_f[..., i] = 0
            continue
        
        if len(xi) == 2:
            # If we only have 2 points use two-point difference
            df = np.diff(f, axis=axis)
            grad_f[..., i] = df/h
            continue
        
        # Main case
        if axis not in cropped_axes:
            # make padding for calculating three-point difference
            # estimate for the beginning and end points.
            f_pad = pad_right(pad_left(f, axis=axis), axis=axis)
        else:
            f_pad = f
        slice_pad = np.s_[[slice(1, -1) if (j != axis and j in cropped_axes) else slice(None) for j in range(len(f.shape))]]
        f_pad = f_pad[(*slice_pad, )]
        
        # The derivative is then calculated
        # as the central difference for all points.
        df_pad = np.diff(f_pad, axis=axis)
        grad_f_i = (np.take(df_pad, np.arange(1, df_pad.shape[axis]), axis=axis)
         + np.take(df_pad, np.arange(0, df_pad.shape[axis] - 1), axis=axis))/(2*h)
        grad_f[..., i] = grad_f_i

    return grad_f.squeeze()

def local_grad(f, index, *args, axes=None, order=1):
    '''Determine gradient of a given order evaluated at the given index.'''
    _f = f.squeeze()
    if type(index) == int:
        index = [index]
    if axes is None:
        axes = np.arange(len(args))
    
    order_grad_f = np.zeros(len(args), dtype=_f.dtype)
    for i, (axis, xi) in enumerate(zip(axes, args)):
        _order = order
        _nbh = slice(None)
        _ind = index[i]
        _x = xi
        _out_i = f[_nbh]
        for j in range(order):
            _nbh, _ind = ut.neighbourhood(_ind % len(_x[_nbh]), len(_x[_nbh]), N_neighbours=_order - j)
            _out_i = grad(_out_i[_nbh], _x[_nbh], axes=[axis])
        
        order_grad_f[i] = _out_i[_ind]
    
    return order_grad_f