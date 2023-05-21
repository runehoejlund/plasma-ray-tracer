'''
This file contains a function, `grad`, which allow us to
differentiate any function defined on a grid through
finite differences. Note, the `grad` function allows for arbitrary
dimensions in the input and output, i.e. f : R^n -> R^m.
'''

import numpy as np

def grad(f, *args, axes=None):
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
    
    def dtake(a, index, axis):
        '''numpy take a single index but keep dimensions.'''
        dims = [1 if i == axis else s for i, s in enumerate(a.shape)]
        return np.take(a=a, indices=index, axis=axis).reshape(dims)
    
    def take_many(f, indeces, axis):
        return (*[dtake(f, i, axis) for i in indeces], )
    
    grad_f = np.zeros((*_f.shape, len(args)))
    if axes is None:
        axes = np.arange(len(args))
    
    for i, (axis, xi) in enumerate(zip(axes, args)):
        # make padding for calculating three-point difference
        # estimate for the beginning and end points.
        f0, f1, f2, fm1, fm2, fm3 = take_many(_f, [0, 1, 2, -1, -2, -3], axis)
        f_pad = np.concatenate([3*(f0 - f1) + f2, _f, 3*(fm1 - fm2) + fm3], axis=axis)
        
        # The derivative is then calculated
        # as the central difference for all points.
        df_pad = np.diff(f_pad, axis=axis)
        h = np.diff(xi[:2])[0]
        grad_f_i = (np.take(df_pad, np.arange(1, df_pad.shape[axis]), axis=axis)
         + np.take(df_pad, np.arange(0, df_pad.shape[axis] - 1), axis=axis))/(2*h)
        grad_f[..., i] = grad_f_i

    return grad_f.squeeze()