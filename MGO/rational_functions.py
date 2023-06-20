import numpy as np
import itertools
from scipy.ndimage import convolve
from math import floor, ceil
from dataclasses import dataclass

def take_slice(a, slice_ind, axis):
    return a[(*(slice(None), )*axis, slice_ind, *(slice(None), )*(a.ndim - axis - 1))]

def _get_degrees(deg):
    '''return array of monomial powers with deg as highest degree(s)
    deg: int or tuple.
        If deg is an int, the function returns the 1D monomial powers: [0, ..., deg]
        If deg is a tuple (d1, .., dn) the function returns all set product
        combinations of [0, ..., d1] x [0, ..., d2] x ... x [0, ..., dn]
        This way, all combinations of (x1**i1), (x2**i2), ..., (xn**in) can be represented.
        '''
    if type(deg) == int:
        return np.arange(deg + 1)
    else:
        return np.array(list(itertools.product(*(range(d + 1) for d in deg))))

def _as_vector(x):
    _x = np.asarray(x)
    if len(_x.shape) == 1:
        _x = _x.reshape(-1, 1)
    return _x

def _vandermonde(x, degrees):
    _x = _as_vector(x)
    return np.stack(np.prod([_x**d for d in degrees], axis=-1), axis=-1)

def _trim_coef(coef):
    out = np.asarray(coef)
    if len(out) == 0:
        return out
    else:
        for axis, s in enumerate(out.shape):
            for i in range(s-1, -1, -1):
                if np.any(np.take(out, i, axis=axis) != 0):
                    break
            out = take_slice(out, slice(i+1), axis=axis)
        return out

def polyadd_ND(f_coef, g_coef):
    f_coef_pad = np.pad(f_coef, [(0, max(sg - sf, 0)) for (sf, sg) in zip(f_coef.shape, g_coef.shape)], mode='constant', constant_values=0)
    g_coef_pad = np.pad(g_coef, [(0, max(sf - sg, 0)) for (sf, sg) in zip(f_coef.shape, g_coef.shape)], mode='constant', constant_values=0)
    return NDPolynomial(f_coef_pad + g_coef_pad)

def polymul_ND(f_coef, g_coef):
    f_coef_pad = np.pad(f_coef, [(ceil((s-1)/2), floor((s-1)/2)) for s in g_coef.shape], mode='constant', constant_values=0)
    return NDPolynomial(convolve(f_coef_pad, g_coef, mode='constant', cval=0.0))

@dataclass
class NDPolynomial:
    '''Class for representing a polynomial function of multiple variables
    coef: numpy.ndarray of shape (d1, d2, ..., dn) where d· are the highest degrees and n is the number of variables the function f(x1, x2, ..., xn) takes.
        the coef matrix should be formatted such that
        f(x1, x2, ..., xn) = coef[i1, i2, ..., in] (x1 ** i1) * ... (x1 ** in)
        where there is implicit sumation over i1, ..., in (e.g. the first sum has i1 going from 0 to di1)
    '''
    coef: np.ndarray

    def __init__(self, coef):
        self.coef = _trim_coef(coef)

    def get_degrees(self):
        return _get_degrees(deg=np.array(self.coef.shape) - 1)

    def __call__(self, x):
        _x = _as_vector(x)
        if len(self.coef) > 0:
            degrees = self.get_degrees()
            return _vandermonde(_x, degrees) @ self.coef.flatten()
        else:
            return np.zeros_like(_x)
    
    def __add__(self, g):
        return polyadd_ND(self.coef, g.coef)
    
    def __sub__(self, g):
        return polyadd_ND(self.coef, -g.coef)
    
    def __mul__(self, g):
        return polymul_ND(self.coef, g.coef)

    def _diff(self, axis):
        axis_deg = self.coef.shape[axis] - 1
        deriv_coef = np.arange(1, axis_deg + 1) * take_slice(self.coef, slice(1, None), axis=axis)
        return NDPolynomial(deriv_coef)

    def deriv(self, axis, order=1):
        deriv = self
        for i in range(order):
            if len(deriv.coef) == 0:
                break
            deriv = deriv._diff(axis=axis)
        return deriv

# def polyadd_ND(f: NDPolynomial, g: NDPolynomial):
#     f_coef_pad = np.pad(f.coef, [(0, max(sg - sf, 0)) for (sf, sg) in zip(f.coef.shape, g.coef.shape)], mode='constant', constant_values=0)
#     g_coef_pad = np.pad(g.coef, [(0, max(sf - sg, 0)) for (sf, sg) in zip(f.coef.shape, g.coef.shape)], mode='constant', constant_values=0)
#     return NDPolynomial(f_coef_pad + g_coef_pad)

# def polymul_ND(f: NDPolynomial, g: NDPolynomial):
#     f_coef_pad = np.pad(f.coef, [(ceil((s-1)/2), floor((s-1)/2)) for s in g.coef.shape], mode='constant', constant_values=0)
#     return NDPolynomial(convolve(f_coef_pad, g.coef, mode='constant', cval=0.0))

@dataclass
class RationalFunction:
    '''Class for representing a rational function, i.e. a function which is the ratio of two polynomials:
    f(x) = P_L(x)/P_M(x)
    coef_L: np.ndarray, nominator coefficients
    coef_M: np.ndarray, denominator coefficients
    '''
    coef_L: np.ndarray
    coef_M: np.ndarray

    def __call__(self, x):
        return NDPolynomial(self.coef_L)(x) / NDPolynomial(self.coef_M)(x)
    
    def _diff(self, axis):
        P_L = NDPolynomial(self.coef_L)
        P_M = NDPolynomial(self.coef_M)
        deriv_coef_L = (P_L._diff(axis) * P_M - P_L * P_M._diff(axis)).coef
        deriv_coef_M = (P_M * P_M).coef
        return  RationalFunction(deriv_coef_L, deriv_coef_M)
    
    def deriv(self, axis, order=1):
        deriv = self
        for i in range(order):
            if len(deriv.coef_L) == 0:
                break
            deriv = deriv._diff(axis=axis)
        return deriv

def deg_as_array(deg, dims):
    if type(deg) == int:
        _deg = np.array([deg]*dims)
    else:
        _deg = np.asarray(deg)
    return _deg

def fit_polynomial(x, y, deg):
    '''Returns NDPolynomial fit of x, y data.
    
    Parameters:
    x: `np.ndarray` of shape (N, M) where N is number of datapoints, M is number of features in x.
    y: `np.ndarray` of shape (N,) where N is number of datapoints
    deg: int or tuple
        If deg is an int, the highest degree of x1, ..., xM will all be deg.
        If deg is a tuple (d1, .., dM), it corresponds to the highest degrees of x1, ..., xM.
    '''
    _x = _as_vector(x)
    _y = _as_vector(y)

    _deg = deg_as_array(deg, dims=_x.shape[-1])
    assert len(_deg) == _x.shape[-1], "length of degrees, deg, must match number of features in x"
    degrees = _get_degrees(_deg)
    monomials = _vandermonde(_x, degrees)
    coef = np.linalg.lstsq(monomials, _y, rcond=None)[0]
    coef_shape = tuple(np.array(_deg) + 1)
    return NDPolynomial(coef.reshape(coef_shape))

def fit_rational_func(x, y, L, M):
    '''Returns RationalFunction fit of x, y data on the form f(x) = P_L(x)/Q_M(x)
    where P_L is a polynomial of order L and Q_M is of order M.
    
    Parameters:
    x: `np.ndarray` of shape (N, M) where N is number of datapoints, M is number of features in x.
    y: `np.ndarray` of shape (N,) where N is number of datapoints
    L: `int` or `tuple`. Nominator degrees
    M: `int` or `tuple`. Denominator degrees
        If L and M are integers, they are taken to be the highest degree
        in the nominator and denominator polynomial respectively.
        If L and M are tuples, the polynomials assume to take multidimensional inputs 
        and a monomial basis for each polynomial is generated automatically.
        The 0th order coefficient of Q_M is always taken to be 1.
    '''
    _x = _as_vector(x)
    _y = _as_vector(y)

    _deg_L = deg_as_array(L, _x.shape[-1])
    _deg_M = deg_as_array(M, _x.shape[-1])
    assert len(_deg_L) == _x.shape[-1], "length of degrees, deg_L, must match number of features in x"
    assert len(_deg_M) == _x.shape[-1], "length of degrees, deg_M, must match number of features in x"

    degrees_L = _get_degrees(_deg_L)
    degrees_M = _get_degrees(_deg_M)
    VL = _vandermonde(_x, degrees_L)
    VM = _vandermonde(_x, degrees_M)
    W = np.concatenate([VL, - _y * VM[:, 1:]], axis=1)
    coef_LM = np.linalg.lstsq(W, y, rcond=None)[0]
    coef_L = coef_LM[:np.prod(_deg_L + 1)]
    coef_M = np.concatenate([np.array([1]), coef_LM[np.prod(_deg_L + 1):]])
    
    L_shape = tuple(_deg_L + 1)
    M_shape = tuple(_deg_M + 1)

    return RationalFunction(coef_L.reshape(L_shape), coef_M.reshape(M_shape))