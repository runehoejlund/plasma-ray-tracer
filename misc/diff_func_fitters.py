from __future__ import annotations

import numpy as np
import itertools
from scipy.ndimage import convolve
from math import floor, ceil
from dataclasses import dataclass
from scipy import optimize
from typing import Callable, List, Tuple
import numdifftools as nd
import sympy as sp

class DifferentiableFunc:
    def __call__(self, x):
        pass

    def deriv(self, axis: int, order: int) -> DifferentiableFunc:
        pass
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
    if len(f_coef) == 0 or len(g_coef) == 0:
        return NDPolynomial([])
    
    f_coef_pad = np.pad(f_coef, [(ceil((s-1)/2), floor((s-1)/2)) for s in g_coef.shape], mode='constant', constant_values=0)
    return NDPolynomial(convolve(f_coef_pad, g_coef, mode='constant', cval=0.0))

@dataclass
class NDPolynomial(DifferentiableFunc):
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

@dataclass
class RationalFunction(DifferentiableFunc):
    '''Class for representing a rational function, i.e. a function which is the ratio of two polynomials:
    f(x) = P_L(x)/P_M(x)
    coef_L: np.ndarray, nominator coefficients
    coef_M: np.ndarray, denominator coefficients
    '''
    coef_L: np.ndarray
    coef_M: np.ndarray

    def __init__(self, coef_L, coef_M):
        assert len(coef_M) > 0, "Can't divide by 0. len of denominator's coef_M must be greater than 0 to represent a nonzero polynomial."
        self.coef_L = coef_L
        self.coef_M = coef_M

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

def fit_polynomial(x, y, deg, exclude_degrees=[]):
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
    coef_shape = tuple(np.array(_deg) + 1)
    coef = np.zeros(coef_shape)

    _exclude_degrees = [(d, ) if type(d) == int else tuple(d) for d in exclude_degrees]
    assert np.all([len(d) == len(_deg) for d in _exclude_degrees]), 'exclude_degrees does not match number of dimensions'
    include = np.array([tuple(d) not in _exclude_degrees for d in degrees])
    coef[include.reshape(coef_shape)] = np.linalg.lstsq(monomials[:, include], _y, rcond=None)[0].squeeze()
    
    return NDPolynomial(coef)

def optimize_rational_func(x, y, coef_L0, coef_M0, deg_L, deg_M):
    # If we also wish to do a non-linear optimization afterwards, this is also possible:
    L_shape = tuple(deg_L + 1)
    M_shape = tuple(deg_M + 1)

    def collect_coef(coef_L, coef_M):
        return np.concatenate([coef_L, coef_M])
    
    def unpack_coef(coef):
        coef_L = coef[:np.prod(deg_L + 1)]
        coef_M = coef[np.prod(deg_L + 1):]
        return coef_L, coef_M
    
    def model(x, *vargs):
        coef = np.concatenate([vargs])
        coef_L, coef_M = unpack_coef(coef)
        return RationalFunction(coef_L.reshape(L_shape), coef_M.reshape(M_shape))(x)

    coef = optimize.curve_fit(model, x, y, p0=collect_coef(coef_L0, coef_M0))[0]
    coef_L, coef_M = unpack_coef(coef)
    return coef_L, coef_M

def fit_rational_func(x, y, L, M, optimize=False):
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

    if optimize:
        coef_L, coef_M = optimize_rational_func(x, y, coef_L, coef_M, _deg_L, _deg_M)

    return RationalFunction(coef_L.reshape(L_shape), coef_M.reshape(M_shape))

@dataclass
class FiniteDiffFunc1D(DifferentiableFunc):
    '''Class for representing function which is differentiated through finite differences
    '''
    func: Callable

    def __call__(self, x):
        _x = _as_vector(x)
        return self.func(_x).squeeze()

    def deriv(self, axis: int, order: int) -> DifferentiableFunc:
        assert axis == 0, 'Only Supports differentiating of 1D function'
        return nd.Derivative(self.func, n=order)

@dataclass
class SymbolicFunc1D:
    '''Class for representing function which is differentiated through symbolic differentiation with sympy
    '''
    sympy_func: sp.core.Symbol
    sympy_arg: sp.core.Symbol
    sympy_params: List[Tuple]
    
    def __call__(self, x):
        func = self.sympy_func.subs(self.sympy_params)
        lamb = sp.lambdify(self.sympy_arg, func, 'numpy')
        if func.is_constant():
            return np.full_like(x, lamb(x))
        else:
            return lamb(x)
    
    def deriv(self, axis, order=1):
        assert axis == 0, 'only supports 1D symbolic func'
        return SymbolicFunc1D(sp.diff(self.sympy_func, (self.sympy_arg, order)), self.sympy_arg, self.sympy_params)

@dataclass
class SumFunc(DifferentiableFunc):
    '''Class for representing sum of two DifferentiableFunc functions
    '''
    funcs: List[DifferentiableFunc]

    def __call__(self, x):
        return sum([f(x) for f in self.funcs])

    def deriv(self, axis, order=1):
        return SumFunc([f.deriv(axis=axis, order=order) for f in self.funcs])
