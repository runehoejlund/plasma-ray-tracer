'''
This file contains a minimal example of ray tracing
for the airy problem using geometrical optics.
The ray tracing is parallelised and therefore quite fast.
However, after ray tracing we need to do an interpolation,
which takes a long time if we increase the number of grid
points.

Currently it takes ~25 s for a 1000 x 10 x 10 (t, y0, z0) grid.
Increasing the grid to 1000 x 50 x 50 points increased the interpolation
time to about 30 min! There might be ways of improving this,
but I haven't explored those. Ideas could be to use another
interpolation library or fitting the discrete datapoints
with a neural network or another simpler regression model.
'''
# %% Setup
from os import path, chdir
chdir(path.dirname(path.abspath(__file__)))

import sys
import os
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
import finite_diff as fd
from trace_ray import trace_ray
from scipy.special import airy
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import LinearNDInterpolator
from scipy.signal import find_peaks
from trace_ray import trace_ray, get_t
from torch_helper import to_torch, to_torch_3D, torch_func, inner_product, angle

import multiprocessing as mp
import time

rcParams.update(mpl.rcParamsDefault)
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"],
    "font.size": 16})
rcParams['axes.titlepad'] = 20

def np_inner_product(a, b):
    return np.einsum('...i,...i->...', a, b)

def get_masks_of_const_sgn(a):
    '''returns list of arrays with masks of 1.0 and 0.0s.
    The masks subdivide the values of the array, `a`,
    into regions where the sign is constant.'''
    masks = [np.ones(a.shape)]
    for axis in range(len(a.shape)):
        splits = np.unique(np.argwhere(np.diff(np.signbit(a), axis=axis))[..., axis] + 1)
        for i, split in enumerate(splits):
            p = np.split(np.ones(a.shape), [split], axis=axis)
            n = np.split(np.zeros(a.shape), [split], axis=axis)
            m1 = np.concatenate([p[0], n[1]], axis=axis)
            m2 = np.concatenate([n[0], p[1]], axis=axis)
            m1s = [m*m1 for m in masks]
            m2s = [m*m2 for m in masks]
            masks = [m for m in [m*m1 for m in masks] + [m*m2 for m in masks] if np.any(m)]
    return [m.astype(bool) for m in masks]

# Load Exact Solution (Needed to set initial condition for field)
def Ai(x):
    ai, *_ = airy(x)
    return ai

def Bi(x):
    _, _, bi, _ = airy(x)
    return bi

x = np.linspace(-8, 0, 1000)
E_ex = Ai(x) # Exact solution is Airy function

########### Ray Tracing using Geometrical Optics #############

# %% Define dispersion relation
@torch_func
def D(r: ('vector'), k: ('vector'), omega: ('scalar')):
    x, y, z = r
    return - x - inner_product(k, k)

# %% Set ICs and grid
first_peak_ind = find_peaks(E_ex)[0][0]
x0  = x[first_peak_ind]
phi0 = E_ex[first_peak_ind]

# Note setting a finer grid increases the interpolation time significantly!
n2, n3 = 10, 10
nt = 1000
y = np.linspace(0, 1, n2)
z = np.linspace(0, 10, n3)

# %% Ray Tracing
rs = np.zeros((nt, n2, n3, 3))
ks = np.zeros((nt, n2, n3, 3))
omega0 = 1.0

# ray tracing stops when it hits boundary, so we don't know
# exact number of timesteps before ray tracing has completed.
t = np.zeros(nt)
sol_nts = []

results = []

def trace_ray_ij(i, j):
    y0, z0 = y[i], z[j]
    sol = trace_ray(r0 = np.array([x0, y0, z0]), k0=np.array([np.sqrt(-x0), 0, 0]), omega0=omega0, tmin=0, tmax=8, D=D, r_min=np.array([x0, 0, 0]), tsteps=nt)
    return (i, j, sol.t, sol.y)

def collect_result(result):
    global results
    results.append(result)

if __name__ == '__main__':
    tic = time.process_time()
    with mp.Pool(processes=mp.cpu_count()) as p:
        # print(p.map(test, range(10)))
        for i, y0 in enumerate(y):
            for j, z0 in enumerate(z):
                p.apply_async(trace_ray_ij, args=(i, j), callback=collect_result)
        p.close()
        p.join()
    
    for (i, j, sol_t, sol_y) in results:
        sol_nt = len(sol_t)
        t = sol_t
        sol_nts.append(sol_nt)
        rs[:sol_nt, i, j, :] = sol_y[:3].T
        ks[:sol_nt, i, j, :] = sol_y[3:].T
    
    # # Clip all rays to the same number of time steps
    nt = np.min(sol_nts)
    t = t[:nt]
    rs = rs[:nt, :, :, :]
    ks = ks[:nt, :, :, :]
    
    print('finished ray tracing in: ', round(time.process_time() - tic, 4), ' s')
    
    tic = time.process_time()
    # %% Field Construction
    J = np.linalg.det(fd.grad(rs, t, y, z))
    phi = phi0*np.real(np.emath.sqrt(J[0, ...]/J))
    gradt_r = fd.grad(rs, t)
    theta0 = 0
    theta = theta0 + cumulative_trapezoid(np_inner_product(ks, gradt_r), t, initial=0, axis=0)

    branch_masks = get_masks_of_const_sgn(J)
    branches = [LinearNDInterpolator(rs[mask], phi[mask]*np.cos(theta[mask])) for mask in branch_masks]

    def interp_field_r(r):
        return sum(f(r) for f in branches)

    def interp_field(x, y, z):
        r = np.stack([x, y, z], axis=-1)
        return interp_field_r(r)

    xi = np.linspace(np.min(rs[..., 0]), np.max(rs[..., 0])+0.1, 200)
    yi = np.linspace(np.min(rs[..., 1]), np.max(rs[..., 1]), 200)
    zi = np.linspace(np.min(rs[..., 2]), np.max(rs[..., 2]), 200)

    E_GO = interp_field(xi, np.zeros_like(xi), np.zeros_like(xi))
    print('reconstructed interpolation in 1D in: ', round(time.process_time() - tic, 4), ' s')

    # %% Plot Results

    # Plot Exact and GO Solution in 1D
    plt.figure(figsize=(5,4))
    plt.plot(x, E_ex, 'k-', label='Exact $E(x) = \mathrm{Ai}(x)$')
    plt.plot(xi, E_GO, '--', color='tab:blue', label='Numerical GO $E(x)$')
    plt.ylim(-1, 1)
    plt.xlabel('$x$')
    plt.ylabel('$E$ [arb. u.]')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/GO_airy_field.png')
    plt.savefig('./plots/GO_airy_field.pdf')
    plt.show()

    tic = time.process_time()
    # Plot GO Solution in 2D
    X, Y = np.meshgrid(x, y, indexing='ij')
    E_XY = interp_field(X, Y, np.zeros_like(X))
    print('reconstructed interpolation in 2D in: ', round(time.process_time() - tic, 4), ' s')

    plt.figure(figsize=(6,4))
    plt.contourf(X, Y, E_XY, cmap='RdBu_r', vmax=0.8, levels=80)
    plt.colorbar(label='$E$ [arb. u.]')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.tight_layout()
    plt.savefig('./plots/GO_airy_field_2d.png')
    plt.show()