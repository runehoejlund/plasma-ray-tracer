import torch
from torch.autograd import grad
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from warnings import warn

def get_boundary_events(r_min, r_max, atol):
    events = []

    if (r_min is not None):
        def hit_min(t, q):
            return np.min(q[:3] - (r_min - atol))
        hit_min.terminal = True
        hit_min.direction = -1
        events.append(hit_min)
    
    if (r_max is not None):
        def hit_max(t, q):
            return np.min((r_max + atol) - q[:3])
        hit_max.terminal = True
        hit_max.direction = -1
        events.append(hit_max)
    
    return events

def get_boundary_events_1D(x_min, x_max, atol):
    events = []

    if (x_min is not None):
        def hit_min(t, q):
            return np.min(q[0] - (x_min - atol))
        hit_min.terminal = True
        hit_min.direction = -1
        events.append(hit_min)
    
    if (x_max is not None):
        def hit_max(t, q):
            return np.min((x_max + atol) - q[0])
        hit_max.terminal = True
        hit_max.direction = -1
        events.append(hit_max)
    
    return events

def trace_ray(r0, k0, omega0, tmin, tmax, D, D_args={}, rtol=1e-3, r_min=None, r_max=None, tsteps=1000):
    q0 = np.hstack((r0,k0))
    
    # RHS of ray-tracer ODE
    def f(t, q):
        x, y, z, kx, ky, kz = q
        omega = torch.tensor(omega0, requires_grad=True)
        r = torch.tensor([x, y, z], requires_grad=True)
        k = torch.tensor([kx, ky, kz], requires_grad=True)

        Di = D(r, k, omega, **D_args)
        Di.backward()
        grad_r = r.grad
        grad_k = k.grad
        RHS_r = - grad_k
        RHS_k = grad_r
        
        return torch.hstack((RHS_r, RHS_k)).detach().numpy()

    sol = solve_ivp(f, [tmin, tmax], q0, t_eval = np.linspace(tmin, tmax, tsteps), events=get_boundary_events(r_min, r_max, atol=(1e-3)*rtol), rtol=rtol, atol = (1e-3)*rtol)
    if not sol.success:
        warn(sol.message)
    return sol

def extract_sol(sol):
    ND = int(sol.y.shape[0]/2)
    xs = sol.y[:ND].T
    ks = sol.y[ND:].T
    zs = np.concatenate([xs, ks], axis=-1)

    # ray tracing stops when it hits boundary, so we don't know
    # exact number of timesteps before ray tracing has completed.
    t = sol.t
    nt = len(t)

    return t, xs, ks, zs, nt

def trace_ray_1D(x0, k0, omega0, tmin, tmax, D, D_args={}, rtol=1e-3, x_min=None, x_max=None, tsteps=1000, solve_ivp_args={}, true_time=False, ghost_ratio=0):
    q0 = np.hstack((x0, k0))
    
    # RHS of ray-tracer ODE
    def f(t, q):
        x_np, k_np = q
        omega = torch.tensor(omega0, requires_grad=True)

        x = torch.tensor(x_np, requires_grad=True)
        k = torch.tensor(k_np, requires_grad=True)

        Di = D(x, k, omega, **D_args)
        Di.backward()
        grad_x = x.grad
        grad_k = k.grad
        RHS_x = - grad_k
        RHS_k = grad_x
        
        if true_time:
            RHS_x = RHS_x/omega.grad
            RHS_k = RHS_k/omega.grad
        
        return torch.hstack((RHS_x, RHS_k)).detach().numpy()

    t_eval = np.linspace(tmin, tmax, tsteps)
    sol = solve_ivp(f, [tmin, tmax], q0, t_eval = t_eval, events=get_boundary_events_1D(x_min, x_max, atol=(1e-3)*rtol), rtol=rtol, atol = (1e-3)*rtol, **solve_ivp_args)
    
    if not sol.success:
        warn(sol.message)
        return

    t_C, xs_C, ks_C, zs_C, nt_C = extract_sol(sol)

    # If we have no ghost points, simply return solution between tmin and tmax
    if ghost_ratio == 0:
        i_start = 0
        i_end = nt_C
        return t_C, xs_C, ks_C, zs_C, i_start, i_end
    
    # Trace backwards and forwards to get ghost point data
    T = tmax - tmin
    dt = T/(tsteps - 1)
    t_ghost_L = np.arange(np.min(sol['t']), np.min(sol['t']) - ghost_ratio*T, -dt)
    t_ghost_R = np.arange(np.max(sol['t']), np.max(sol['t']) + ghost_ratio*T, dt)

    sol_L = solve_ivp(f, [np.max(t_ghost_L), np.min(t_ghost_L)], sol['y'][..., 0], t_eval = t_ghost_L, rtol=rtol, atol = (1e-3)*rtol, **solve_ivp_args)
    sol_R = solve_ivp(f, [np.min(t_ghost_R), np.max(t_ghost_R)], sol['y'][..., -1], t_eval = t_ghost_R, rtol=rtol, atol = (1e-3)*rtol, **solve_ivp_args)

    t_L, xs_L, ks_L, zs_L, nt_L = extract_sol(sol_L)
    t_L, xs_L, ks_L, zs_L = t_L[:0:-1], xs_L[:0:-1], ks_L[:0:-1], zs_L[:0:-1]
    nt_L = nt_L - 1
    t_R, xs_R, ks_R, zs_R, nt_R = extract_sol(sol_R)
    t_R, xs_R, ks_R, zs_R = t_R[1:], xs_R[1:], ks_R[1:], zs_R[1:]
    nt_R = nt_R - 1

    # Concatenate ghost point traces and central trace:
    t = np.hstack([t_L, t_C, t_R])
    xs = np.vstack([xs_L, xs_C, xs_R])
    ks = np.vstack([ks_L, ks_C, ks_R])
    zs = np.vstack([zs_L, zs_C, zs_R])
    
    i_start = nt_L # index of first (non-ghost) datapoint
    i_end = len(t) - nt_R # index of first right ghost datapoint

    return t, xs, ks, zs, i_start, i_end

def get_t(sol, omega, D, D_args = {}):
    # Get derivative of dispersion rel. w.r.t. omega
    grad_omega = np.zeros_like(sol.t)

    omega = torch.tensor(omega, requires_grad=True)
    for i, _t in enumerate(sol.t):
        Di = D(r=sol.y[:3,i], k=sol.y[3:,i], omega=omega, **D_args)
        grad_omega[i] = grad(Di, omega, create_graph=True)[0].detach().numpy()

    # Calculate physical time as t = ∫ d tau D_omega(tau)
    t = cumulative_trapezoid(grad_omega, sol.t, initial=0)
    return t