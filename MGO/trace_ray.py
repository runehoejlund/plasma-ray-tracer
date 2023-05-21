import torch
from torch.autograd import grad
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid

def get_boundary_events(r_min, r_max):
    events = []

    if (r_min is not None):
        def hit_min(t, q):
            tol = 1e-4
            return np.min(q[:3] - (r_min - tol))
        hit_min.terminal = True
        hit_min.direction = -1
        events.append(hit_min)
    
    if (r_max is not None):
        def hit_max(t, q):
            tol = 1e-4
            return np.min((r_max + tol) - q[:3])
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

    sol = solve_ivp(f, [tmin, tmax], q0, t_eval = np.linspace(tmin, tmax, tsteps), events=get_boundary_events(r_min, r_max), rtol=rtol, atol = (1e-3)*rtol)
    return sol

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