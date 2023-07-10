import numpy as np
import finite_diff as fd
import util as ut
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d, RegularGridInterpolator, LinearNDInterpolator
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root
from gauss_freud_quad import integrate_gauss_freud_quad
from diff_func_fitters import fit_polynomial, fit_rational_func
from warnings import warn

#%% Symplectic Transformation
def gram_schmidt_orthogonalize(Q):
    N = Q.shape[-1]
    P = np.zeros(Q.shape)

    def norm(A, i):
        norms = np.sqrt(ut.inner_product(A[..., i], A[..., i]))
        return np.stack([norms]*A.shape[-1], axis=-1)

    P[..., 0] = Q[..., 0]/norm(Q, 0)
    for k in range(1, N):
        P[..., k] = (Q[..., k]
                        - sum([(
            np.stack([ut.inner_product(Q[..., k], P[..., j])]*N, axis=-1)
            ) * P[..., j] for j in range(k)]))
        P[..., k] = P[..., k]/norm(P, k)
    return P

def get_symplectic_tangent_trfm(zs, t, ND):
    gradt_z = fd.grad(zs, t)
    norms = np.sqrt(ut.inner_product(gradt_z, gradt_z))
    T1 = gradt_z/np.stack([norms]*zs.shape[-1], axis=-1) # normalised grad_t z(t, y0, z0)

    # For each tau, create an identity matrix
    eye = ut.eye((*zs.shape, zs.shape[-1]))
    
    # For each tau, create orthonormal basis starting from
    # single tangent vector using Gram Schmidt orthogonalization
    ONB = np.copy(eye)
    ONB[..., 0] = T1
    ONB = gram_schmidt_orthogonalize(ONB)

    # Tangent space is first 3 vectors of basis:
    # Note: Basis vectors in T are shaped as columns!
    T = ONB[..., :ND]

    symplJ = np.zeros_like(ONB)
    symplJ[..., :, :] =  np.block([
        [  np.zeros((ND, ND)),  np.eye((ND))        ],
        [  -np.eye(ND),         np.zeros((ND, ND))  ]])
    
    N = -np.matmul(symplJ, T)
    R = np.concatenate((T, N), axis=-1)
    S = ut.transpose(R)
    return S

def decompose_symplectic_trfm(S, gradtau_z, ND):
    A, B = S[..., :ND, :ND], S[..., :ND, ND:]
    Q = np.block([[ut.transpose(A)], [ut.transpose(B)]])
    R = ut.transpose(Q) @ gradtau_z[..., np.newaxis]

    Us, lambs, Vs = np.linalg.svd(B)
    Lambdas = ut.diag(lambs)
    ranks = np.linalg.matrix_rank(B)

    eyes = ut.eye(A.shape)
    eye_rhos = np.copy(eyes)
    for i in range(A.shape[-1]):
        eye_rhos[ranks == i, i:, i:] = 0
    eye_zetas = eyes - eye_rhos
    A_tilde = ut.transpose(Us) @ A @ Vs
    A_zetas = (ut.transpose(eye_zetas) @ A_tilde @ eye_zetas) + eye_rhos
    A_rhos = (ut.transpose(eye_rhos) @ A_tilde @ eye_rhos) + eye_zetas
    Lambda_rhos = (ut.transpose(eye_rhos) @ Lambdas @ eye_rhos) + eye_zetas

    return A, B, Q, R, ranks, A_zetas, A_rhos, Lambda_rhos

# %% Calculating Prefactor

def get_prefactor(phi0, xs, ks, t, B, ranks, Lambda_rhos, A_zetas, R):
    # Calculate prefactor
    dt_x0 = fd.grad(xs[:3, ..., 0], t)[0, ...]
    Nt = (phi0 * np.emath.sqrt(dt_x0/np.mean(dt_x0))
        * np.exp(1j * ( cumulative_trapezoid(ut.inner_product(fd.grad(xs.squeeze(), t)[..., np.newaxis], ks), t, initial=0, axis=0) ))
        ) / (
        np.emath.power((- 1j * 2*np.pi), (ranks/2)) * (
            ut.continuous_sqrt_of_reals(
                np.sign(B.squeeze()) * np.abs(np.linalg.det(Lambda_rhos) * np.linalg.det(A_zetas)) * np.linalg.det(R)
            )
        )
    )
    assert np.all(ranks == 1), 'only supports ranks 1'
    return Nt

# %% Calculating Upsilon

def get_branches(J):
    branch_masks = ut.get_masks_of_const_sgn(J, ND=1)
    J_desc = np.argsort(np.abs(J))[::-1]
    seeds = []
    branch_ranges = []
    for branch in branch_masks:
        branch_min, branch_max = np.min(np.argwhere(branch)), np.max(np.argwhere(branch))
        seed = J_desc[(branch_min <= J_desc) & (J_desc <= branch_max)][0]

        range_back, range_forward = range(seed, max(branch_min -1, -1), -1), range(seed, min(branch_max + 1, J.shape[0]), +1)
        if len(range_back) > 1:
            branch_ranges.append(range_back)
            seeds.append(seed - 1)
        if len(range_forward) > 1:
            branch_ranges.append(range_forward)
            seeds.append(seed)
    return branch_masks, seeds, branch_ranges

def start_angles(ddf, eps_0=0):
    alpha = np.angle(ddf(eps_0))
    sigma_p = -np.pi/4 - alpha/2 + np.pi/2
    sigma_m = -np.pi/4 - alpha/2 - np.pi/2
    return sigma_p, sigma_m

def new_angles(f, sigma_p, sigma_m, l_p, l_m):
    '''Calculate new direction of steepest descent
        as the descent which is closest to current direction, sigma.'''
    r = np.mean([np.abs(l_p), np.abs(l_m)])
    C_circ = lambda _r, theta: _r*np.exp(1j*theta)
    F_circ = lambda theta: np.imag(f(C_circ(r, theta)))
    sigmas = np.linspace(0, 2*np.pi, 1000)

    argmaxima = argrelextrema(F_circ(sigmas), np.greater)[0]
    new_sigma_p_arg = argmaxima[np.argmin(np.abs( ((sigma_p % (2*np.pi)) - sigmas[argmaxima] + np.pi) % (2*np.pi) - np.pi ))]

    argmaxima_m = argmaxima[argmaxima != new_sigma_p_arg]
    new_sigma_m_arg = argmaxima_m[np.argmin(np.abs( ((sigma_m % (2*np.pi)) - sigmas[argmaxima_m] + np.pi) % (2*np.pi) - np.pi ))]

    new_sigma_p = sigmas[new_sigma_p_arg]
    new_sigma_m = sigmas[new_sigma_m_arg]
    return new_sigma_p, new_sigma_m

def get_l_and_s(f, sigma_p, sigma_m, l_p, l_m, s0, eps_0=0):
    Delta_F = 1

    C_p = lambda l: eps_0 + np.abs(l) * np.exp(1j*sigma_p)
    C_m = lambda l: eps_0 + np.abs(l) * np.exp(1j*sigma_m)

    F_p = lambda l: np.imag(f(C_p(l)))
    F_m = lambda l: np.imag(f(C_m(l)))

    l0_p = l_p
    l0_m = l_m
    if l_p == None or l_m == None:
        l0 = np.sqrt(np.abs(Delta_F/s0))
        l0_p, l0_m = l0, l0
    sol_p = root(lambda l: F_p(l) - F_p(0) - Delta_F, l0_p)
    sol_m = root(lambda l: F_m(l) - F_m(0) - Delta_F, l0_m)

    _l_p, _l_m = l_p, l_m
    if sol_p.success:
        _l_p = np.abs(sol_p['x'][0])
    else:
        warn('problem with finding l_p:' + sol_p['message'])

    if sol_m.success:
        _l_m = np.abs(sol_m['x'][0])
    else:
        warn('problem with finding l_p:' + sol_m['message'])

    _s_p = Delta_F/(np.abs(_l_p)**2)
    _s_m = Delta_F/(np.abs(_l_m)**2)

    return _l_p, _l_m, _s_p, _s_m

def get_angles_and_l_and_s(sigma_p, sigma_m, l_p, l_m, s0, f, ddf=None, eps_0=0):
    '''returns new _sigma_p, _sigma_m, _l_p, _l_m, _s_p, _s_m'''
    if sigma_p == None or sigma_m == None:
        _sigma_p, _sigma_m = start_angles(ddf, eps_0)
    else:
        _sigma_p, _sigma_m = new_angles(f, sigma_p, sigma_m, l_p, l_m)
    _l_p, _l_m, _s_p, _s_m = get_l_and_s(f, _sigma_p, _sigma_m, l_p, l_m, s0)

    return _sigma_p, _sigma_m, _l_p, _l_m, _s_p, _s_m

def integrate_osc_func(f, g, sigma_p, sigma_m, s_p, s_m, gauss_quad_order=10):
    sigma_p, sigma_m, s_p, s_m = sigma_p.astype(complex), sigma_m.astype(complex), s_p.astype(complex), s_m.astype(complex)
    h = lambda eps: g(eps) * np.exp(1j*f(eps))
    dl_p, dl_m = np.exp(1j*sigma_p)/np.sqrt(s_p), np.exp(1j*sigma_m)/np.sqrt(s_m)
    I = integrate_gauss_freud_quad(
        lambda l: (h(l*dl_p) * dl_p - h(l*dl_m) * dl_m),
        n = gauss_quad_order,
        dims=len(sigma_p)
    )
    return I

def check_rays(t, zs):
    ND = int(zs.shape[-1]/2)
    assert ND == 1, 'Only 1D MGO currently supported, last dimension of zs was found to be ' + str(zs.shape[-1]) + ' corresponding to ND = ' + str(ND) 

def get_mgo_field(t, zs, phi0, i_save=[],
                  analytic_cont={'phase': {'fit_func': fit_polynomial, 'kwargs': {'deg': 3, 'exclude_degrees': [1]}},
                             'amplitude': {'fit_func': fit_rational_func, 'kwargs': {'L': 2, 'M': 1, 'optimize': False}}},
                  gauss_quad_order=10):
    '''Returns branch_masks, ray_field, info
    '''
    check_rays(t, zs)
    nt = len(t)
    ND = int(zs.shape[-1]/2)
    xs = zs[..., :ND]
    ks = zs[..., ND:]

    S = get_symplectic_tangent_trfm(zs, t, ND)
    gradtau_z = fd.grad(zs, t)
    A, B, Q, R, ranks, A_zetas, A_rhos, Lambda_rhos = decompose_symplectic_trfm(S, gradtau_z, ND)
    Nt = get_prefactor(phi0, xs, ks, t, B, ranks, Lambda_rhos, A_zetas, R)

    saved_results = []
    Upsilon = np.zeros(nt, dtype=np.cdouble)
    J = gradtau_z[:, :ND].squeeze()
    branch_masks, seeds, branch_ranges = get_branches(J)
    
    for seed, branch_range in zip(seeds, branch_ranges):
        print('branch:', branch_range, ' '*40, end='\r')

        sigma_p, sigma_m, l_p, l_m, s_p, s_m = np.array([None]), np.array([None]), np.array([None]), np.array([None]), np.array([None]), np.array([None])

        for it in branch_range:
            S_t1 = S[it]
            Xs_t1_all = (S_t1[:ND, :] @ zs[..., np.newaxis])
            gradtau_Xs_t1_all = fd.grad(Xs_t1_all.squeeze(), t)
            J_t1 = gradtau_Xs_t1_all
            mask_t1 = ut.sgn_mask_from_seed(J_t1, (it)) # get current branch
            it1 = int(np.argwhere(t[mask_t1] == t[it]))
            
            Phi_t1 = np.emath.sqrt(J_t1[it]/J_t1[mask_t1]) # Amplitude set to 1 at tau = t1

            Xs_t1 = Xs_t1_all[mask_t1]
            gradtau_Xs_t1 = gradtau_Xs_t1_all[mask_t1][..., np.newaxis]
            Ks_t1 = (S_t1[ND:, :] @ zs[mask_t1, :, np.newaxis])
            int_0_to_tau = cumulative_trapezoid(Ks_t1.squeeze()*gradtau_Xs_t1.squeeze(), t[mask_t1], initial=0, axis=0)
            Theta_t1 = int_0_to_tau - int_0_to_tau[it1]
            
            rho = ranks[it]
            a_rho = A_rhos[it, :rho, :rho]
            Lambda_rho = Lambda_rhos[it, :rho, :rho]
            eps_t1 = Xs_t1 - Xs_t1[it1]
            eps_rho = eps_t1[:, :rho]
            Ks_rho = Ks_t1[:, :rho]
            f_t1 = Theta_t1 + (
                    - ((1/2) * ut.transpose(eps_rho[..., np.newaxis]) @ a_rho @ np.linalg.inv(Lambda_rho) @ eps_rho[..., np.newaxis]).squeeze()
                    - (ut.transpose(eps_rho[..., np.newaxis]) @ Ks_rho[it1, ..., np.newaxis]).squeeze()
                    ).squeeze()
            
            f_fit = analytic_cont['phase']['fit_func'](eps_rho.squeeze(), f_t1.squeeze(), **analytic_cont['phase']['kwargs'])
            g_fit = analytic_cont['amplitude']['fit_func'](eps_rho.squeeze(), Phi_t1.squeeze(), **analytic_cont['amplitude']['kwargs'])
            ddf_fit = f_fit.deriv(axis=0, order=2)
            
            for l in range(rho):
                s0 = fd.local_grad(f_t1, it1, eps_rho.squeeze(), axes=[l], order=2)
                sigma_p[l], sigma_m[l], l_p[l], l_m[l], s_p[l], s_m[l] = get_angles_and_l_and_s(sigma_p[l], sigma_m[l], l_p[l], l_m[l], s0, f_fit, ddf_fit)
            
            if it in i_save:
                saved_results.append({'t1': t[it], 'it': it, 'mask_t1': mask_t1, 'it1': it1,
                                'Xs_t1_all': Xs_t1_all, 'Xs_t1': Xs_t1, 'Ks_t1': Ks_t1, 'eps_rho': eps_rho,
                                'sigma_p': np.copy(sigma_p), 'sigma_m': np.copy(sigma_m), 'l_p': np.copy(l_p), 'l_m': np.copy(l_m), 's_p': np.copy(s_p), 's_m': np.copy(s_m),
                                'f_t1': f_t1, 'f_fit': f_fit, 'Theta_t1': Theta_t1, 'Phi_t1': Phi_t1, 'g_fit': g_fit, 'ddf_fit': ddf_fit})
            Upsilon[it] = integrate_osc_func(f_fit, g_fit, sigma_p[:rho], sigma_m[:rho], s_p[:rho], s_m[:rho], gauss_quad_order=gauss_quad_order)

    ray_field = Nt*Upsilon
    info = {'saved_results': saved_results,
            'Nt': Nt, 'Upsilon': Upsilon,
            'S': S, 'Q': Q, 'R': R,
            'ranks': ranks, 'A_zetas': A_zetas, 'A_rhos': A_rhos, 'Lambda_rhos': Lambda_rhos}
    return branch_masks, ray_field, info

def get_A0_and_interpolation(phi0, x0, xs, branch_masks, ray_field):
    '''returns A0, interp_field needed to superpose ray fields satisfying boundary condition.'''
    branches = [interp1d(xs[mask].squeeze(), ray_field[mask], bounds_error=False, fill_value='extrapolate') for mask in branch_masks]

    def interp_field(x):
        return sum(f(x) for f in branches)

    A0 = phi0/interp_field(x0)

    return A0, interp_field

def superpose_ray_fields(phi0, x0, xs, branch_masks, ray_field):
    '''returns interpolated function `field` '''
    A0, interp_field = get_A0_and_interpolation(phi0, x0, xs, branch_masks, ray_field)

    def field(x):
        return A0 * interp_field(x)
    
    return field

def get_go_field_1D(t, zs, phi0):
    '''returns branch_masks, ray_field'''
    check_rays(t, zs)
    ND = int(zs.shape[-1]/2)
    xs = zs[..., :ND]
    ks = zs[..., ND:]

    gradtau_x = fd.grad(xs.squeeze(), t)
    J = gradtau_x.squeeze()
    branch_masks, seeds, branch_ranges = get_branches(J)
    theta = cumulative_trapezoid(ks.squeeze()*gradtau_x.squeeze(), t, initial=0) + ut.continuous_angle_of_reals(J)
    phi = phi0*ut.continuous_sqrt_of_reals(J[0]/J)
    ray_field = phi * np.exp(1j*theta)

    return branch_masks, ray_field

def get_go_field_3D(t, y0s, z0s, zs, phi0):
    '''returns branch_masks, ray_field'''
    ND = int(zs.shape[-1]/2)
    rs = zs[..., :ND]
    ks = zs[..., ND:]
    gradtau_r = fd.grad(rs.squeeze(), t, y0s, z0s)
    gradt_r = gradtau_r[..., 0]
    J = np.linalg.det(gradtau_r).squeeze()
    branch_masks = ut.get_masks_of_const_sgn(J)
    theta = cumulative_trapezoid(ut.inner_product(ks, gradt_r), t, initial=0, axis=0)
    phi = phi0 * ut.continuous_sqrt_of_reals(J[0, ...]/J)
    ray_field = phi * np.exp(1j*theta)

    return branch_masks, ray_field

def get_covered_region(rs):
    '''returns `in_region` function which takes positions as input and returns boolean
    values indicating whether each position is covered by the rays with positions `rs`'''
    ND = rs.shape[-1]
    bins = (*(s for s in rs.shape[:ND]), )
    H, edges = np.histogramdd(rs.reshape(-1, ND), bins=bins)
    centers = [edge[:-1] + np.diff(edge)/2 for edge in edges]
    in_region = RegularGridInterpolator(tuple(centers), (H > 0).astype(int), method='nearest', fill_value=0, bounds_error=False)
    return in_region

def superpose_ray_fields_3D(rs, branch_masks, ray_field, in_region=None):
    '''returns interp_field_r, interp_field, branch_interpolations, in_region'''
    if in_region == None:
        in_region = get_covered_region(rs)
    
    branch_interpolations = [LinearNDInterpolator(rs[mask], ray_field[mask], fill_value=0) for mask in branch_masks]

    def interp_field_r(r):
        return in_region(r)*sum(f(r) for f in branch_interpolations)

    def interp_field(x, y, z):
        r = np.stack([x, y, z], axis=-1)
        return interp_field_r(r)
    
    return interp_field_r, interp_field, branch_interpolations, in_region

def get_in_out_interpolations(in_region, branch_interpolations):
    '''returns interp_field_in_r, interp_field_in, interp_field_out_r, interp_field_out'''
    interp_field_in_r = lambda r: in_region(r)*branch_interpolations[0](r)
    interp_field_out_r = lambda r: in_region(r)*branch_interpolations[1](r)

    def interp_field_in(x, y, z):
        r = np.stack([x, y, z], axis=-1)
        return interp_field_in_r(r)

    def interp_field_out(x, y, z):
        r = np.stack([x, y, z], axis=-1)
        return interp_field_out_r(r)
    
    return interp_field_in_r, interp_field_in, interp_field_out_r, interp_field_out