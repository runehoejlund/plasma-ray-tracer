import numpy as np
import finite_diff as fd
import util as ut
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d, RegularGridInterpolator, LinearNDInterpolator
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root
from gauss_freud_quad import integrate_gauss_freud_quad, get_nodes_and_weights, get_max_nodes
from warnings import warn
from baryrat import aaa

nodes, _ = get_nodes_and_weights(1)
node0 = nodes[0]

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
        P[..., k][norm(P, k) != 0] = P[..., k][norm(P, k) != 0]/norm(P, k)[norm(P, k) != 0]
    return P

def get_symplectic_tangent_trfm(zs, t, ND, i_start, i_end):
    gradt_z = fd.grad(zs, t)[i_start:i_end]
    norms = np.sqrt(ut.inner_product(gradt_z, gradt_z))
    T1 = gradt_z/np.stack([norms]*zs.shape[-1], axis=-1) # normalised grad_t z(t, y0, z0)

    # For each tau, create an identity matrix
    eye = ut.eye((*zs.shape, zs.shape[-1]))[i_start:i_end]
    
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

def get_prefactor(phi0, xs, ks, t, i_start, i_end, B, ranks, Lambda_rhos, A_zetas, R):
    # Calculate prefactor
    dt_x0 = fd.local_grad(xs, i_start, t)
    if dt_x0 == 0:
        dt_x0 = 1
    Nt = (phi0 * np.emath.sqrt(dt_x0/np.mean(dt_x0))
        * np.exp(1j * ( cumulative_trapezoid(ut.inner_product(fd.grad(xs.squeeze(), t)[i_start:i_end, ..., np.newaxis], ks[i_start:i_end]), t[i_start:i_end], initial=0, axis=0) ))
        ) / (
        np.emath.power((- 1j * 2*np.pi), (ranks/2)) * (
            ut.continuous_sqrt_of_reals(
                np.sign(B.squeeze())
                * np.abs(np.linalg.det(Lambda_rhos)
                         * np.linalg.det(A_zetas))
                * np.linalg.det(R)
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

def start_angles(ddf0):
    alpha = np.angle(ddf0)
    sigma_p = -np.pi/4 - alpha/2 + np.pi/2
    sigma_m = -np.pi/4 - alpha/2 - np.pi/2
    return sigma_p, sigma_m

def new_angles(f, sigma_p, sigma_m, lamb):
    '''Calculate new direction of steepest descent
        as the descent which is closest to current direction, sigma.'''
    r = np.abs(node0 * lamb)
    C_circ = lambda _r, theta: _r*np.exp(1j*theta)
    F_circ = lambda theta: np.imag(f(C_circ(r, theta)))
    sigmas = np.linspace(0, 2*np.pi, 1000)

    try:
        argmaxima = argrelextrema(F_circ(sigmas), np.greater)[0]
        new_sigma_p_arg = argmaxima[np.argmin(np.abs( ((sigma_p % (2*np.pi)) - sigmas[argmaxima] + np.pi) % (2*np.pi) - np.pi ))]

        argmaxima_m = argmaxima[argmaxima != new_sigma_p_arg]
        new_sigma_m_arg = argmaxima_m[np.argmin(np.abs( ((sigma_m % (2*np.pi)) - sigmas[argmaxima_m] + np.pi) % (2*np.pi) - np.pi ))]

        new_sigma_p = sigmas[new_sigma_p_arg]
        new_sigma_m = sigmas[new_sigma_m_arg]
    except:
        warn('error finding steepest descent direction. will reuse last iterations direction')
        new_sigma_p = sigma_p
        new_sigma_m = sigma_m
    return new_sigma_p, new_sigma_m

def get_l_and_s(f, sigma_p, sigma_m, l_p, l_m, smin, ddf0, eps_0=0):
    Delta_F = 1

    C_p = lambda l: eps_0 + np.abs(l) * np.exp(1j*sigma_p)
    C_m = lambda l: eps_0 + np.abs(l) * np.exp(1j*sigma_m)

    F_p = lambda l: np.imag(f(C_p(l)))
    F_m = lambda l: np.imag(f(C_m(l)))

    l0_p = l_p
    l0_m = l_m
    if l_p == None or l_m == None:
        l0 = np.sqrt(np.abs(Delta_F/ddf0))
        l0_p, l0_m = l0, l0
    sol_p = root(lambda l: F_p(l) - F_p(0) - Delta_F, l0_p)
    sol_m = root(lambda l: F_m(l) - F_m(0) - Delta_F, l0_m)

    _l_p, _l_m = l0_p, l0_m
    if sol_p.success:
        _l_p = np.abs(sol_p['x'][0])
    else:
        warn('problem with finding l_p:' + sol_p['message'])

    if sol_m.success:
        _l_m = np.abs(sol_m['x'][0])
    else:
        warn('problem with finding l_p:' + sol_m['message'])

    _s_p = max(smin, Delta_F/(np.abs(_l_p)**2))
    _s_m = max(smin, Delta_F/(np.abs(_l_m)**2))

    return _l_p, _l_m, _s_p, _s_m

def get_angles(sigma_p, sigma_m, lamb, f, ddf0):
    '''returns new _sigma_p, _sigma_m'''
    if sigma_p == None or sigma_m == None:
        _sigma_p, _sigma_m = start_angles(ddf0)
    else:
        _sigma_p, _sigma_m = new_angles(f, sigma_p, sigma_m, lamb)
    return _sigma_p, _sigma_m

def integrate_osc_func(f, g, sigma_p, sigma_m, lamb, gauss_quad_order):
    sigma_p, sigma_m, lamb = sigma_p.astype(complex), sigma_m.astype(complex), lamb.astype(complex)
    h = lambda eps: g(eps) * np.exp(1j*f(eps))
    dl_p, dl_m = lamb * np.exp(1j*sigma_p), lamb * np.exp(1j*sigma_m)
    I = integrate_gauss_freud_quad(
        lambda l: (h(l*dl_p) * dl_p - h(l*dl_m) * dl_m),
        n = gauss_quad_order,
        dims=len(sigma_p)
    )
    return I

def check_rays(t, zs):
    ND = int(zs.shape[-1]/2)
    assert ND == 1, 'Only 1D MGO currently supported, last dimension of zs was found to be ' + str(zs.shape[-1]) + ' corresponding to ND = ' + str(ND) 

def _get_ND(zs):
    return int(zs.shape[-1]/2)

def _get_eps_rho(Xs_t1, it1, rho):
    eps_t1 = Xs_t1 - Xs_t1[it1]
    eps_rho = eps_t1[:, :rho]
    return eps_rho

def _get_SVD_projected_qtys(A_rhos, Lambda_rhos, Xs_t1, Ks_t1, ranks, it, it1):
    rho = ranks[it]
    a_rho = A_rhos[it, :rho, :rho]
    Lambda_rho = Lambda_rhos[it, :rho, :rho]
    Ks_rho = Ks_t1[:, :rho]
    return rho, a_rho, Lambda_rho, Ks_rho

def _get_max_abs(eps):
    return min(np.max(eps[eps > 0], initial=0), np.max(-eps[eps < 0], initial=0))

def _get_max_extrapolation(S_t1, zs, it_all, eps_all):
    ND = _get_ND(zs)
    K = np.abs((S_t1[ND:, :] @ zs[it_all, :, np.newaxis]).squeeze())
    wavelength = 2*np.pi/K
    epsmax = _get_max_abs(eps_all)
    if epsmax == 0:
        print('epsmax is zero at it_all:', it_all)
    max_extrapolation = 1 + wavelength/(2*epsmax)
    return max_extrapolation

def _get_Xs_t1_and_Ks_t1(Xs_t1_all, mask_t1, S_t1, zs):
    ND = _get_ND(zs)
    Xs_t1 = Xs_t1_all[mask_t1]
    Ks_t1 = (S_t1[ND:, :] @ zs[mask_t1, :, np.newaxis])
    return Xs_t1, Ks_t1

def _get_eikonal_fields(t, Xs_t1, Ks_t1, J_t1_all, mask_t1, it1, it, it_all, eps_rho, A_rhos, Lambda_rhos, ranks):
    # Eikonal Amplitude
    gradtau_Xs_t1 = J_t1_all[mask_t1][..., np.newaxis]
    Phi_t1 = np.emath.sqrt(J_t1_all[it_all]/J_t1_all[mask_t1]) # Amplitude set to 1 at tau = t1

    # Eikonal Phase
    int_0_to_tau = cumulative_trapezoid(Ks_t1.squeeze()*gradtau_Xs_t1.squeeze(), t[mask_t1], initial=0, axis=0)
    Theta_t1 = int_0_to_tau - int_0_to_tau[it1]

    # Phase factor for inverse MT
    rho, a_rho, Lambda_rho, Ks_rho = _get_SVD_projected_qtys(A_rhos, Lambda_rhos, Xs_t1, Ks_t1, ranks, it, it1)
    f_t1 = Theta_t1 + (
                    - ((1/2) * ut.transpose(eps_rho[..., np.newaxis]) @ a_rho @ np.linalg.inv(Lambda_rho) @ eps_rho[..., np.newaxis]).squeeze()
                    - (ut.transpose(eps_rho[..., np.newaxis]) @ Ks_rho[it1, ..., np.newaxis]).squeeze()
                    ).squeeze()

    return f_t1, Phi_t1, Theta_t1, rho, Xs_t1, Ks_t1

def _get_lamb(t, Xs_t1_all, S_t1, zs, J_t1_all, it, it_all, A_rhos, Lambda_rhos, ranks):
    def grad(order):
        mask_nbh, it_nbh = ut.neighbourhood(it_all, len(t), N_neighbours=order)
        Xs_nbh, Ks_nbh = _get_Xs_t1_and_Ks_t1(Xs_t1_all, mask_nbh, S_t1, zs)
        eps_rho_nbh = _get_eps_rho(Xs_nbh, it_nbh, ranks[it])
        f_nbh, *_ = _get_eikonal_fields(t, Xs_nbh, Ks_nbh, J_t1_all, mask_nbh, it_nbh, it, it_all, eps_rho_nbh, A_rhos, Lambda_rhos, ranks)
        return fd.local_grad(f_nbh[:, np.newaxis].squeeze(), it_nbh, eps_rho_nbh.squeeze(), axes=[0], order=order)
    
    def Taylor_coeff(order):
        deriv = grad(order)
        return (1/(np.math.factorial(order)) * np.abs(deriv))**(1/order)
    
    n = 2
    coeff_n = Taylor_coeff(n)
    coeff_np1 = Taylor_coeff(n + 1)
    while coeff_n < coeff_np1:
        n = n + 1
        coeff_n = coeff_np1
        coeff_np1 = Taylor_coeff(n + 1)
    
    lamb = coeff_n**(-1)

    return lamb

def _get_gauss_quad_order(max_nodes, lamb, epsmax):
    gauss_quad_order = np.sum(lamb * max_nodes < epsmax)
    if gauss_quad_order == 0:
        gauss_quad_order = 1
    return gauss_quad_order

def _get_mask_t1(max_nodes, t, Xs_t1_all, S_t1, zs, J_t1_all, it, it_all, A_rhos, Lambda_rhos, ranks):
    # Get mask for current branch
    mask_sgn = ut.sgn_mask_from_seed(J_t1_all, (it_all))

    # Get mask for range required for Gauss Freud Quadratures
    eps_all = _get_eps_rho(Xs_t1_all, it_all, ranks[it])
    max_extrapolation = _get_max_extrapolation(S_t1, zs, it_all, eps_all)
    _lamb = _get_lamb(t, Xs_t1_all, S_t1, zs, J_t1_all, it, it_all, A_rhos, Lambda_rhos, ranks)
    epsmax = max_extrapolation * _get_max_abs(eps_all)
    gauss_quad_order = _get_gauss_quad_order(max_nodes, _lamb, epsmax)
    gnodes, _ = get_nodes_and_weights(gauss_quad_order)
    lamb = min(_lamb, epsmax/gnodes[-1]) # ensure lambda * gnodes[-1] does not bring us out of the domain
    mask_eps = np.abs(eps_all.squeeze()) < 1.1 * (lamb * gnodes[-1]) # only include points necessary for the Gaussian quadratures

    # Combine masks
    mask_t1 = np.logical_and(mask_sgn, mask_eps)
    return mask_t1, lamb, gauss_quad_order

def get_mgo_field(t, zs, phi0, i_start, i_end, i_save=[],
        analytic_cont={'phase': {'fit_func': aaa, 'kwargs': {'mmax': 20}},
                       'amplitude': {'fit_func': aaa, 'kwargs': {'mmax': 20}}},
        max_gauss_quad_order=5):
    '''Returns branch_masks, ray_field, info
    '''
    check_rays(t, zs)
    max_nodes = get_max_nodes(max_gauss_quad_order)
    nt = i_end - i_start
    ND = _get_ND(zs)
    xs = zs[..., :ND]
    ks = zs[..., ND:]

    S = get_symplectic_tangent_trfm(zs, t, ND, i_start, i_end)
    gradtau_z = fd.grad(zs, t)[i_start:i_end]
    A, B, Q, R, ranks, A_zetas, A_rhos, Lambda_rhos = decompose_symplectic_trfm(S, gradtau_z, ND)
    Nt = get_prefactor(phi0, xs, ks, t, i_start, i_end, B, ranks, Lambda_rhos, A_zetas, R)

    saved_results = []
    Upsilon = np.zeros(nt, dtype=np.cdouble)
    J = gradtau_z[:, :ND].squeeze()
    branch_masks, seeds, branch_ranges = get_branches(J)
    
    for seed, branch_range in zip(seeds, branch_ranges):
        sigma_p, sigma_m = np.array([None]), np.array([None])

        for it in branch_range:
            if ((it - branch_range.start) % 50) == 0:
                print('branch:', branch_range, ', it:', it, ' '*40, end='\r')
            S_t1 = S[it]
            it_all = i_start + it
            Xs_t1_all = (S_t1[:ND, :] @ zs[..., np.newaxis])
            J_t1_all = fd.grad(Xs_t1_all.squeeze(), t)
            
            mask_t1, lamb, gauss_quad_order = _get_mask_t1(max_nodes, t, Xs_t1_all, S_t1, zs, J_t1_all, it, it_all, A_rhos, Lambda_rhos, ranks)
            it1 = int(np.argwhere(t[mask_t1] == t[it_all]))

            Xs_t1, Ks_t1 = _get_Xs_t1_and_Ks_t1(Xs_t1_all, mask_t1, S_t1, zs)
            eps_rho = _get_eps_rho(Xs_t1, it1, ranks[it])
            f_t1, Phi_t1, Theta_t1, rho, Xs_t1, Ks_t1 = _get_eikonal_fields(t, Xs_t1, Ks_t1, J_t1_all, mask_t1, it1, it, it_all, eps_rho, A_rhos, Lambda_rhos, ranks)
            
            f_fit = analytic_cont['phase']['fit_func'](eps_rho.squeeze(), f_t1.squeeze(), **analytic_cont['phase']['kwargs'])
            g_fit = analytic_cont['amplitude']['fit_func'](eps_rho.squeeze(), Phi_t1.squeeze(), **analytic_cont['amplitude']['kwargs'])

            for l in range(rho):
                ddf0 = fd.local_grad(f_t1.squeeze(), it1, eps_rho.squeeze(), axes=[l], order=2)
                sigma_p[l], sigma_m[l] = get_angles(sigma_p[l], sigma_m[l], lamb, f_fit, ddf0)
            
            if it in i_save:
                saved_results.append({'t1': t[it_all], 'it': it, 'it_all': it_all, 'mask_t1': mask_t1, 'it1': it1, 'gauss_quad_order': gauss_quad_order, 'lamb': lamb,
                                'S_t1': S_t1, 'Xs_t1_all': Xs_t1_all, 'Xs_t1': Xs_t1, 'Ks_t1': Ks_t1, 'eps_rho': eps_rho,
                                'sigma_p': np.copy(sigma_p), 'sigma_m': np.copy(sigma_m),
                                'f_t1': f_t1, 'f_fit': f_fit, 'Theta_t1': Theta_t1, 'Phi_t1': Phi_t1, 'g_fit': g_fit})
            Upsilon[it] = integrate_osc_func(f_fit, g_fit, sigma_p[:rho], sigma_m[:rho], lamb, gauss_quad_order=gauss_quad_order)

    ray_field = Nt*Upsilon
    info = {'saved_results': saved_results,
            'Nt': Nt, 'Upsilon': Upsilon,
            'S': S, 'Q': Q, 'R': R,
            'ranks': ranks, 'A_zetas': A_zetas, 'A_rhos': A_rhos, 'Lambda_rhos': Lambda_rhos}
    return branch_masks, ray_field, info

def interpolate_eikonal(x, field):
    phi = np.abs(field)
    theta = ut.continuous_angle(field)
    r_phi = aaa(x, phi, mmax=20)
    r_theta = aaa(x, theta, mmax=20)
    r = lambda x: r_phi(x) * np.exp(1j*r_theta(x))
    return r

def get_mask_with_margin(mask):
    margin_mask = np.logical_or(
            np.diff((mask).astype(int), append=0) > 0,
            np.diff((mask).astype(int), prepend=0) < 0
            )
    
    return np.logical_or(mask, margin_mask)

def get_A0_and_interpolation(phi0, x0, xs, i_start, i_end, branch_masks, ray_field, interpolation_method='linear'):
    '''returns A0, interp_field needed to superpose ray fields satisfying boundary condition.
    interpolation_method must one of 'eikonal_baryrat', 'baryrat' for a barycentric rational interpolation
        or alternatively one of 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous' or 'next' to use scipy's interp1d interpolation function.
    '''
    ND = xs.shape[-1]
    in_regions = [get_covered_region(xs[i_start:i_end][get_mask_with_margin(mask)]) for mask in branch_masks]
    
    if interpolation_method in ['linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']:
        branches = [interp1d(xs[i_start:i_end][mask].squeeze(), ray_field[mask], kind=interpolation_method, bounds_error=False, fill_value='extrapolate') for mask in branch_masks]
    elif interpolation_method == 'eikonal_baryrat':
        branches = [interpolate_eikonal(xs[i_start:i_end][mask].squeeze(), ray_field[mask]) for mask in branch_masks]
    elif interpolation_method == 'baryrat':
        branches = [aaa(xs[i_start:i_end][mask].squeeze(), ray_field[mask], mmax=20) for mask in branch_masks]

    def interp_field(x):
        return sum(in_region(np.asarray(x).reshape(-1, ND)) * f(x) for in_region, f in zip(in_regions, branches))

    A0 = phi0/interp_field(x0)

    return A0, interp_field

def superpose_ray_fields(phi0, x0, xs, branch_masks, ray_field, i_start, i_end, interpolation_method='linear'):
    '''returns interpolated function `field` 
        interpolation_method must one of 'eikonal_baryrat', 'baryrat' for a barycentric rational interpolation
        or alternatively one of 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous' or 'next' to use scipy's interp1d interpolation function.
        '''
    
    A0, interp_field = get_A0_and_interpolation(phi0, x0, xs, i_start, i_end, branch_masks, ray_field, interpolation_method=interpolation_method)

    def field(x):
        return A0 * interp_field(x)
    
    return field

def get_go_field_1D(t, zs, phi0, i_start, i_end):
    '''returns branch_masks, ray_field'''
    check_rays(t, zs)
    ND = int(zs.shape[-1]/2)
    xs = zs[..., :ND]
    ks = zs[..., ND:]

    gradtau_x = fd.grad(xs.squeeze(), t)[i_start:i_end]
    J = gradtau_x.squeeze()
    branch_masks, seeds, branch_ranges = get_branches(J)
    theta = cumulative_trapezoid(ks[i_start:i_end].squeeze()*gradtau_x.squeeze(), t[i_start:i_end], initial=0) + ut.continuous_angle_of_reals(J)
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

def get_covered_region(rs, points_per_bin=2):
    '''returns `in_region` function which takes positions as input and returns boolean
    values indicating whether each position is covered by the rays with positions `rs`'''
    ND = rs.shape[-1]
    bins = (*(int(s/points_per_bin) for s in rs.shape[:ND]), )
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