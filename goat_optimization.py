import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, BFGS

from scipy.constants import h
hbar = h / (2*np.pi)

def pack_complex_list(mats):
    if mats.ndim == 2:
        mats = mats[np.newaxis, ...]
    # n = mats.shape[0]
    real_vec = np.concatenate([mats.real.flatten(), mats.imag.flatten()])
    # flat = mats.reshape(n, -1)
    # real_vec = np.concatenate([flat.real, flat.imag], axis=1).ravel()
    return real_vec

def unpack_complex_vector(vec, n_blocks, N):
    """
    vec: real vector of length 2 * n_blocks * N*N
    returns complex array shape (n_blocks, N, N)
    """
    assert vec.size == 2 * n_blocks * N * N
    real_part = vec[:n_blocks*N*N].reshape(n_blocks, N, N)
    imag_part = vec[n_blocks*N*N:].reshape(n_blocks, N, N)
    return real_part + 1j * imag_part

def make_ode_rhs(H0, H_controls, control_funcs, control_param_derivs, control_extra_params=None):
    """
    Returns a function rhs(t, y, alpha) that computes the time derivative of the augmented state.
    - H0: (N,N) complex base Hamiltonian
    - H_controls: list of (N,N) complex control Hamiltonians H_k
    - control_funcs: list of functions c_k(t, alpha) -> scalar (real)
    - control_param_derivs: list of functions dc_k_dalpha_i(t, alpha) -> array(len(alpha))
       Alternatively, provide a single function that returns an (n_controls, n_params) array.
    The augmented state y contains:
       U (N,N) and for each parameter i: dU/dalpha_i (N,N)
    """
    N = H0.shape[0]
    n_controls = len(H_controls) # number of control hamiltonians

    def ode_rhs(t, y, alpha):
        C = alpha.size
        n_blocks = 1 + C
        mats = unpack_complex_vector(y, n_blocks, N)
        U = mats[0]
        dUs = mats[1:] # shape (C, N, N)

        # Built H(t)
        H = H0.copy()
        c = control_funcs(t, alpha, control_extra_params=control_extra_params)
        for k in range(n_controls):
            H = H + c[k] * H_controls[k]

        # get derivatives of H w.r.t. alpha
        dc = control_param_derivs(t, alpha, control_extra_params=control_extra_params) # shape (n_controls, C)
        dH_dalpha = np.zeros((C, N, N), dtype=complex)
        for i in range(C):
            for k in range(n_controls):
                dH_dalpha[i] += dc[k,i] * H_controls[k]

        # using coefficient matrix, compute RHS of ODE 
        dU_dt = -1j * (H @ U)
        ddU_dt = np.zeros((C, N, N), dtype=complex)
        for i in range(C):
            ddU_dt[i] = -1j * (H @ dUs[i]) - 1j * (dH_dalpha[i] @ U)

        # pack back into real vector
        mats_out = np.vstack([dU_dt[np.newaxis, ...], ddU_dt])
        return pack_complex_list(mats_out)
    
    return ode_rhs

def make_unitary_hessian_ode_rhs(H0, H_controls, control_funcs, control_param_derivs, control_param_hessians, control_extra_params=None):
    N = H0.shape[0]
    n_controls = len(H_controls)

    def ode_rhs(t, y, alpha):
        C = alpha.size
        n_blocks = 1 + C + C*C
        mats = unpack_complex_vector(y, n_blocks, N)
        U = mats[0]
        dUs = mats[1:C+1] # shape (C, N, N)
        ddUs = mats[C+1:] # shape (C*C, N, N)

        # Build H(t)
        H = H0.copy()
        c = control_funcs(t, alpha, control_extra_params=control_extra_params)
        for k in range(n_controls):
            H = H + c[k] * H_controls[k]
        
        # get derivatives of H w.r.t. alpha
        dc = control_param_derivs(t, alpha, control_extra_params=control_extra_params) # shape (n_controls, C)
        dH_dalpha = np.zeros((C, N, N), dtype=complex)
        for i in range(C):
            for k in range(n_controls):
                dH_dalpha[i] += dc[k,i] * H_controls[k]

        # get second derivaties of H w.r.t. alpha
        ddc = control_param_hessians(t, alpha, control_extra_params=control_extra_params) # shape (n_controls, C, C)
        ddH_ddalpha = np.zeros((C*C, N, N), dtype=complex)
        for i in range(C):
            for j in range(C):
                for k in range(n_controls):
                    ddH_ddalpha[C*i+j] += ddc[k,i,j] * H_controls[k]

        # using hamiltonian derivatives, compute RHS of ODE
        dU_dt = -1j * (H @ U)

        ddU_dt = np.zeros((C, N, N), dtype=complex)
        for i in range(C):
            ddU_dt[i] = -1j * (H @ dUs[i]) - 1j * (dH_dalpha[i] @ U)

        dddU_dt = np.zeros((C*C, N, N), dtype=complex)
        for i in range(C):
            for j in range(C):
                dddU_dt[C*i+j] = -1j * (ddH_ddalpha[C*i+j] @ U + 
                                        dH_dalpha[j] @ dUs[i] + 
                                        dH_dalpha[i] @ dUs[j] + 
                                        H @ ddUs[C*i+j])
        
        # pack back into real vector
        mats_out = np.vstack([dU_dt[np.newaxis, ...], ddU_dt, dddU_dt])
        return pack_complex_list(mats_out)

    return ode_rhs

def make_robust_ode_rhs(H0, H_controls, control_funcs, control_param_derivs, control_param_hessians, control_param_third_orders, control_extra_params=None):
    N = H0.shape[0]
    n_controls = len(H_controls)

    def ode_rhs(t, y, alpha):
        C = alpha.size
        n_blocks = 1 + 3*C
        mats = unpack_complex_vector(y, n_blocks, N)
        U = mats[0]
        dUs = mats[1:C+1] # shape (C, N, N)
        ddUs = mats[C+1:2*C+1] # shape (C, N, N)
        dddUs = mats[2*C+1:]

        H = H0.copy()
        c = control_funcs(t, alpha, control_extra_params)
        for k in range(n_controls):
            H = H + c[k]*H_controls[k]
        
        dc = control_param_derivs(t, alpha, control_extra_params=control_extra_params) # shape (n_controls, C)
        dH_dalpha = np.zeros((C, N, N), dtype=complex)
        for i in range(C):
            for k in range(n_controls):
                dH_dalpha[i] += dc[k,i] * H_controls[k]

        ddc = control_param_hessians(t, alpha, control_extra_params=control_extra_params)[:, :, 0] # shape (n_controls, C)
        ddH_ddalpha = np.zeros((C, N, N), dtype=complex)
        for i in range(C):
            for k in range(n_controls):
                ddH_ddalpha[i] += ddc[k,i] * H_controls[k]
        
        dddH_dddalpha = np.zeros((C, N, N), dtype=complex)

        dU_dt = -1j * (H @ U)

        ddU_dt = np.zeros((C, N, N), dtype=complex)
        for i in range(C):
            ddU_dt[i] = -1j * (H @ dUs[i]) - 1j * (dH_dalpha[i] @ U)
        
        dddU_dt = np.zeros((C, N, N), dtype=complex)
        for i in range(C):
            dddU_dt[i] = -1j * (ddH_ddalpha[i] @ U + 
                                dH_dalpha[i] @ dUs[0] +
                                dH_dalpha[0] @ dUs[i] + 
                                H @ ddUs[i])

        ddddU_dt = np.zeros((C, N, N), dtype=complex)
        for i in range(C):
            ddddU_dt[i] = -1j * (dddH_dddalpha[i] @ U +
                                 2 * ddH_ddalpha[i] @ dUs[0] +
                                 ddH_ddalpha[0] @ dUs[i] +
                                 dH_dalpha[i] @ ddUs[0] +
                                 2 * dH_dalpha[0] @ ddUs[i] +
                                 H @ dddUs[i])
            
        mats_out = np.vstack([dU_dt[np.newaxis, ...], ddU_dt, dddU_dt, ddddU_dt])
        return pack_complex_list(mats_out)
    
    return ode_rhs

def goat_infidelity_and_grad(U_target, U, dU_dalpha_list, single_qubit_phase, single_qubit_phase_weights):
    N = U.shape[0]

    single_qubit_phase_weights = np.array(single_qubit_phase_weights)
    U_sqphase = np.diag(np.exp(1j * single_qubit_phase * single_qubit_phase_weights)) 
    dU_sqphase = np.diag(1j * single_qubit_phase_weights * np.exp(1j * single_qubit_phase * single_qubit_phase_weights))

    O = np.trace(U_target.conj().T @ U_sqphase @ U)
    infidelity = 1 - (np.abs(O)/N) 

    C = dU_dalpha_list.shape[0]
    grad = np.zeros(C + 1, dtype=float)

    for i in range(C):
        dO = np.trace(U_target.conj() @ U_sqphase @ dU_dalpha_list[i])
        grad[i] = -(1/N) * np.real( (np.conj(O) / np.abs(O)) * dO)
    dO = np.trace(U_target.conj() @ dU_sqphase @ U)
    grad[-1] = -(1/N) * np.real( (np.conj(O) / np.abs(O)) * dO)

    return infidelity, grad

def goat_full_infidelity_and_grad(U_target, U, dU_dalpha_list, single_qubit_phase, single_qubit_phase_weights):
    N = 4

    U_full = np.zeros((N, N), dtype=complex)
    U_full[0, 0] = 1
    U_full[1::2, 1::2] = U[0:2, 0:2]
    U_full[2:, 2:] = U[0:2, 0:2]

    dU_dalpha_list_full = np.zeros((dU_dalpha_list.shape[0], N, N), dtype=complex)
    dU_dalpha_list_full[:, 1::2, 1::2] = dU_dalpha_list[:, 0:2, 0:2]
    dU_dalpha_list_full[:, 2:, 2:] = dU_dalpha_list[:, 0:2, 0:2]

    single_qubit_phase_weights = np.array([0, single_qubit_phase_weights[0], single_qubit_phase_weights[0], single_qubit_phase_weights[1]])
    U_sqphase = np.diag(np.exp(1j * single_qubit_phase * single_qubit_phase_weights)) 
    dU_sqphase = np.diag(1j * single_qubit_phase_weights * np.exp(1j * single_qubit_phase * single_qubit_phase_weights))

    O = np.trace(U_target.conj().T @ U_sqphase @ U_full)
    infidelity = 1 - (np.abs(O)/N) 

    C = dU_dalpha_list_full.shape[0]
    grad = np.zeros(C + 1, dtype=float)

    for i in range(C):
        dO = np.trace(U_target.conj() @ U_sqphase @ dU_dalpha_list_full[i])
        grad[i] = -(1/N) * np.real( (np.conj(O) / np.abs(O)) * dO)
    dO = np.trace(U_target.conj() @ dU_sqphase @ U_full)
    grad[-1] = -(1/N) * np.real( (np.conj(O) / np.abs(O)) * dO)

    return infidelity, grad

def to_infidelity_and_grad(U_target, U, dU_dalpha_list, single_qubit_phase, single_qubit_phase_weights):
    """This function only works for time-optimal gates and using the truncated target unitary"""
    single_qubit_phase_weights = np.array(single_qubit_phase_weights)
    U_sqphase = np.diag(np.exp(1j * single_qubit_phase * single_qubit_phase_weights)) 
    dU_sqphase = np.diag(1j * single_qubit_phase_weights * np.exp(1j * single_qubit_phase * single_qubit_phase_weights))

    U_ops = U_target.conj().T @ U_sqphase @ U
    a01 = U_ops[0,0]
    a11 = U_ops[1,1]
    tr_sum = 1 + 2*a01 + a11
    fidelity = (1/20) * ( (np.abs(tr_sum) ** 2) + 1 + 2*(np.abs(a01) ** 2) + (np.abs(a11) ** 2) )

    C = dU_dalpha_list.shape[0]
    grad = np.zeros(C + 1, dtype=float)
    for i in range(C):
        dU_ops = U_target.conj().T @ U_sqphase @ dU_dalpha_list[i]
        da01 = dU_ops[0,0]
        da11 = dU_ops[1,1]
        grad[i] += (1/10) * np.real(np.conj(tr_sum) * (2*da01 + da11)) 
        grad[i] += (1/10) * (2*np.real(np.conj(a01) * da01) + np.real(np.conj(a11) * da11))
    dU_phase_ops = U_target.conj().T @ dU_sqphase @ U
    dphase_a01 = dU_phase_ops[0,0]
    dphase_a11 = dU_phase_ops[1,1]
    grad[-1] += (1/10) * (np.real(np.conj(tr_sum) * (2*dphase_a01 + dphase_a11)))
    grad[-1] += (1/10) * (2*np.real(np.conj(a01) * dphase_a01) + np.real(np.conj(a11) * dphase_a11))

    return 1 - fidelity, -grad

def to_hessian(U_target, U, dU, ddU, single_qubit_phase, single_qubit_phase_weights):
    C = dU.shape[0]
    single_qubit_phase_weights = np.array(single_qubit_phase_weights)

    U_sqphase = np.diag(np.exp(1j * single_qubit_phase * single_qubit_phase_weights)) 
    dU_sqphase = np.diag(1j * single_qubit_phase_weights * np.exp(1j * single_qubit_phase * single_qubit_phase_weights))
    ddU_sqphase = np.diag(-(single_qubit_phase_weights ** 2) * np.exp(1j * single_qubit_phase * single_qubit_phase_weights))

    U_ops = U_target.conj().T @ U_sqphase @ U
    a01 = U_ops[0,0]
    a11 = U_ops[1,1]
    tr_sum = 1 + 2*a01 + a11

    da01s = np.zeros(C+1, dtype=complex)
    da11s = np.zeros(C+1, dtype=complex)
    for i in range(C+1):
        if i == C:
            dsqU_ops = U_target.conj().T @ dU_sqphase @ U
            da01s[i] = dsqU_ops[0,0]
            da11s[i] = dsqU_ops[1,1]
        else:
            dU_ops = U_target.conj().T @ U_sqphase @ dU[i]
            da01s[i] = dU_ops[0,0]
            da11s[i] = dU_ops[1,1]

    hessian = np.zeros((C+1, C+1), dtype=float)
    for i in range(C+1):
        for j in range(C+1):
            if i == C and j == C:
                ddU_ops = U_target.conj().T @ ddU_sqphase @ U 
            elif i == C:
                ddU_ops = U_target.conj().T @ dU_sqphase @ dU[j]
            elif j == C:
                ddU_ops = U_target.conj().T @ dU_sqphase @ dU[i]
            else:
                ddU_ops = U_target.conj().T @ U_sqphase @ ddU[C*i+j]

            dda01 = ddU_ops[0,0]
            dda11 = ddU_ops[1,1]

            hessian[i,j] = (1/10) * np.real( (2*dda01 + dda11) * np.conj(tr_sum) + (2*da01s[i] + da11s[i]) * np.conj(2*da01s[j] + da11s[j])
                                            + 2*dda01*np.conj(a01) + 2*da01s[i]*np.conj(da01s[j]) + dda11*np.conj(a11) + da11s[i]*np.conj(da11s[j]) )
    return hessian

def to_infidelity_hessian(
        H0, 
        H_controls, 
        control_funcs, 
        control_param_derivs, 
        control_param_hessians,
        alpha,
        U_target,
        t_span,
        U_truncator,
        single_qubit_phase_weights,
        control_extra_params=None,
        ode_rtol=1e-7,
        ode_atol=1e-9,
):
    N = H0.shape[0]
    C = alpha.size - 1
    n_blocks = 1 + C + C*C
    mats0 = np.zeros((n_blocks, N, N), dtype=complex)
    mats0[0] = np.eye(N, dtype=complex)
    y0 = pack_complex_list(mats0)

    hessian_ode_rhs_func = make_unitary_hessian_ode_rhs(H0, 
                                                        H_controls, 
                                                        control_funcs, 
                                                        control_param_derivs, 
                                                        control_param_hessians, 
                                                        control_extra_params=control_extra_params)

    sol = solve_ivp(
        fun = lambda t, y: hessian_ode_rhs_func(t, y, alpha[:-1]), # exclude single qubit phase
        t_span=t_span,
        y0=y0,
        rtol=ode_rtol,
        atol=ode_atol,
        method="DOP853"
    )

    yf = sol.y[:, -1]

    mats_final = unpack_complex_vector(yf, n_blocks, N)
    if U_truncator is not None:
        U = U_truncator(mats_final[0])
        dU = U_truncator(mats_final[1:C+1])
        ddU = U_truncator(mats_final[C+1:])
    else:
        U = mats_final[0]
        dU = mats_final[1:C+1]
        ddU = mats_final[C+1:]

    # compute fidelity hessian

    return to_hessian(U_target, U, dU, ddU, alpha[-1], single_qubit_phase_weights)

def to_robustness_cost_and_grad(U_target, U, dU, ddU, dddU, single_qubit_phase, single_qubit_phase_weights):
    N = U.shape[0]
    C = dU.shape[0]
    single_qubit_phase_weights = np.array(single_qubit_phase_weights)

    U_sqphase = np.diag(np.exp(1j * single_qubit_phase * single_qubit_phase_weights)) 
    dU_sqphase = np.diag(1j * single_qubit_phase_weights * np.exp(1j * single_qubit_phase * single_qubit_phase_weights))
    ddU_sqphase = np.diag(-(single_qubit_phase_weights ** 2) * np.exp(1j * single_qubit_phase * single_qubit_phase_weights))
    dddU_sqphase = np.diag(-1j * (single_qubit_phase_weights ** 3) * np.exp(1j * single_qubit_phase * single_qubit_phase_weights))

    A = U_target.conj().T @ U_sqphase @ U
    dAs = np.zeros((C+1, N, N), dtype=complex)
    ddAs = np.zeros((C+1, N, N), dtype=complex)
    dddAs = np.zeros((C+1, N, N), dtype=complex)
    for i in range(C):
        dAs[i] = U_target.conj().T @ U_sqphase @ dU[i]
        ddAs[i] = U_target.conj().T @ U_sqphase @ ddU[i]
        dddAs[i] = U_target.conj().T @ U_sqphase @ dddU[i]
    dAs[-1] = U_target.conj().T @ dU_sqphase @ U
    ddAs[-1] = U_target.conj().T @ ddU_sqphase @ U
    dddAs[-1] = U_target.conj().T @ dddU_sqphase @ U

    tr_A = 1 + 2*A[0,0] + A[1,1]
    tr_dAs = np.zeros(C+1, dtype=complex)
    tr_ddAs = np.zeros(C+1, dtype=complex)
    tr_dddAs = np.zeros(C+1, dtype=complex)
    for i in range(C+1):
        tr_dAs[i] = 2*dAs[i][0,0] + dAs[i][1,1]
        tr_ddAs[i] = 2*ddAs[i][0,0] + ddAs[i][1,1]
        tr_dddAs[i] = 2*dddAs[i][0,0] + dddAs[i][1,1]

    robustness_cost = (1/10) * np.real(tr_ddAs[0] * np.conj(tr_A) + tr_dAs[0] * np.conj(tr_dAs[0])
                                       + 2 * ddAs[0][0,0] * np.conj(A[0,0]) + ddAs[0][1,1] * np.conj(A[1,1])
                                       + 2 * dAs[0][0,0] * np.conj(dAs[0][0,0]) + dAs[0][1,1] * np.conj(dAs[0][1,1]) )

    robustness_grad = np.zeros(C)
    for i in range(1,C+1):
        robustness_grad[i-1] = (1/10) * np.real(tr_dddAs[i] * np.conj(tr_A) + 2 * tr_ddAs[i] * np.conj(tr_dAs[0]) + tr_ddAs[0] * np.conj(tr_dAs[i])
                                              + 2 * dddAs[i][0,0] * np.conj(A[0,0]) + dddAs[i][1,1] * np.conj(A[1,1])
                                              + 4 * ddAs[i][0,0] * np.conj(dAs[0][0,0]) + 2 * ddAs[i][1,1] * np.conj(dAs[0][1,1]) 
                                              + 2 * ddAs[0][0,0] * np.conj(dAs[i][0,0]) + ddAs[0][1,1] * np.conj(dAs[i][1,1]) )

    return robustness_cost, robustness_grad

def run_goat_optimization(
        H0,
        H_controls,
        control_funcs,
        control_param_derivs,
        alpha0,
        U_target,
        t_span,
        U_truncator,
        single_qubit_phase_weights,
        optimization_method="BFGS",
        fidelity_func_name="GOAT",
        alpha_bounds=None,
        constraints=None,
        control_extra_params=None,
        cost_extra_params=None,
        optimizer_opts=None,
        ode_rtol=1e-7,
        ode_atol=1e-9,
        calculate_numerical_hessian=False,
        callback=None
):
    N = H0.shape[0]
    C = alpha0.size - 1

    ode_rhs_func = make_ode_rhs(H0, H_controls, control_funcs, control_param_derivs, control_extra_params=control_extra_params)

    if fidelity_func_name == "TO":
        fidelity_func = to_infidelity_and_grad
    elif fidelity_func_name == "GOAT_full":
        fidelity_func = goat_full_infidelity_and_grad
    else:
        fidelity_func = goat_infidelity_and_grad 

    def eval_cost_and_grad(alpha, single_qubit_phase):
        n_blocks = 1 + C
        mats0 = np.zeros((n_blocks, N, N), dtype=complex)
        mats0[0] = np.eye(N, dtype=complex)
        y0 = pack_complex_list(mats0)

        sol = solve_ivp(
            fun=lambda t, y: ode_rhs_func(t, y, alpha),
            t_span=t_span,
            y0=y0,
            rtol=ode_rtol,
            atol=ode_atol,
            method="DOP853"
        )

        yf = sol.y[:, -1]
        mats_final = unpack_complex_vector(yf, n_blocks, N)
        if U_truncator is not None:
            U_final = U_truncator(mats_final[0])
            dU_final = U_truncator(mats_final[1:])
        else:
            U_final = mats_final[0]
            dU_final = mats_final[1:]

        cost, grad = fidelity_func(U_target, U_final, dU_final, single_qubit_phase, single_qubit_phase_weights)

        if cost_extra_params is not None:
            cost += cost_extra_params["bw_lambda"] * np.sum(cost_extra_params["bw_k"] * np.abs(alpha))
            grad[:-1] += cost_extra_params["bw_lambda"] * cost_extra_params["bw_k"] * np.sign(alpha)

        return cost, grad

    if optimization_method == "trust-constr":
        if optimizer_opts is None:
            optimizer_opts = {"maxiter": 1000, "verbose": 3, "gtol": 1e-10, "xtol": 1e-10, 
                            "initial_barrier_parameter": 1e-3, "initial_barrier_tolerance": 1e-3, "barrier_tol": 1e-6} # for trust-constr

        res = minimize(
            fun=lambda a: eval_cost_and_grad(a[:-1], a[-1])[0],
            x0=alpha0,
            jac=lambda a: eval_cost_and_grad(a[:-1], a[-1])[1],
            hess=BFGS(),
            bounds=alpha_bounds,
            constraints=constraints,
            method="trust-constr",
            options=optimizer_opts,
            callback=callback
        )
    else:
        if optimizer_opts is None:
            # optimizer_opts = {"maxiter": 200, "disp": True, "ftol": 1e-15, "gtol": 1e-15} # for L-BFGS-B
            optimizer_opts = {"maxiter": 200, "disp": True, "ftol": 1e-15, "gtol": 1e-15} # for BFGS

        res = minimize(
            fun=lambda a: eval_cost_and_grad(a[:-1], a[-1])[0],
            x0=alpha0,
            jac=lambda a: eval_cost_and_grad(a[:-1], a[-1])[1],
            bounds=alpha_bounds,
            constraints=constraints,
            method="BFGS",
            options=optimizer_opts,
            callback=callback
        )
    
    if calculate_numerical_hessian:
        numerical_hessian = np.zeros((C+1, C+1), dtype=complex)
        h = 1e-8
        for i in range(C+1):
            alpha_forward = res.x.copy()
            alpha_backward = res.x.copy()

            alpha_forward[i] += h
            alpha_backward[i] -= h

            grad_forward = eval_cost_and_grad(alpha_forward[:-1], alpha_forward[-1])[1]
            grad_backward = eval_cost_and_grad(alpha_backward[:-1], alpha_backward[-1])[1]

            numerical_hessian[:, i] = (grad_forward - grad_backward) / (2*h)
        numerical_hessian = -0.5 * np.real(numerical_hessian + numerical_hessian.conj().T)
    else:
        numerical_hessian = None

    return res, numerical_hessian

def run_goat_optimization_robust(
    H0,
    H_controls,
    control_funcs,
    control_param_derivs,
    control_param_hessians,
    control_param_third_orders,
    alpha0,
    hamiltonian_params,
    U_target,
    t_span,
    U_truncator,
    single_qubit_phase_weights,
    control_extra_params=None,
    cost_extra_params=None,
    ode_rtol=1e-7,
    ode_atol=1e-9,
    optimizer_opts=None,
    callback=None,
):
    N = H0.shape[0]
    R = hamiltonian_params.size
    C = alpha0.size - 1 + R

    robust_ode_rhs_func = make_robust_ode_rhs(H0, 
                                              H_controls,
                                              control_funcs,
                                              control_param_derivs,
                                              control_param_hessians,
                                              control_param_third_orders,
                                              control_extra_params=control_extra_params)
    
    fidelity_func = to_infidelity_and_grad
    robustness_cost_func = to_robustness_cost_and_grad

    def eval_cost_and_grad(alpha, single_qubit_gate_phase):
        n_blocks = 1 + 3*C
        mats0 = np.zeros((n_blocks, N, N), dtype=complex)
        mats0[0] = np.eye(N, dtype=complex)
        y0 = pack_complex_list(mats0)

        all_params = np.concatenate([hamiltonian_params, alpha])

        sol = solve_ivp(
            fun=lambda t, y: robust_ode_rhs_func(t, y, all_params),
            t_span=t_span,
            y0=y0,
            rtol=ode_rtol,
            atol=ode_atol,
            method="DOP853"
        )

        yf = sol.y[:, -1]
        mats_final = unpack_complex_vector(yf, n_blocks, N)
        U = U_truncator(mats_final[0])
        dU = U_truncator(mats_final[1:C+1])
        ddU = U_truncator(mats_final[C+1:2*C+1])
        dddU = U_truncator(mats_final[2*C+1:])

        cost, grad = fidelity_func(U_target, U, dU, single_qubit_gate_phase, single_qubit_phase_weights)
        robustness_cost, robustness_grad = robustness_cost_func(U_target, U, dU, ddU, dddU, single_qubit_gate_phase, single_qubit_phase_weights)

        # return cost, grad[R:]
        # return cost - robustness_cost, grad[R:] - robustness_grad
        print(f"Cost: {cost}, Robustness cost: {robustness_cost}")

        # return (cost + cost_extra_params["robustness_lambda"] * np.abs(robustness_cost), 
        #         grad[R:] + cost_extra_params["robustness_lambda"] * np.sign(robustness_cost) * robustness_grad,
        #         cost,
        #         robustness_cost)
        
        # fid_scale = max(cost, 1e-9)
        # rob_scale = max(robustness_cost, 1e-9)

        return (cost + cost_extra_params["robustness_lambda"] * (robustness_cost**2),
                grad[R:] + cost_extra_params["robustness_lambda"] * 2 * robustness_cost * robustness_grad,
                cost,
                robustness_cost)

        # return ( (cost / fid_scale) + cost_extra_params["robustness_lambda"] * ( (robustness_cost / rob_scale) ** 2),
        #         (grad[R:] / fid_scale) + cost_extra_params["robustness_lambda"] * 2 * (robustness_cost / rob_scale) * (robustness_grad / rob_scale),
        #         cost,
                # robustness_cost)

    if optimizer_opts is None: 
        optimizer_opts = {"maxiter": 1000, "disp": True, "ftol": 1e-15, "gtol": 1e-15} # for L-BFGS-B
        # optimizer_opts = {"maxiter": 1000, "disp": True, "gtol": 1e-12} # for BFGS

    res = minimize(
        fun=lambda a: eval_cost_and_grad(a[:-1], a[-1])[0],
        x0=alpha0,
        jac=lambda a: eval_cost_and_grad(a[:-1], a[-1])[1],
        method="L-BFGS-B",
        options=optimizer_opts,
        callback=lambda x: callback(x, eval_cost_and_grad)
    )

    return res