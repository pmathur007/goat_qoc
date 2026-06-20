from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from scipy.integrate import solve_ivp

from utilities import DerivativeRequest, DerivativeData

class ControlSegment(ABC):
    """Abstract base class for control segments. All control segment subclasses must define the n_params member variable and the get_unitary_and_derivs function"""

    """Enforce that all control segments define the n_params variable"""
    @property
    def n_params(self):
        return self._n_params

    @n_params.setter
    def n_params(self, value):
        self._n_params = value

    @abstractmethod
    def get_unitary_and_derivs(self, params: np.ndarray, requested_derivs: DerivativeRequest) -> DerivativeData:
        """Computes the segment's unitary and its derivatives with respect to control parameters.

        Args:
            params (np.ndarray): A list of length self.n_params with the parameters that should be used to compute the segment's unitary and derivatives.
            requested_derivs (DerivativeRequest): Any requested higher-order derivatives.

        Returns:
            DerivativeData: The segment's unitary, all its first order derivatives, and any second and third order derivatives requested in order2_indices and order3_indices.
        """

class AnalogControlSegment(ControlSegment):
    def __init__(self, n_params: int, 
                 H_0: np.ndarray, 
                 H_controls: list[np.ndarray], 
                 controls: Callable[[float, np.ndarray], np.ndarray], 
                 dcontrols: Callable[[float, np.ndarray], np.ndarray], 
                 t_span: tuple[int, int],
                 d2controls: Callable[[float, np.ndarray], np.ndarray] | None = None, 
                 d3controls: Callable[[float, np.ndarray], np.ndarray] | None = None,
                 U_truncator: Callable[[np.ndarray], np.ndarray] | None = None):
        """Initialize an AnalogControlSegment object.
        Args:
            n_params (int): The number of parameters used in this control segment.
            H_0 (np.ndarray): The drift Hamiltonian.
            H_controls (np.ndarray): A list of control Hamiltonians.
            controls (Callable[[float, np.ndarray], np.ndarray]): A function that evaluates control coeffiecients at a specified time and parameter array.
            dcontrols (Callable[[float, np.ndarray], np.ndarray]): A function that evaluates control coeffiecient derivatives at a specified time and parameter array.
            t_span (tuple[int, int]): The start and end time for the control segment's evolution.
            d2controls (Callable[[float, np.ndarray], np.ndarray], optional): A function that evaluates control coeffiecient 2nd-order derivatives at a specified time and parameter array.
            d3controls (Callable[[float, np.ndarray], np.ndarray], optional): A function that evaluates control coeffiecient 3rd-order derivatives at a specified time and parameter array.
            U_truncator (Callable[[np.ndarray], np.ndarray], optional): A function that truncates the evolved unitary and its derivatives before returning to the caller.
        """
        self.H_0 = H_0
        self.H_controls = H_controls
        
        self.controls = controls
        self.dcontrols = dcontrols
        self.d2controls = d2controls
        self.d3controls = d3controls

        self.t_span = t_span

        self.hilbert_dim = H_0.shape[0]
        self.n_params = n_params
        self.n_controls = len(H_controls)
        self.U_truncator = U_truncator

    @staticmethod
    def _pack_complex_matrices(mats: np.ndarray) -> np.ndarray:
        """Utility function for flattening a list of complex matrices into a 1D array for a numerical integration step.
        Args:
            mats (np.ndarray): a list of complex matrices to flatten into a 1D array
        Returns:
            np.ndarray: a flattened 1D array, whose first half consist of the real part of the complex matrices and second half consists of their imaginary parts.
        """
        if mats.ndim == 2:
            mats = mats[np.newaxis, ...]
        packed_vec = np.concatenate([mats.real.flatten(), mats.imag.flatten()])
        return packed_vec

    @staticmethod
    def _unpack_complex_matrices(vec: np.ndarray, n_blocks: int, mat_dim: int) -> np.ndarray:
        """Unpacks a flattened real array into a list of complex matrices.
        Args:
            vec (np.ndarray): The flattened array to be unwrapped into complex matrices.
            n_blocks (int): The number of matrices in the complex matrix list to be created.
            mat_dim (int): The dimension of each square complex matrix in the list to be created.
        Returns:
            np.ndarray: A list of complex matrices with shape (n_blocks, mat_dim, mat_dim).
        Raises:
            ValueError: If the length of the flat vector is not consistent with the request number of blocks and matrix dimension.
        """
        if vec.size != 2 * n_blocks * mat_dim * mat_dim:
            raise ValueError(
                f"Expected vec.size == {2 * n_blocks * mat_dim * mat_dim}, "
                f"got {vec.size}."
            )

        real_part = vec[:n_blocks * mat_dim * mat_dim].reshape((n_blocks, mat_dim, mat_dim))
        imag_part = vec[n_blocks * mat_dim * mat_dim:].reshape((n_blocks, mat_dim, mat_dim))
        return real_part + 1j * imag_part

    def _ode_rhs(self, t: int, y: np.ndarray, params: np.ndarray, requested_derivs: DerivativeRequest) -> np.ndarray:
        """Computes the RHS of the ODE used to compute U, dU, d2U, and d3U.

        Args:
            t (float): Time at which to evaluate the RHS.
            y (np.ndarray): Flattened vector of U, dU, d2U, and d3U matrices used to evaluate the RHS.
            params (np.ndarray): Parameters used to evaluate the controls cofficients of the Hamiltonian terms that appear on the RHS.
            requested_derivs (DerivativeRequest): Any requested higher-order derivatives. 

        Returns:
            np.ndarray: Flattened vector of the ODE RHS, consisting of dU_dt d(dU)_dt, d(d2U)_dt, and d(d3U)_dt.
        """
        o2is = requested_derivs.order2_indices
        o3is = requested_derivs.order3_indices
        o2l = requested_derivs.order2_lookup

        n_mats = 1 + self.n_params
        n_mats += 0 if o2is is None else len(o2is)
        n_mats += 0 if o3is is None  else len(o3is)
        mats = self._unpack_complex_matrices(y, n_mats, self.hilbert_dim)

        U = mats[0]
        dU = mats[1:1+self.n_params]

        # construct Hamiltonian from controls
        H = self.H_0.copy()
        c = self.controls(t, params)
        for k in range(self.n_controls):
            H = H + c[k] * self.H_controls[k]

        # compute derivatives of the Hamiltonian w.r.t. controls
        dc = self.dcontrols(t, params)
        dH = np.zeros((self.n_params, self.hilbert_dim, self.hilbert_dim), dtype=complex)
        for i in range(self.n_params):
            for k in range(self.n_controls):
                dH[i] += dc[k,i] * self.H_controls[k]

        # construct RHS of ODE for the unitary and its first-order derivatives
        dU_dt = -1j * (H @ U)
        ddU_dt = np.zeros((self.n_params, self.hilbert_dim, self.hilbert_dim), dtype=complex) 
        for i in range(self.n_params):
            ddU_dt[i] = -1j * (H @ dU[i]) - 1j * (dH[i] @ U)
        
        mats_out = np.vstack([dU_dt[np.newaxis, ...], ddU_dt])

        # construct RHS of ODE for the requested list of second-order derivatives of the unitary
        if o2is is not None:
            n_d2s = len(o2is) # number of 2nd-order derivatives requested
            d2U = mats[1+self.n_params:1+self.n_params+n_d2s] 

            # compute 2nd-order derivatives of the Hamiltonian w.r.t. controls
            d2c = self.d2controls(t, params)
            d2H = np.zeros((n_d2s, self.hilbert_dim, self.hilbert_dim), dtype=complex)
            for i in range(n_d2s):
                for k in range(self.n_controls):
                    d2H[i] += d2c[k, o2is[i][0], o2is[i][1]] * self.H_controls[k]

            # construct RHS of ODE for 2nd-order derivatives of the unitary 
            dd2U_dt = np.zeros((n_d2s, self.hilbert_dim, self.hilbert_dim), dtype=complex)
            for i in range(n_d2s):
                di = o2is[i][0]
                dj = o2is[i][1]
                dd2U_dt[i] = -1j * (d2H[i] @ U +
                                    dH[dj] @ dU[di] +
                                    dH[di] @ dU[dj] +
                                    H @ d2U[i])

            mats_out = np.vstack([mats_out, dd2U_dt]) 

        if o3is is not None:
            n_d3s = len(o3is) # number of 3rd-order derivatives requested
            d3U = mats[1+self.n_params+len(o2is):]

            # compute 3rd-order derivatives of the Hamiltonian w.r.t. controls
            d3c = self.d3controls(t, params)
            d3H = np.zeros((n_d3s, self.hilbert_dim, self.hilbert_dim), dtype=complex)
            for i in range(n_d3s):
                for k in range(self.n_controls):
                    d3H[i] += d3c[k, o3is[i][0], o3is[i][1], o3is[i][2]] * self.H_controls[k]

            # construct RHS of ODE for 3rd-order derivatives of the unitary 
            dd3U_dt = np.zeros((n_d3s, self.hilbert_dim, self.hilbert_dim), dtype=complex)
            for i in range(n_d3s):
                di = o3is[i][0]
                dj = o3is[i][1]
                dk = o3is[i][2]
                d2ij = o2l[(di, dj)] # lookup the indicies of the appropriate 2nd-order derivatives in the d2H and d2U arrays
                d2ik = o2l[(di, dk)]
                d2jk = o2l[(dj, dk)]

                dd3U_dt[i] = -1j * (d3H[i] @ U + 
                                    d2H[d2ij] @ dU[dk] +
                                    d2H[d2ik] @ dU[dj] +
                                    d2H[d2jk] @ dU[di] +
                                    dH[di] @ d2U[d2jk] +
                                    dH[dj] @ d2U[d2ik] +
                                    dH[dk] @ d2U[d2ij] +
                                    H @ d3U[i])

            mats_out = np.vstack([mats_out, dd3U_dt]) 

        return self._pack_complex_matrices(mats_out)

    def get_unitary_and_derivs(self, params: np.ndarray, requested_derivs: DerivativeRequest) -> DerivativeData: 
        """Compute the unitary and its derivatives by constructing and integrating and ODE.

        Args:
            params (np.ndarray): A list of length self.n_params with the parameters that should be used to compute the segment's unitary and derivatives.
            requested_derivs (DerivativeRequest): Any requested higher-order derivatives.

        Returns:
            DerivativeData: The segment's unitary, all its first order derivatives, and any second and third order derivatives requested in order2_indices and order3_indices.

        Raises:
            ValueError: If the length of params is not equal to self.n_params.
            AttributeError: If a list of order2_indices is requested, but self.d2controls is not set.
            AttributeError: If a list of order3_indices is requested, but self.d3controls or self.d2controls are not set. 
            ValueError: If a list of order3_indices is requested, but, order2_lookup is not passed.
        """
        if params.size != self.n_params:
            raise ValueError(f"Expected params.size == {self.n_params}, got {params.size}.")

        o2is = requested_derivs.order2_indices
        o3is = requested_derivs.order3_indices
        o2l = requested_derivs.order2_lookup
        if o2is is not None:
            if self.d2controls is None:
                raise AttributeError("Second-order derivatives requested, but d2controls not set.")
        if o3is is not None:
            if self.d3controls is None:
                raise AttributeError("Third-order derivatives requested, but d3controls not set.")
            if self.d2controls is None:
                raise AttributeError("Third-order derivatives requested, but d2controls not set.")
            if o2l is None:
                raise ValueError("Third-order derivatives requested, but order2_lookup not passed to self._ode_rhs")

        n_d2s = 0 if o2is is None else len(o2is) # number of 2nd-order derivatives requested
        n_d3s = 0 if o3is is None else len(o3is) # number of 3rd-order derivatives requested

        # build initial conditions for the numerical integration
        mats_0 = np.zeros((1 + self.n_params + n_d2s + n_d3s, self.hilbert_dim, self.hilbert_dim), dtype=complex)
        mats_0[0] = np.eye(self.hilbert_dim, dtype=complex)
        y_0 = self._pack_complex_matrices(mats_0)

        # numerically integrate to get the unitary and its derivatives implemented by params
        sol = solve_ivp(
            fun=lambda t, y: self._ode_rhs(t, y, params, requested_derivs),
            t_span=self.t_span,
            y0=y_0,
            rtol=1e-7,
            atol=1e-9,
            method="DOP853"
        )

        # extract matrices from the ODE solution
        y_f = sol.y[:, -1]
        mats_f = self._unpack_complex_matrices(y_f, 1 + self.n_params + n_d2s + n_d3s, self.hilbert_dim)

        U_f = mats_f[0]
        dU_f = mats_f[1:1+self.n_params]
        if self.U_truncator is not None:
            U_f = self.U_truncator(U_f)
            dU_f = self.U_truncator(dU_f)
        deriv_data = DerivativeData(U=U_f, dU=dU_f)

        if o2is is not None:
            deriv_data.d2U = mats_f[1 + self.n_params:1 + self.n_params + n_d2s]
            if self.U_truncator is not None:
                deriv_data.d2U = self.U_truncator(deriv_data.d2U)

        if o3is is not None:
            deriv_data.d3U = mats_f[1 + self.n_params + n_d2s:]
            if self.U_truncator is not None:
                deriv_data.d3U = self.U_truncator(deriv_data.d3U)

        return deriv_data

class SingleQubitPhaseSegment(ControlSegment):
    def __init__(self, phase_weights: list[int]):
        """Initialize a SingleQubitPhaseSegment object.

        Args:
            phase_weights (list[int]): Phase weights for each qubit in the single qubit phase gate.
        """
        self.n_params = 1
        self.phase_weights = np.array(phase_weights)
 
    def get_unitary_and_derivs(self, params: np.ndarray, requested_derivs: DerivativeRequest) -> DerivativeData:
        """Compute the single qubit gate unitary and its derivatives w.r.t. the gate phase.

        Args:
            params (np.ndarray): A list of length self.n_params with the parameters that should be used to compute the segment's unitary and derivatives.
            requested_derivs (DerivativeRequest): Any requested higher-order derivatives.

        Returns:
            DerivativeData: The segment's unitary, all its first order derivatives, and any second and third order derivatives requested in order2_indices and order3_indices.

        Raises:
            ValueError: If the length of params is not equal to self.n_params.
        """
        if params.size != self.n_params:
            raise ValueError(f"Expected params.size == {self.n_params}, got {params.size}.")

        phase = params[0]

        # directly construct the single qubit gate unitary and its derivative
        U = np.diag(np.exp(1j * phase * self.phase_weights))
        dU = np.array([np.diag(1j * self.phase_weights * np.exp(1j * phase * self.phase_weights))])
        deriv_data = DerivativeData(U=U, dU=dU)

        # compute higher order derivatives as necessary
        if requested_derivs.order2_indices is not None:
            deriv_data.d2U = np.array([np.diag(-(self.phase_weights ** 2) * np.exp(1j * phase * self.phase_weights))])

        if requested_derivs.order3_indices is not None:
            deriv_data.d3U = np.array([np.diag(-1j * (self.phase_weights ** 3) * np.exp(1j * phase * self.phase_weights))])

        return deriv_data
