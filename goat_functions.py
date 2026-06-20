import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from utilities import DerivativeData

# References for fidelity metrics
# [1] S. Machnes, E. Ass´emat, D. Tannor, and F. K. Wilhelm, “Tunable, Flexible, and Effcient Optimization of Control Pulses for Practical Qubits”, Physical Review Letters 120, 150401 (2018).
# [2] S. Jandura and G. Pupillo, “Time-Optimal Two- and Three-Qubit Gates for Rydberg Atoms”, Quantum 6, arXiv:2202.00903 [quant-ph], 712 (2022).

class GOATFunction(ABC):
    """Abstract base class for all GOAT functions that consume unitaries and their derivatives."""

    """A flag indicating whether the function requires second-order derivatives of the gate unitary"""
    uses_order2: bool = False 

    """A flag indicating whether the function requires third-order derivatives of the gate unitary"""
    uses_order3: bool = False 

    """A flag indicating whether the function requires the gate paramters"""
    uses_params: bool = False 

    """If the function uses second-order derivatives of the gate unitary, order2_indices is a list of pairs of indices corresponding to
    the second derivatives that are required."""
    @property
    def order2_indices(self) -> list[tuple[int, int]] | None:
        return None

    """If the function uses third-order derivatives of the gate unitary, order2_indices is a list of pairs of indices corresponding to
    the third derivatives that are required."""
    @property
    def order3_indices(self) -> list[tuple[int, int, int]] | None:
        return None

    """A function called each time a GOATOptimizer requests for the gate unitary and its derivatives from its control segments. This can
    be overriden in the case that order2_indices and order3_indices depend on the optimization parameters, or other quantities only
    available at runtime.""" 
    def get_deriv_request(self, **kwargs) -> dict:
        return {
            "order2_indices": self.order2_indices,
            "order3_indices": self.order3_indices
        }

    @abstractmethod
    def __call__(self, deriv_data: DerivativeData, target_unitary: np.ndarray, **kwargs) -> Any:
        """Evaluate function.
        
        Args:
            deriv_data (DerivativeData): The unitary and its derivatives used for the cost computation.
            target_unitary (np.ndarray): The target unitary used for the cost computation.
        """

class GOATCostFunction(GOATFunction):
    """Abstract base class for all GOAT cost functions that are to be used during a gate optimization."""
    @abstractmethod
    def __call__(self, deriv_data: DerivativeData, target_unitary: np.ndarray, **kwargs) -> tuple[float, np.ndarray]:
        """Evaluate the cost and its gradient.
        
        Args:
            deriv_data (DerivativeData): The unitary and its derivatives used for the cost computation.
            target_unitary (np.ndarray): The target unitary used for the cost computation.

        Returns:
            tuple[float, np.ndarray]: The cost and its gradient.
        """

class HaarAverageInfidelity(GOATCostFunction):
    """Cost function for the Haar average infidelity metric and its gradient, defined in Ref [2]."""
    def __call__(self, deriv_data: DerivativeData, target_unitary: np.ndarray, **kwargs) -> tuple[float, np.ndarray]:
        U_ops = target_unitary.conj().T @ deriv_data.U
        a01 = U_ops[0,0]
        a11 = U_ops[1,1]
        tr_sum = 1 + 2*a01 + a11
        fidelity = (1/20) * ( (np.abs(tr_sum) ** 2) + 1 + 2*(np.abs(a01 ** 2)) + (np.abs(a11) ** 2) )

        n_params = deriv_data.dU.shape[0]
        grad = np.zeros(n_params, dtype=float)
        for i in range(n_params):
            dU_ops = target_unitary.conj().T @ deriv_data.dU[i]
            da01 = dU_ops[0,0]
            da11 = dU_ops[1,1]
            grad[i] += (1/10) * np.real(np.conj(tr_sum) * (2*da01 + da11))
            grad[i] += (1/10) * (2*np.real(np.conj(a01) * da01) + np.real(np.conj(a11) * da11))

        return 1 - fidelity, -grad

class HaarAverageInfidelityHessian(GOATFunction):
    """Compute the hessian of the Haar average infidelity metric."""
    uses_order2 = True # order2_indices is left as None to compute all second-order derivatives

    def __call__(self, deriv_data: DerivativeData, target_unitary: np.ndarray, **kwargs) -> np.ndarray:
        n_params = deriv_data.dU.shape[0]
        
        U_ops = target_unitary.conj().T @ deriv_data.U
        a01 = U_ops[0,0]
        a11 = U_ops[1,1]
        tr_sum = 1 + 2*a01 + a11

        da01 = np.zeros(n_params, dtype=complex)
        da11 = np.zeros(n_params, dtype=complex)
        for i in range(n_params):
            dU_ops = target_unitary.conj().T @ deriv_data.dU[i]
            da01[i] = dU_ops[0,0]
            da11[i] = dU_ops[1,1]
        
        hessian = np.zeros((n_params, n_params))
        for i in range(n_params):
            for j in range(n_params):
                d2U_ops = target_unitary.conj().T @ deriv_data.d2U[n_params*i + j]
                d2a01 = d2U_ops[0,0]
                d2a11 = d2U_ops[1,1]

                hessian[i,j] = (1/10) * np.real( (2*d2a01 + d2a11) * np.conj(tr_sum) + (2*da01[i] + da11[i]) * np.conj(2*da01[j] + da11[j])
                                                + 2*d2a01*np.conj(a01) + 2*da01[i]*np.conj(da01[j]) + d2a11*np.conj(a11) + da11[i]*np.conj(da11[j]) )
        return -hessian

class SUDistanceInfidelity(GOATCostFunction):
    """Compute the SU-distance infidelity metric and its gradient, defined in Ref [1]."""

    def __call__(self, deriv_data: DerivativeData, target_unitary: np.ndarray, **kwargs) -> tuple[float, np.ndarray]:
        N = deriv_data.U.shape[0]
        U_ops = target_unitary.conj().T @ deriv_data.U
        O = np.trace(U_ops)
        
        n_params = deriv_data.dU.shape[0]
        grad = np.zeros(n_params, dtype=float)
        for i in range(n_params):
            dU_ops = target_unitary.conj().T @ deriv_data.dU[i]
            grad[i] = -(1/N) * np.real( (np.conj(O) / np.abs(O)) * np.trace(dU_ops) )

        return 1 - (np.abs(O)/N), grad

class HaarInfidelityAndRobustness(GOATCostFunction):
    """Cost function for the Haar average infidelity metric and an additional amplitude-robustness term."""
    uses_order2 = True
    uses_order3 = True

    def __init__(self, robustness_lambda: float):
        """
        Args:
            robustness_lambda (float): The scaling factor to apply to the robustness terms of the cost function.
        """
        self.robustness_lambda = robustness_lambda

    def get_deriv_request(self, params):
        return {
            "order2_indices": [(0, i) for i in range(len(params))],
            "order3_indices": [(0, 0, i) for i in range(len(params))]
        } 

    def __call__(self, deriv_data: DerivativeData, target_unitary: np.ndarray, **kwargs) -> tuple[float, np.ndarray]:
        N = deriv_data.U.shape[0]
        n_params = deriv_data.dU.shape[0]

        A = target_unitary.conj().T @ deriv_data.U
        dA = np.zeros((n_params, N, N), dtype=complex)
        ddA = np.zeros((n_params, N, N), dtype=complex)
        dddA = np.zeros((n_params, N, N), dtype=complex)
        tr_A = 1 + 2*A[0,0] + A[1,1]
        tr_dA = np.zeros(n_params, dtype=complex)
        tr_ddA = np.zeros(n_params, dtype=complex)
        tr_dddA = np.zeros(n_params, dtype=complex)
        for i in range(n_params):
            dA[i] = target_unitary.conj().T @ deriv_data.dU[i]
            ddA[i] = target_unitary.conj().T @ deriv_data.d2U[i]
            dddA[i] = target_unitary.conj().T @ deriv_data.d3U[i]
            tr_dA[i] = 2*dA[i][0,0] + dA[i][1,1]
            tr_ddA[i] = 2*ddA[i][0,0] + ddA[i][1,1]
            tr_dddA[i] = 2*dddA[i][0,0] + dddA[i][1,1]

        cost = (1/20) * ( (np.abs(tr_A) ** 2) + 1 + 2*(np.abs(A[0,0]) ** 2) + (np.abs(A[1,1]) ** 2) ) 
        cost += self.robustness_lambda * (1/10) * np.real(tr_ddA[0] * np.conj(tr_A) + tr_dA[0] * np.conj(tr_dA[0])
                                                           + 2 * ddA[0][0,0] * np.conj(A[0,0]) + ddA[0][1,1] * np.conj(A[1,1])
                                                           + 2 * dA[0][0,0] * np.conj(dA[0][0,0]) + dA[0][1,1] * np.conj(dA[0][1,1]))
        
        grad = np.zeros(n_params)
        for i in range(1, n_params):
            grad[i] += (1/10) * np.real(tr_dA[i] * np.conj(tr_A) 
                                        + 2 * dA[i][0,0] * np.conj(A[0,0]) + dA[i][1,1] * np.conj(A[1,1])) 
            grad[i] += self.robustness_lambda * (1/10) * np.real(tr_dddA[i] * np.conj(tr_A) + 2 * tr_ddA[i] * np.conj(tr_dA[0]) + tr_ddA[0] * np.conj(tr_dA[i])
                                                                 + 2 * dddA[i][0,0] * np.conj(A[0,0]) + dddA[i][1,1] * np.conj(A[1,1])
                                                                 + 4 * ddA[i][0,0] * np.conj(dA[0][0,0]) + ddA[i][1,1] * np.conj(dA[0][1,1])
                                                                 + 2 * ddA[0][0,0] * np.conj(dA[i][0,0]) + ddA[0][1,1] * np.conj(dA[i][1,1]))

        return 1 - cost, -grad

class BandwidthCost(GOATCostFunction):
    """Cost function for imposing a controls bandwidth cost in addition to a base fidelity cost.
        It is assumed that the parameters are coefficients of Slepian functions, which are bandlimited.
    """

    uses_params = True

    def __init__(self, base_fidelity_func: GOATCostFunction,
                 param_range: tuple[int, int], 
                 bw_lambda: float, 
                 bw_cost_weights: np.ndarray):
        """
        Args:
            base_fidelity_func (GOATCostFunction): Base fidelity to which bandwidth costs should be added.
            param_range (tuple[int, int]): Subarray of params that should be used to compute bandwidth costs.
            bw_lambda (float): The scaling factor to apply to the bandwidth terms of the cost function.
            bw_cost_weights (np.ndarray): The cost weights to give to each parameter in the params array.
        """

        self.base_fidelity_func = base_fidelity_func
        self.param_range = param_range
        self.bw_lambda = bw_lambda
        self.bw_cost_weights = bw_cost_weights

    def __call__(self, deriv_data: DerivativeData, target_unitary: np.ndarray, **kwargs) -> tuple[float, np.ndarray]:
        params = kwargs["params"]
        cost, grad = self.base_fidelity_func(deriv_data, target_unitary)
        pstart = self.param_range[0]
        pend = self.param_range[1] + 1
        cost += self.bw_lambda * np.sum(self.bw_cost_weights * np.abs(params[pstart:pend]))
        grad[pstart:pend] += self.bw_lambda * self.bw_cost_weights * np.sign(params[pstart:pend])

        return cost, grad
