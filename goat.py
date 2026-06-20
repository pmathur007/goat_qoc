from typing import Any
import itertools
import numpy as np
from scipy.optimize import minimize

from control_segments import ControlSegment
from utilities import DerivativeData, DerivativeRequest, ThirdDerivativeType
from goat_functions import GOATFunction, GOATCostFunction

class GOATOptimizer:
    def __init__(self, hilbert_dim: int, control_segments: list[ControlSegment]):
        """Initialize a GOATOptimizer object.
        
        Args:
            hilbert_dim (int): Dimension of the hilbert space of the unitaries returned by each control segment.
            control_segments (list[ControlSegment]): List of control segments in the optimizer.
        """
        self.hilbert_dim = hilbert_dim
        self.control_segments = control_segments
        self.optimization_result = None # the optimization result after calling run_optimization will be stored here

        # preprocess array segments
        self.n_segments = len(control_segments)
        self.n_total_params = 0
        self.seg_indexes = [] # seg_indexes[i] is the segment to which the i'th parameter in the params array belongs to
        self.seg_ranges = [] # seg_ranges[i] is a tuple (start, end), the starting and ending indexes of the subarray of params that corresponds to segment i
        seg_start = 0
        for seg_i, seg in enumerate(control_segments):
            self.n_total_params += seg.n_params
            self.seg_indexes.extend([seg_i] * seg.n_params)
            self.seg_ranges.append((seg_start, seg_start + seg.n_params))
            seg_start = seg_start + seg.n_params

    def _get_unitary_and_order1_derivs(self, params: np.ndarray) -> DerivativeData:
        """Compute unitaries and their derivatives from each control segments using the input params array, then build the global unitary and its first-order derivatives.
        Args:
            params (np.ndarray): A list of length self.n_total_params that should be used to calculate the unitary and its derivatives.
        Returns:
            DerivativeData: The unitary and its first-order derivatives.
        """
        U = np.eye(self.hilbert_dim, dtype=complex)
        dU = np.tile(np.eye(self.hilbert_dim, dtype=complex), (params.size, 1, 1))

        # iterate through each control segment and build the global unitary and its derivatives
        for seg_i in range(len(self.control_segments)):
            param_start = self.seg_ranges[seg_i][0]
            param_end = self.seg_ranges[seg_i][1]

            # compute the unitary and its derivatives for this segment using the appropriate subarray of params
            unitary_and_derivs = self.control_segments[seg_i].get_unitary_and_derivs(params[param_start:param_end], DerivativeRequest())
            seg_U = unitary_and_derivs.U
            seg_dU = unitary_and_derivs.dU

            U = U @ seg_U # append current segment's unitary to the global unitary
            for i in range(params.size):
                # for each parameter, if it is in the current segment, append the appropriate derivative to the global derivative...
                if i >= param_start and i < param_end:
                    dU[i] = dU[i] @ seg_dU[i - param_start]
                # ...otherwise just append the unitary, since the derivative was appended by another segment
                else:
                    dU[i] = dU[i] @ seg_U

        return DerivativeData(U=U, dU=dU)

    def _get_unitary_and_order2_derivs(self, params: np.ndarray, order2_indices: list[tuple[int, int]] | None = None) -> DerivativeData:
        """Compute unitaries and their first- and second-derivatives from each control segments using the input params array, then build the global unitary and its first- and second-order derivatives.
        Args:
            params (np.ndarray): A list of length self.n_total_params that should be used to calculate the unitary and its derivatives. 
            order2_indices (list[tuple[int, int]], optional): A list of 2-tuples of indices from the params array that indicate which second-order derivatives d2U should be calculated. 
                                                              If left as None, then all (self.n_total_params ** 2) derivatives will be computed.
        Returns:
            DerivativeData: The unitary and its first- and second-order derivatives.
        """
        # if order2_indices is left as None, compute all (self.n_total_params ** 2) second-order derivatives
        if order2_indices is None:
            order2_indices = []
            for deriv in itertools.product(range(self.n_total_params), repeat=2):
                order2_indices.append(deriv)

        # local_flags[i] is True if the i'th pair of indices in order2_indices are both in the same segment, and False if the indices are in different segments.
        # If local_flags[i] is true, then the appropriate control segment needs to be told to compute the second-order derivative corresponding to the i'th pair of indices.
        # The local_indices list keeps track of this, where local_indices[i] is a list of all pairs of indices that fall in the same segment,
        # and thus require second-order derivative calculation by the i'th control segment.
        local_flags = []
        local_indices = [[] for _ in range(self.n_segments)] 

        for o2i in order2_indices: 
            seg1 = self.seg_indexes[o2i[0]]
            seg2 = self.seg_indexes[o2i[1]]
            if seg1 == seg2: # the two indices are in the same segment
                local_flags.append(True)
                param_offset = self.seg_ranges[seg1][0] # offset local_indices to index the control segment's params subarray instead of the global params array
                local_indices[seg1].append((o2i[0] - param_offset, o2i[1] - param_offset))
            else: # the two indices are in different segments
                local_flags.append(False)

        unitary_and_derivs = [] # list of DerivativeData for each segment
        for i in range(self.n_segments):
            param_start = self.seg_ranges[i][0]
            param_end = self.seg_ranges[i][1]

            # compute the unitary and its derivatives for this segment using the appropriate subarray of params
            if len(local_indices[i]) > 0:
                deriv_request = DerivativeRequest(order2_indices=local_indices[i])
            else:
                deriv_request = DerivativeRequest()
            unitary_and_derivs.append(self.control_segments[i].get_unitary_and_derivs(params[param_start:param_end], deriv_request))

        # build the global unitary from each segment's unitary 
        U = np.eye(self.hilbert_dim, dtype=complex)
        for i in range(self.n_segments):
            U = U @ unitary_and_derivs[i].U

        # build the global unitary first-order derivatives from each segment's first-order derivatives 
        dU = np.tile(np.eye(self.hilbert_dim, dtype=complex), (params.size, 1, 1))
        for i in range(self.n_segments):
            seg_U = unitary_and_derivs[i].U
            seg_dU = unitary_and_derivs[i].dU
            for pi in range(self.n_total_params):
                # for each parameter, if it is in the current segment, append the appropriate derivative to the global derivative...
                if self.seg_indexes[pi] == i:
                    dU[pi] = dU[pi] @ seg_dU[pi - self.seg_ranges[i][0]]
                # ...otherwise just append the unitary, since the derivative was appended by another segment
                else:
                    dU[pi] = dU[pi] @ seg_U

        # build the global unitary second_order derivatives from each segment's first- and second-order derivatives
        d2U = np.tile(np.eye(self.hilbert_dim, dtype=complex), (len(order2_indices), 1, 1))
        li_idx = [0 for _ in range(self.n_segments)] # for each segment, keep track of which local second-order needs to be consumed next
        for i in range(len(order2_indices)):
            o2i = order2_indices[i]
            if local_flags[i]: # the current derivative is w.r.t. two parameters in the same segment
                dseg = self.seg_indexes[o2i[0]]
                for seg in range(self.n_segments): # build the second derivative segment-by-segment
                    # if the current segment contains the current derivative's indices... 
                    if seg == dseg:
                        d2U[i] = d2U[i] @ unitary_and_derivs[seg].d2U[li_idx[seg]] # ...append the appropriate local second-order derivative to the global derivative...
                        li_idx[seg] += 1 # ...and increment the counter so that the next derivative is consumed next...
                    # ...otherwise just append the unitary, since the derivative was appended by another segment
                    else:
                        d2U[i] = d2U[i] @ unitary_and_derivs[seg].U
            else: # the current derivative is w.r.t. two parameters in different segments
                dseg1 = self.seg_indexes[o2i[0]]
                dseg2 = self.seg_indexes[o2i[1]]
                for seg in range(self.n_segments):
                    # if the current segment contains either of the current derivative's two indices, append the appropriate local derivatives to the global derivative...
                    if seg == dseg1:
                        d2U[i] = d2U[i] @ unitary_and_derivs[seg].dU[o2i[0] - self.seg_ranges[dseg1][0]]
                    elif seg == dseg2:
                        d2U[i] = d2U[i] @ unitary_and_derivs[seg].dU[o2i[1] - self.seg_ranges[dseg2][0]]
                    # ...otherwise just append the unitary, sionce the derivative was appended by another segment
                    else:
                        d2U[i] = d2U[i] @ unitary_and_derivs[seg].U

        return DerivativeData(U=U, dU=dU, d2U=d2U)

    def _get_unitary_and_order3_derivs(self, params: np.ndarray, 
                                   order2_indices: list[tuple[int, int]] | None = None, order3_indices: list[tuple[int, int, int]] | None = None
                                   ) -> DerivativeData:
        """Compute unitaries and their first-, second-, and third-order derivatives from each control segments using the input params array, then build the global unitary and its first-, second-, and third-order derivatives.

        Args:
            params (np.ndarray): A list of length self.n_total_params that should be used to calculate the unitary and its derivatives. 
            order2_indices (list[tuple[int, int]], optional): A list of 2-tuples of indices from the params array that indicate which second-order derivatives d2U should be calculated. 
                                                              If left as None, then all (self.n_total_params ** 2) derivatives will be computed.
            order3_indices (list[tuple[int, int, int]], optional): A list of 3-tuples of indices from the params array that indicate which third-order derivatives d3U should be calculated. 
                                                                   If left as None, then all (self.n_total_params ** 3) derivatives will be computed.
        Returns:
            DerivativeData: The unitary and its first-, second, and third-order derivatives.
        """
        # populate order 2 and order 3 indices if they are none
        if order2_indices is None:
            order2_indices = []
            for deriv in itertools.product(range(self.n_total_params), repeat=2):
                order2_indices.append(deriv)

        if order3_indices is None:
            order3_indices = []
            for deriv in itertools.product(range(self.n_total_params), repeat=3):
                order3_indices.append(deriv)

        # STAGE 1: Build metadata for second- and third-order derivatives.

        # o2_local_flags[i] is True if the i'th pair of indices in order2_indices are both in the same segment, and False if the indices are in different segments.
        # If o2_local_flags[i] is True, then the appropriate control segment needs to be told to compute the second-order derivative corresponding to the i'th pair of indices.
        # The o2_local_indices list keeps track of this, where o2_local_indices[i] is a list of all pairs of indices that fall in the same segment,
        # and thus require second-order derivative calculation by the i'th control segment.
        # o2_index_lookup is a list of dictionaries, with one dict per control segment. For the i'th control segment, the dict's keys are pairs of indices corresponding to derivatives
        # that need to be computed for that segment, and the corresponding values are the index of that pair in the o2_local_indices[i] array, which are used when computing third-order derivatives.
        o2_local_flags = []
        o2_local_indices = [[] for _ in range(self.n_segments)]
        o2_index_lookup = [{} for _ in range(self.n_segments)]

        # A helper function that adds a new second-order derivative index to the metadata described above. 
        def add_to_o2_indices(seg, new_index):
            param_offset = self.seg_ranges[seg][0] # offset local_indices to index the control segment's params subarray instead of the global params array
            local_index = (new_index[0] - param_offset, new_index[1] - param_offset) 
            if local_index not in o2_index_lookup[seg]: # ensure that derivatives aren't computed more than once.
                o2_index_lookup[seg][local_index] = len(o2_local_indices[seg])
                o2_local_indices[seg].append(local_index)

        # A helper function that uses o2_index_lookup to look up a global second-derivative pair's index in the appropriate array in o2_local_indices.
        def get_o2_deriv_index(seg, index):
            param_offset = self.seg_ranges[seg][0]
            local_index = (index[0] - param_offset, index[1] - param_offset) # use the index from the control segment's params subarray for lookup
            if local_index in o2_index_lookup[seg]:
                return o2_index_lookup[seg][local_index]
            else:
                return None

        # build metadata for second-order derivatives
        for o2i in order2_indices:
            seg1 = self.seg_indexes[o2i[0]]
            seg2 = self.seg_indexes[o2i[1]]
            if seg1 == seg2: # the two indices are in the same segment
                o2_local_flags.append(True)
                add_to_o2_indices(seg1, o2i)
            else:
                o2_local_flags.append(False)

        # o3_local_flags[i] takes on a value from the ThirdDerivativeType enum based on which of the i'th third-derivative triple's indices are in the same segment.
        # Based on the value of o3_local_flags[i], the appropriate control segments need to be told to compute second or third-order derivatives corresponding to the i'th triple of indices.
        # The o3_local_indices list keeps track of any third-derivatives that need to be computed by each control segment, where o3_local_indices[i] is a list of all triples of indices
        # that all fall in the same segment. Note that the code that builds third-order metadata also updates the second-order metadata appropriately with any local second-derivatives that need to be computed.
        o3_local_flags = []
        o3_local_indices = [[] for _ in range(self.n_segments)]

        # build metadata for third-order derivatives
        for o3i in order3_indices:
            seg1 = self.seg_indexes[o3i[0]]
            seg2 = self.seg_indexes[o3i[1]]
            seg3 = self.seg_indexes[o3i[2]]

            if seg1 == seg2 and seg1 == seg3: # all three parameters are in the same segment
                o3_local_flags.append(ThirdDerivativeType.ALL_LOCAL)
                param_offset = self.seg_ranges[seg1][0]
                add_to_o2_indices(seg1, (o3i[0], o3i[1]))
                add_to_o2_indices(seg1, (o3i[0], o3i[2]))
                add_to_o2_indices(seg1, (o3i[1], o3i[2]))
                o3_local_indices[seg1].append((o3i[0] - param_offset,
                                               o3i[1] - param_offset,
                                               o3i[2] - param_offset))
            elif seg1 == seg2: # only the first and second parameters are in the same segment
                o3_local_flags.append(ThirdDerivativeType.IJ_LOCAL)
                add_to_o2_indices(seg1, (o3i[0], o3i[1]))
            elif seg1 == seg3: # only the first and third parameters are in the same segment
                o3_local_flags.append(ThirdDerivativeType.IK_LOCAL)
                add_to_o2_indices(seg1, (o3i[0], o3i[2]))
            elif seg2 == seg3: # only the second and third parameters are in the same segment
                o3_local_flags.append(ThirdDerivativeType.JK_LOCAL)
                add_to_o2_indices(seg2, (o3i[1], o3i[2]))
            else: # all parameters are in different segments
                o3_local_flags.append(ThirdDerivativeType.ALL_DISTINCT)

        # STAGE 2: Compute local unitaries and their derivatives
        
        unitary_and_derivs = [] # list of DerivativeData for each segment
        for i in range(self.n_segments):
            param_start = self.seg_ranges[i][0]
            param_end = self.seg_ranges[i][1]

            # compute the unitary and its derivatives for this segment using the appropriate subarray of params
            if len(o3_local_indices[i]) > 0:
                deriv_request = DerivativeRequest(order2_indices=o2_local_indices[i],
                                                  order3_indices=o3_local_indices[i],
                                                  order2_lookup=o2_index_lookup[i])
            elif len(o2_local_indices[i]) > 0:
                deriv_request = DerivativeRequest(order2_indices=o2_local_indices[i])
            else:
                deriv_request = DerivativeRequest()

            unitary_and_derivs.append(self.control_segments[i].get_unitary_and_derivs(params[param_start:param_end], deriv_request))

        # STAGE 3: Build the global unitary and its derivatives from local unitaries and its derivatives

        # build the global unitary from each segment's unitary 
        U = np.eye(self.hilbert_dim, dtype=complex)
        for i in range(self.n_segments):
            U = U @ unitary_and_derivs[i].U

        # build the global unitary first-order derivatives from each segment's first-order derivatives 
        dU = np.tile(np.eye(self.hilbert_dim, dtype=complex), (params.size, 1, 1))
        for i in range(self.n_segments):
            seg_U = unitary_and_derivs[i].U
            seg_dU = unitary_and_derivs[i].dU
            for pi in range(self.n_total_params):
                # for each parameter, if it is in the current segment, append the appropriate derivative to the global derivative...
                if self.seg_indexes[pi] == i:
                    dU[pi] = dU[pi] @ seg_dU[pi - self.seg_ranges[i][0]]
                # ...otherwise just append the unitary, since the derivative was appended by another segment
                else:
                    dU[pi] = dU[pi] @ seg_U
        
        # build the global unitary second-order derivatives from each segment's first- and second-order derivatives 
        d2U = np.tile(np.eye(self.hilbert_dim, dtype=complex), (len(order2_indices), 1, 1))
        for i in range(len(order2_indices)):
            o2i = order2_indices[i]
            if o2_local_flags[i]: # the current derivative's two paramters are in the same segment
                dseg = self.seg_indexes[o2i[0]]
                for seg in range(self.n_segments): # build the second derivative segment-by-segment
                    # if the current segment contains the current derivative's indices... 
                    if seg == dseg:
                        deriv_index = get_o2_deriv_index(seg, o2i) # ...find the location of the appropriate local second-order derivative...
                        d2U[i] = d2U[i] @ unitary_and_derivs[seg].d2U[deriv_index] # ...and append it to the global derivative...
                    # ...otherwise just append the unitary, since the derivative was appended by another segment
                    else:
                        d2U[i] = d2U[i] @ unitary_and_derivs[seg].U
            else: # the current derivative's two parameters are in different segments
                dseg1 = self.seg_indexes[o2i[0]] 
                dseg2 = self.seg_indexes[o2i[1]] 
                for seg in range(self.n_segments):
                    # if the current segment contains either of the current derivative's two indices, append the appropriate local derivatives to the global derivative...
                    if seg == dseg1:
                        d2U[i] = d2U[i] @ unitary_and_derivs[seg].dU[o2i[0] - self.seg_ranges[dseg1][0]]
                    elif seg == dseg2:
                        d2U[i] = d2U[i] @ unitary_and_derivs[seg].dU[o2i[1] - self.seg_ranges[dseg2][0]]
                    # ...otherwise just append the unitary, sionce the derivative was appended by another segment
                    else:
                        d2U[i] = d2U[i] @ unitary_and_derivs[seg].U

        # build the global unitary third-order derivatives from each segment's first-, second-, and third-order derivatives
        d3U = np.tile(np.eye(self.hilbert_dim, dtype=complex), (len(order3_indices), 1, 1)) 
        li_idx = [0 for _ in range(self.n_segments)] # for each segment, keep track of which local third-order derivative needs to be consumed next
        for i in range(len(order3_indices)):
            o3i = order3_indices[i]
            if o3_local_flags[i] == ThirdDerivativeType.ALL_LOCAL: # the current derivative's parameters are all in the same segment
                dseg = self.seg_indexes[o3i[0]]
                for seg in range(self.n_segments): # build the third-derivative segment-by-segment
                    # if the current segment contains the current derivative's indices...
                    if seg == dseg:
                        d3U[i] = d3U[i] @ unitary_and_derivs[seg].d3U[li_idx[seg]] # ...append the appropriate local third-order derivative to the global derivative...
                        li_idx[seg] += 1 # ...and increment the counter so that the next derivative is consumed next...
                    # ...otherwise just append the unitary, since the derivative was appended by another segment
                    else:
                        d3U[i] = d3U[i] @ unitary_and_derivs[seg].U
            elif o3_local_flags[i] == ThirdDerivativeType.ALL_DISTINCT: # the current derivative's parameters are all in different segments
                dseg1 = self.seg_indexes[o3i[0]]
                dseg2 = self.seg_indexes[o3i[1]]
                dseg3 = self.seg_indexes[o3i[2]]
                for seg in range(self.n_segments):
                    # if the current segment contains any of the current derivative's three indices, append the appropriate local derivatives to the global derivative...
                    if seg == dseg1:
                        d3U[i] = d3U[i] @ unitary_and_derivs[seg].dU[o3i[0] - self.seg_ranges[seg][0]]
                    elif seg == dseg2:
                        d3U[i] = d3U[i] @ unitary_and_derivs[seg].dU[o3i[1] - self.seg_ranges[seg][0]]
                    elif seg == dseg3:
                        d3U[i] = d3U[i] @ unitary_and_derivs[seg].dU[o3i[2] - self.seg_ranges[seg][0]]
                    # ...otherwise just append the unitary, since the derivative was appended by another segment
                    else:
                        d3U[i] = d3U[i] @ unitary_and_derivs[seg].U
            else: # exactly two of the current derivative's parameters are in the same segment
                if o3_local_flags[i] == ThirdDerivativeType.IJ_LOCAL:
                    o2i = (o3i[0], o3i[1])
                    o1i = o3i[2]
                elif o3_local_flags[i] == ThirdDerivativeType.IK_LOCAL:
                    o2i = (o3i[0], o3i[2])
                    o1i = o3i[1]
                elif o3_local_flags[i] == ThirdDerivativeType.JK_LOCAL:
                    o2i = (o3i[1], o3i[2])
                    o1i = o3i[0]

                o2seg = self.seg_indexes[o2i[0]]
                o1seg = self.seg_indexes[o1i]
                for seg in range(self.n_segments):
                    # if the current segment contains the required second-order derivative, append the local second-order derivative to the to the global derivative...
                    if seg == o2seg:
                        d3U[i] = d3U[i] @ unitary_and_derivs[seg].d2U[get_o2_deriv_index(seg, o2i)]
                    # ...otherwise if the current segment contains the required first-order derivative, append the local first-order derivative to the global derivative...
                    elif seg == o1seg:
                        d3U[i] = d3U[i] @ unitary_and_derivs[seg].dU[o1i - self.seg_ranges[o1seg][0]]
                    # ...otherwise just append the unitary, since the derivative was appended by another segment
                    else:
                        d3U[i] = d3U[i] @ unitary_and_derivs[seg].U

        return DerivativeData(U=U, dU=dU, d2U=d2U, d3U=d3U)

    def run_optimization(self, initial_params: np.ndarray, target_unitary: np.ndarray, fidelity_func: GOATCostFunction, 
                         optimizer_opts: dict | None = None, store_prev_params=False, **kwargs):
        """Run a gate optimization and store the result in self.optimization_result.
        Args:
            initial_params (np.ndarray): Initial values of gate paramters to pass to the optimizer.
            target_unitary (np.ndarray): Target unitary used to compute gate fidelity.
            fidelity_func (GOATCostFunction): Function to compute gate fidelity and its gradient.
            optimizer_opts (dict, optional): Options to pass directly to the optimizer.
            _______________.
        Raises:
            ValueError: If the length of params is not equal to self.n_total_params.
        """

        if initial_params.size != self.n_total_params:
            raise ValueError(f"Expected params.size == {self.n_total_params}, got {initial_params.size}.")
        
        # cost and gradient function to pass to the optimizer
        def cost_and_grad_func(params):
            deriv_request = fidelity_func.get_deriv_request(params=params) # get indices of requested higher-order derivatives, if any

            if fidelity_func.uses_order3:
                deriv_data = self._get_unitary_and_order3_derivs(params, 
                                                                 order2_indices=deriv_request["order2_indices"], 
                                                                 order3_indices=deriv_request["order3_indices"])
                if fidelity_func.uses_params:
                    return fidelity_func(deriv_data, target_unitary, params=params, **kwargs)
                else:
                    return fidelity_func(deriv_data, target_unitary, **kwargs)
            elif fidelity_func.uses_order2:
                deriv_data = self._get_unitary_and_order2_derivs(params, 
                                                                 order2_indices=deriv_request["order2_indices"])
                if fidelity_func.uses_params:
                    return fidelity_func(deriv_data, target_unitary, params=params, **kwargs)
                else:
                    return fidelity_func(deriv_data, target_unitary, **kwargs)
            else:
                deriv_data = self._get_unitary_and_order1_derivs(params)
                if fidelity_func.uses_params:
                    return fidelity_func(deriv_data, target_unitary, params=params, **kwargs)
                else:
                    return fidelity_func(deriv_data, target_unitary, **kwargs)

        # default optimizer options
        if optimizer_opts is None:
            optimizer_opts = {"maxiter": 200, "disp": True, "ftol": 1e-15, "gtol": 1e-15}

        # compute and store optimized gate parameters
        self.optimization_result = minimize(
            fun=cost_and_grad_func,
            x0=initial_params,
            jac=True,
            method="L-BFGS-B",
            options=optimizer_opts
            # callback=, TODO: add this later
        )

    def evaluate_function(self, function: GOATFunction, eval_params: np.ndarray | None = None, **kwargs) -> Any:
        """Evaluate a function which requires computation of the gate unitary and its derivatives.
        Args:
            function (Callabe[..., Any]): The function to be evaluated.
            eval_params (np.ndarray, None): The parameters to be used to compute the gate unitary and its derivatives. If None, then self.optimization_result will be used.
            **kwargs: Any extra parameters to be directly passed to the function.
        Returns:
            Any: The value returned by the function after evaluation.
        Raises:
            ValueError: If eval_params is passed and its its length is not equal to self.n_total_params,
                        or if eval_params is not passed and self.optimization_result is None.
        """
        eval_params = self.optimization_result.x if eval_params is None else eval_params

        deriv_request = function.get_deriv_request(params=eval_params)

        if function.uses_order3:
            deriv_data = self._get_unitary_and_order3_derivs(eval_params, 
                                                             order2_indices=deriv_request["order2_indices"],
                                                             order3_indices=deriv_request["order3_indices"])
            return function(deriv_data, **kwargs)
        elif function.uses_order2:
            deriv_data = self._get_unitary_and_order2_derivs(eval_params,
                                                             order2_indices=deriv_request["order2_indices"]) 
            return function(deriv_data, **kwargs)
        else:
            deriv_data = self._get_unitary_and_order1_derivs(eval_params)
            return function(deriv_data, **kwargs)
