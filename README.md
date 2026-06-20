# goat_qoc 
Gradient Optimization of Analytic Controls for quantum optimal control.

---

``goat_qoc`` is a Python implementation of the GOAT algorithm introduced in Ref. [1]. The library is designed for optimization of parameterized control pulses for quantum devices.

## Installation

```bash
git clone https://github.com/pmathur007/goat-qoc.git 
cd goat-qoc
```

Dependencies:
- numpy
- scipy

## Examples

Several examples of how to use the ``goat_qoc`` library to optimize two-qubit Rydberg gates in neutral atom quantum computers can be found in ``goat_examples.ipynb``.

## Core Concepts

### Control Segments

In ``goat_qoc``, the gate to be optimized is broken up into *control segments* that can each be governed by distinct Hamiltonians and control parameters. Thus, the unitary implementing the gate can be written as,
$$U(\overline{\alpha})=U_1(\overline{\alpha}_1)U_2(\overline{\alpha}_2)\cdot\ldots\cdot U_N(\overline{\alpha}_N),$$
where $U$ is the overall gate unitary, $U_k$ is the unitary implemented by the $k$'th control segment, and $\overline{\alpha}$ is the list of control parameters consisting of $\overline{\alpha}_1,\overline{\alpha}_2,\ldots,\overline{\alpha}_N$, the control parameters for all the segements concatenated together. Setting up the system in this way enables optimization of arbitrarily complex control sequences.

Control segments can be specified in many ways (see ``goat_examples.ipynb`` for examples), but the only requirement is that there is some mechanism for computing the segment's unitary $U_k$ and its derivatives with respect to its control paramters $\partial_{\overline{\alpha}_k}U_k$. An abstract ``ControlSegment`` class can be found in ``control_segments.py``, along with the ``AnalogControlSegment`` class which is used to implement GOAT as originally specified in Ref [1].

### Costs and Other Functions

Optimzing a gate with GOAT requires providing the algorithm a cost function that is a function of the gate unitary and its derivatives. The most common cost function is gate fidelity, but one of the advantages of GOAT is that the cost function can be defined flexiblity, as demonstrated in ``goat_examples.ipynb``. In any case, ``goat_qoc`` defines two abstract, callable classes that consume the gate unitary and its derivatives. A ``GOATCostFunction`` is meant to be used as a cost function when optimzing a gate, and thus the return type of its ``__call__`` method must be ``tuple[float, np.ndarray]``, i.e. the cost and its gradient. A ``GOATFunction``, on the other hand, is more general and can have any return type. The abstract ``GOATCostFunction`` and ``GOATFunction`` classes, along with some examples of each, can be found in ``goat_functions.py``. 

### The GOATOptimizer

The ``GOATOptimizer`` class is the entry point for optimizing gates with GOAT. It is initialized with a list of control segments that encapsulate all the physics of the desired gate and how the gate is parameterized. After initialization, the ``run_optimization`` member function can be used to optimize gates, and the ``evaluate_function`` member function can be used to evaluate any GOAT function in an ad-hoc manner. The ``GOATOptimizer``'s main function is to connect control segments which produce gate unitaries and their derivatives with GOAT functions that consume gate unitaries and their derivatives. Another important function is to build up derivatives of the global unitary $\partial_{\overline{\alpha}}U$ from local derivatives of each control segment's unitaries $\partial_{\overline{\alpha}_k}U_k$, which are supplied by the control segments.


## References

[1] S. Machnes, E. Assémat, D. Tannor, and F. K. Wilhelm, “Tunable, Flexible, and Effcient Optimization of Control Pulses for Practical Qubits”, Physical Review Letters 120, 150401 (2018).

Please direct any questions, suggestions, or corrections about this code to Pranav Mathur (pranavmathur@g.harvard.edu).