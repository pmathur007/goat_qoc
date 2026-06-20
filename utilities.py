from dataclasses import dataclass
from enum import Enum
import numpy as np

@dataclass
class DerivativeData:
    """A structure to store the unitary and its derivatives returned from get_unitary_and_derivs in a control segment.

    Attributes:
        U (np.ndarray): The unitary.
        dU (np.ndarray): First-order derivatives of the unitary.
        d2U (np.ndarray): Second-order derivatives of the unitary.
        d3U (np.ndarray): Third-order derivatives of the unitary.
    """
    U: np.ndarray
    dU: np.ndarray
    d2U: np.ndarray | None = None
    d3U: np.ndarray | None = None

@dataclass
class DerivativeRequest:
    """A structure to store requests for higher-order derivatives passed to get_unitary_and_derivs in a control segment.

    Attributes:
        order2_indices (list[tuple[int, int]], optional): A list of 2-tuples of indices from the params array that indicate which second-order derivatives ddU should be calculated.
        order3_indices (list[tiple[int, int, int]], optional): A list of 3-tuples of indices from the params array that indicate which third-order derivatives dddU should be calculated.
        order2_lookup (dict[tuple[int, int], int], optional): A dict keyed by 2-tuples of indices from the params array with values corresponding to the index of each 2-tuple key in the order2_indices list.
    """
    order2_indices: list[tuple[int, int]] | None = None
    order3_indices: list[tuple[int, int, int]] | None = None
    order2_lookup: dict[tuple[int, int], int] | None = None

class ThirdDerivativeType(Enum):
    """The five different third-order derivative classifications used in the GOATOptimizer.get_unitary_and_order3_derivs function."""

    """All parameters are in the same segment."""
    ALL_LOCAL = 0

    """Only the first and second parameters fall in the same segment."""
    IJ_LOCAL = 1

    """Only the first and third parameters fall in the same segment."""
    IK_LOCAL = 2
    
    """Only the second and third parameters fall in the same segment."""
    JK_LOCAL = 3

    """All three parameters fall in different segments."""
    ALL_DISTINCT = 4