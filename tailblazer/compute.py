
import numpy as np
from numpy.typing import NDArray
from typing import Callable

#### Core Mechanics in 1D ####

def pct_rank_1d(x: NDArray) -> NDArray:
    '''
    Internal implementation of cumulative
    distribution over a 1D NumPy array. This
    is performant and memory safe.

    x: A NumPy array
    '''

    if len(x.shape) != 1:
        raise ValueError('Only 1D arrays are supported')

    indices = np.argsort(x)
    last_pct = None
    last_val = None
    pcts_vec = np.empty(x.shape)
    total_ns = x.shape[0]
    position = total_ns + 1

    for i in reversed(indices):
    
        position -= 1
        current = x[i]

        if current == last_val:
            pcts_vec[i] = last_pct
            continue

        last_pct = position / total_ns
        last_val = current
        pcts_vec[i] = last_pct

    return pcts_vec

def cume_tail_mean_1d(x: NDArray, tail: float=0.95) -> NDArray:
    '''
    Internal implementation of cumulative
    tail means over a 1D NumPy array. This
    is performant and memory safe.

    x: A NumPy array
    tail: A tail threshold between 0.01 and 0.99
    '''

    if tail < 0.01 or tail > 0.99:
        raise ValueError('`tail` must be between 0.01 and 0.99')

    indices = np.argsort(x)
    pcts_vec = pct_rank(x)
    tail_floor = 0
    mean_vec = np.empty(x.shape)
    tail_sum = 0
    n_tail = 0

    for i in indices:

        tail_sum += x[i]
        n_tail += 1
        curr_tail = tail * pcts_vec[i]

        while pcts_vec[indices[tail_floor]] < curr_tail:
            n_tail -= 1
            tail_sum -= x[indices[tail_floor]]
            tail_floor += 1

        mean_vec[i] = tail_sum / n_tail

    return mean_vec

#### Internal Helper ####

def apply_multidim(x: NDArray, axis: int, target_fun: Callable) -> NDArray:
    '''
    An internal helper function designed to make
    applying algorithims on multidimensional
    NumPy arrays simpler with standard error
    handling.
    '''

    try:
        x = np.array(x, dtype=float)
    except:
        raise TypeError('`x` must be an array or coercible to an array')

    if np.isnan(x).any():
        # NOTE: Consider adding support for nan values later
        raise ValueError('`x` cannot contain nan values')

    if axis not in (0, 1):
        raise ValueError('`axis` must be 0 or 1')

    n_dim = len(x.shape)
    transpose = n_dim == 2 and axis == 0

    if n_dim >= 3:
        raise ValueError('Only 1, and 2D arrays are supported')

    flip = lambda x: x.T if transpose else x

    if n_dim == 2:
        return flip(np.array([target_fun(i) for i in flip(x)]))

    return target_fun(x)

#### Exported Algorithms ####

def pct_rank(x: NDArray, axis: int=0) -> NDArray:
    '''
    Computes cumulative distribution over a NumPy
    array of up to two dimensions. This works for
    any numeric-like NumPy array.

    x: A NumPy array
    axis: The axis (0 or 1) to operate over
    '''

    return apply_multidim(x, axis, pct_rank_1d)

def cume_tail_mean(x: NDArray, axis: int=0, tail: float=0.95) -> NDArray:
    '''
    Computes cumulative tail menas over a NumPy
    array of up to two dimensions. This works for
    any numeric-like NumPy array.

    x: A NumPy array
    axis: The axis (0 or 1) to operate over,
    tail: A tail threshold between 0.01 and 0.99
    '''

    return apply_multidim(x, axis, lambda x: cume_tail_mean_1d(x, tail))
