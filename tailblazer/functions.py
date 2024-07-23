
import numpy as np
from numpy.typing import NDArray

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

        while x[indices[tail_floor]] < curr_tail:
            n_tail -= 1
            tail_sum -= x[indices[tail_floor]]
            tail_floor += 1

        mean_vec[i] = tail_sum / n_tail

    return mean_vec

def apply_multidim(x: NDArray, axis: int, target_fun: callable) -> NDArray:
    '''
    An internal helper function designed to make
    applying algorithims on multidimensional
    NumPy arrays simpler with standard error
    handling.
    '''

    n_dim = len(x.shape)
    transpose = n_dim == 2 and axis == 0

    if n_dim >= 3:
        raise ValueError('Only 0, 1, and 2D arrays are supported')

    flip = lambda x: x.T if transpose else x

    if n_dim == 2:
        return flip(np.array([target_fun(i) for i in flip(x)]))

    return target_fun(x)

def pct_rank(x: NDArray, axis=0):
    '''
    Computes cumulative distribution over a NumPy
    array of up to two dimensions. This works for
    any numeric-like NumPy array.

    x: A NumPy array
    axis: The axis (0 or 1) to operate over
    '''

    return apply_multidim(x, axis, pct_rank_1d)

def cume_tail_mean(x: NDArray, axis=0, tail: float=0.95):
    '''
    Computes cumulative tail menas over a NumPy
    array of up to two dimensions. This works for
    any numeric-like NumPy array.

    x: A NumPy array
    axis: The axis (0 or 1) to operate over,
    tail: A tail threshold between 0.01 and 0.99
    '''

    return apply_multidim(x, axis, lambda x: cume_tail_mean_1d(x, tail))
