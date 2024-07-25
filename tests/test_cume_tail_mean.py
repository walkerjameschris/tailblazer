import tailblazer as tbzr
import numpy as np
import pytest

#### Test Data ####

# Simple arrays
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1.0, 1.5, 2.5, 3.0, 4.0, 4.5, 5.5, 6.0])

# Shuffled arrays
shuffle_idx = [0, 7, 1, 2, 6, 3, 5, 4]
x_shuf = np.array([x[i] for i in shuffle_idx])
y_shuf = np.array([y[i] for i in shuffle_idx])

# 2D arrays
x_2d = np.array([[1, 2, 3, 4], [3, 4, 5, 6]])
y_2d_a0 = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
y_2d_a1 = np.array([[1.0, 1.5, 2.5, 3.0], [3.0, 3.5, 4.5, 5.0]])

# Special objects
x_nan = np.array([i if i != 1 else np.nan for i in x])
empty_arr = np.array([])
weird_data = {'weird': 'data'}
three_dim_arr = [[[1, 2]]]

#### Assertion Tests ####

def test_basic_tail_mean_usage():
    'The most simplistic case on a 1D array'

    assert all(tbzr.cume_tail_mean(x, tail=0.5) == y)

def test_unordered_input():
    'The same simple case but shuffled'

    assert all(tbzr.cume_tail_mean(x_shuf, tail=0.5) == y_shuf)

def test_empty_array():
    'Allows for empty arrays to simply pass through'

    assert tbzr.cume_tail_mean(empty_arr).shape == (0, )

def test_missing_values():
    'Ensures nan values are halted'

    with pytest.raises(ValueError):
        tbzr.cume_tail_mean(x_nan)

def test_weird_objects():
    'Ensures we must have a valid numeric array'

    with pytest.raises(TypeError):
        tbzr.cume_tail_mean(weird_data) # type: ignore

def test_2d_object():
    'Tests operating on different axes'

    assert np.all(tbzr.cume_tail_mean(x_2d, 0, 0.5) == y_2d_a0)
    assert np.all(tbzr.cume_tail_mean(x_2d, 1, 0.5) == y_2d_a1)

def test_3d_objects():
    'Tests halting on 3D+ arrays'

    with pytest.raises(ValueError):
        tbzr.cume_tail_mean(three_dim_arr) # type: ignore

def test_werid_axis():
    'Tests halting on invalid axis'

    with pytest.raises(ValueError):
        tbzr.cume_tail_mean(x, axis=0.5) # type: ignore
