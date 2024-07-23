import tailblazer as tbzr
import numpy as np
import pytest

#### Test Data ####

# Simple arrays
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])

# Shuffled arrays
shuffle_idx = [0, 7, 1, 2, 6, 3, 5, 4]
x_shuf = np.array([x[i] for i in shuffle_idx])
y_shuf = np.array([y[i] for i in shuffle_idx])

# 2D arrays
x_2d = np.array([[1, 2], [3, 4]])
y_2d_a0 = np.array([[0.5, 0.5], [1.0, 1.0]])
y_2d_a1 = np.array([[0.5, 1.0], [0.5, 1.0]])

# Special objects
x_nan = np.array([i if i != 1 else np.nan for i in x])
empty_arr = np.array([])
weird_data = {'weird': 'data'}
three_dim_arr = [[[1, 2]]]

#### Assertion Tests ####

def test_basic_pct_rank_usage():
    'The most simplistic case on a 1D array'

    assert all(tbzr.pct_rank(x) == y)

def test_unordered_input():
    'The same simple case but shuffled'

    assert all(tbzr.pct_rank(x_shuf) == y_shuf)

def test_empty_array():
    'Allows for empty arrays to simply pass through'

    assert tbzr.pct_rank(empty_arr).shape == (0, )

def test_missing_values():
    'Ensures nan values are halted'

    with pytest.raises(ValueError):
        tbzr.pct_rank(x_nan)

def test_weird_objects():
    'Ensures we must have a valid numeric array'

    with pytest.raises(TypeError):
        tbzr.pct_rank(weird_data)

def test_2d_object():
    'Tests operating on different axes'

    assert np.all(tbzr.pct_rank(x_2d, 0) == y_2d_a0)
    assert np.all(tbzr.pct_rank(x_2d, 1) == y_2d_a1)

def test_3d_objects():
    'Tests halting on 3D+ arrays'

    with pytest.raises(ValueError):
        tbzr.pct_rank(three_dim_arr)

def test_werid_axis():
    'Tests halting on invalid axis'

    with pytest.raises(ValueError):
        tbzr.pct_rank(x, axis=0.5)
