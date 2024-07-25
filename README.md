## tailblazer <img src="img/logo.png" align="right"height="138"/>

A Python package containing several useful functions for working
with the tails of a distribution. This is a natural extension to
NumPy arrays and even supports 0, 1, and 2D arrays.

```python
import tailblazer as tbzr
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6])
tbzr.cume_tail_mean(x, tail=0.7)

# Output
>>> array([1. , 1.5, 2. , 2.5, 3. , 3.5])
```

## Included Algorithms

The following sections provide an overview for the mechanics
of the included algorithms and how the balance flexibility and
efficiency.

### `pct_rank()`

This algorithim computes cumulative distribution rankings like
`dplyr::cume_dist()` in R. This was a notable gap in the NumPy
ecosystem and this function does so in an efficient way using
in-place modification following an `np.argsort()`. This means
that the sorting operation only occurs once and only two new
vectors of memory are allocated. The actual implementation is a
bit more complex, but this code captures the main idea:

```python
# Some array
x = array()

# Get a list of indices from `x` sorted
indices = argsort(x)

# Create container for output
pcts = empty(x.shape)

# Iterate over the indices in reverse
# This means ties are ranked the same
for i in reversed(indices):

    # There is some code here that establishes:
    # - first: are we on the first iteration
    # - current: the current value
    # - previous: the previous value (to handle ties)
    # - last_pct: ther percentile from the last iteration
    # - position: the integer position in the rank
    # - n: the total number of values

    if first or current == previous:
        pcts[i] = last_pct
        continue

    pcts[i] = position / n

return pcts
```

This is efficient because we allocate memory for the argsort
and percentiles, but don't need to sort, unsort, or allocate
any other temporary vectors!

### `cume_tail_mean()`

This function is a highly performant algorithim which computes
the average of all values within the top Nth percentile of a
vector below some quantile. In other words, if we iterate
through a vector and look at the top 5% of values below or
equal to that point, what is their mean? This is a challenging
function to implement because it is tempting to write a for
loop which iterates over the values, filters down the vector,
and computes the means:

```python
# Lazy implementation

# Some vector
x = array()

# Create container
means = empty(x.shape)

# Iterate over vector
for i, v in enumerate(x):
    under = x[x <= i]
    means[i] = mean(under >= quantile(under, 0.95))

return means
```

The implementation above is extremely computationally intensive
because the quantiles are estimated once **for every value**.
Instead, we can build on our `pct_rank()` function from above
and leverage the power of rescaling to determine a tail threshold
for every value.

Suppose we are at some value in the vector. We can use `pct_rank()`
to tell us its cumulative percentile. Maybe this value is 0.45. We
can approximate the 95th percentile using the current value percentile!
Multiply `0.45 * 0.95` and voila the current 95th tail is `0.4725`.
Since we now know the 95th percentile for every value, we can iterate
through the vector and keep track of how many items are in the current
tail and their sums.

The basic idea is to iterate through the vector, add the current value
to the tail, and strip away values which are now excluded. I call this
algorithm the catch-up algorithm because the tail is catching up to the
current value:

```python
# Some array
x = array()

# Get a list of indices and percentiles
indices = argsort(x)
pcts = pct_rank(x)
means = empty(np.shape)

# Establish the lower bound for the tail
tail_floor = 0

for i, v in enumerate(indices):

    curr_tail = 0.95 * pcts[i]
    tail_sum += v
    n_tail += 1

    while pcts[indices[i]] < curr_tail]:
        tail_sum -= x[indices[tail_floor]]
        n_tail -= 1
        tail_floor += 1

    means[i] = tail_sum / n_tail

return means
```