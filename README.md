## tailblazer

A Python package containing several useful functions for working with the tails of a distribution. This is a natural extension to NumPy arrays and even supports 0, 1, and 2D arrays.

```python
import tailblazer as tbzr
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6])
tbzr.cume_tail_mean(x)
```
