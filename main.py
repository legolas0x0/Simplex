from Simplex import Simplex
import numpy as np
c = np.array([[1],[1],[-2],[0]], dtype='float64')
b = np.array([[10],[5]], dtype='float64')
A = np.array([[1, 1, 1, 0], [1, -2, 0, 1]], dtype='float64')
s = Simplex(c, A, b)
print(s.optimize())
