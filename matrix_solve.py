from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve
import numpy as np
import math
def cholesky(l, p, b):
    M = p + l
    N = p@b
    c, inf = cho_factor(M)
    return cho_solve((c, inf), N)

def LU(M, N):
    LU, p = lu_factor(M)
    return lu_solve((LU, p), N)