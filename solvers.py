import numpy as np

def lu_decomposition(A): #finding P,L,U such that P @ A = L @ U
    
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("lu_decomposition requires a square matrix")

    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)       

    for k in range(n - 1):
        # partial pivot, find row with largest absolute value in column k
        pivot_row = k + np.argmax(np.abs(U[k:, k]))
        if pivot_row != k:
            U[[k, pivot_row]] = U[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            # fix already-computed L columns to the left of k
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        # elimination
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return P, L, U