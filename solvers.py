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
            if U[k, k] == 0:
                continue                    # singular column – skip
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return P, L, U


def forward_substitution(L, b): #  forward substitution  Ly = b  (L lower)
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y


def back_substitution(U, y): #  back substitution  Ux = y  (U upper)
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x

def solve(A, b): #solve Ax = b using LU decomposition with partial pivoting
    
    A = np.asarray(A, dtype=float) # Ax = b
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise np.linalg.LinAlgError("solve requires a square matrix")
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b sizes are incompatible")

    P, L, U = lu_decomposition(A)   # PA = LU

    # check for singular diagonal in U
    if np.any(np.abs(np.diag(U)) < 1e-14):
        raise np.linalg.LinAlgError("Matrix is singular to working precision")

    Pb = P @ b  
    y  = forward_substitution(L, Pb)  #Forward-sub:  Ly = Pb
    x  = back_substitution(U, y)    #Back-sub: Ux = y
    return x

def _bidiagonalise(A):
    #reduce A (m×n, m>=n) to upper bidiagonal form B = U_b @ A @ V_b
    
    A = A.copy()
    m, n = A.shape
    U_b = np.eye(m)
    V_b = np.eye(n)

    for k in range(n):
        # left: zero out below A[k, k]
        x = A[k:, k].copy()
        norm_x = np.linalg.norm(x)
        if norm_x > 0:
            x[0] += np.sign(x[0]) * norm_x
            x /= np.linalg.norm(x)
            H = np.eye(m)
            H[k:, k:] -= 2.0 * np.outer(x, x)
            A = H @ A
            U_b = H @ U_b

        # right: zero out to the right of A[k, k+1] 
        if k < n - 2:
            x = A[k, k+1:].copy()
            norm_x = np.linalg.norm(x)
            if norm_x > 0:
                x[0] += np.sign(x[0]) * norm_x
                x /= np.linalg.norm(x)
                H = np.eye(n)
                H[k+1:, k+1:] -= 2.0 * np.outer(x, x)
                A = A @ H
                V_b = V_b @ H

    return U_b.T, A, V_b   #returns U_b (m×m), B (m×n), V_b (n×n)


def _householder_svd(A):
    #full SVD of A via bidiagonalisation + numpy's SVD on the bidiagonal
    
    m, n = A.shape
    transpose = m < n
    if transpose:
        A = A.T
        m, n = A.shape

    U_b, B, V_b = _bidiagonalise(A)
    U_s, s, Vt_s = np.linalg.svd(B, full_matrices=False)

    U   = U_b @ U_s     
    Vt  = Vt_s @ V_b.T

    if transpose:
        return Vt.T, s, U.T
    return U, s, Vt  #returns U, s, Vt  (same convention as np.linalg.svd(A, full_matrices=False)


def pinv(A, rcond=None):
    # compute the Moore-Penrose pseudo-inverse of A.

    A = np.asarray(A, dtype=float)

    U, s, Vt = _householder_svd(A)

    if rcond is None:
        rcond = max(A.shape) * np.finfo(float).eps

    # threshold and invert singular values
    threshold = rcond * s[0]   
    s_inv = np.where(s <= threshold, 0.0, 1.0 / s)

    # A⁺ =  A⁺ = V Σ⁺ Uᵀ = V diag(s_inv) Uᵀ
    return (Vt.T * s_inv) @ U.T


