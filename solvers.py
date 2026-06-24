import numpy as np

def solve(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)

    # elimination
    for k in range(n-1):
        for i in range(k+1, n):
            factor = A[i,k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - A[i,i+1:] @ x[i+1:]) / A[i,i]

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


def fit_weights(X, y):

    d, n = X.shape[1], X.shape[0]

    if d < n:
        A = X.T @ X
        b = X.T @ y
        try:
            w = solve(A, b)
        except np.linalg.LinAlgError:
            w = pinv(X) @ y
    else:
        w = pinv(X) @ y

    return w


#testing the functions and camparing them with numpy result:
rng = np.random.default_rng(42)

def check(name, ours, ref):
    err = np.max(np.abs(ours - ref))
    print(name, err)

# solve
A = rng.standard_normal((5, 5))
b = rng.standard_normal(5)
check("solve", solve(A, b), np.linalg.solve(A, b))

# pinv (tall)
A = rng.standard_normal((10, 4))
check("pinv tall", pinv(A), np.linalg.pinv(A))

# pinv (wide)
A = rng.standard_normal((4, 10))
A[2] = A[0] + A[1]
check("pinv wide", pinv(A), np.linalg.pinv(A))

# fit_weights 
X = rng.standard_normal((50, 8))
w_true = rng.standard_normal(8)
y = X @ w_true + 0.01 * rng.standard_normal(50)
check("fit_weights", fit_weights(X, y), w_true)

############################################
#   solve 5.551115123125783e-16            #
#   pinv tall 2.7755575615628914e-16       #
#   pinv wide 2.7755575615628914e-16       #
#   fit_weights 0.003794992715173473       #
############################################
