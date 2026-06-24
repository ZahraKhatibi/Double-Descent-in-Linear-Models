import numpy as np


def ridge_solution(X, y, lam):
    
    n, d = X.shape

    if d <= n:
        A = X.T @ X + lam * np.eye(d)   # (d , d)
        b = X.T @ y                      # (d ,)
        w = np.linalg.solve(A, b)
    else:
        K = X @ X.T + lam * np.eye(n)   # (n , n)
        alpha = np.linalg.solve(K, y)   # (n ,)
        w = X.T @ alpha                 # (d ,)

    return w
