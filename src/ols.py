import numpy as np
def ols_solution(X, y): 
    # compute the OLS, minimum norm solution
    
    n, d = X.shape

    if d < n:     
        # normal equations 
        # w = (X^T X)^{-1} X^T y
        
        A = X.T @ X          # (d, d)
        b = X.T @ y          # (d,)
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(X) @ y

    else:
        # minimum-norm solution (dual form)
        K = X @ X.T          # (n, n)
        try:
            alpha = np.linalg.solve(K, y)   # (n,)
            w = X.T @ alpha                 # (d,)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(X) @ y

    return w