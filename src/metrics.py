import numpy as np

def compute_mse(w, X, y):
    
    residuals = X @ w - y
    
    return float(np.mean(residuals ** 2))


def compute_weight_norm(w):
    
    return float(np.linalg.norm(w))
