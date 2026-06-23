import numpy as np

#synthetic dataset generation for the double-descent experiments

def generate_data(n, d, sigma, rng, n_test=2000):

    w_star = np.ones(d) / np.sqrt(d)       
    
    X_train = rng.standard_normal((n, d))
    y_train = X_train @ w_star + rng.normal(0.0, sigma, size=n)

    X_test = rng.standard_normal((n_test, d))
    y_test = X_test @ w_star + rng.normal(0.0, sigma, size=n_test)

    return X_train, y_train, X_test, y_test, w_star
