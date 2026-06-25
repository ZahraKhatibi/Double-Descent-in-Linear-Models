import numpy as np
from src.data import generate_data
from src.ols import ols_solution
from src.ridge import ridge_solution
from src.metrics import compute_mse, compute_weight_norm


def run_experiment(n, d_values, sigma, lambdas, n_trials, base_seed=42, verbose=True):

    nd = len(d_values)

    # save per-d, per-trial arrays for OLS
    ols_tr_buf   = np.zeros((nd, n_trials))
    ols_te_buf   = np.zeros((nd, n_trials))
    ols_norm_buf = np.zeros((nd, n_trials))

    # save for ridge
    ridge_tr_buf   = {lam: np.zeros((nd, n_trials)) for lam in lambdas}
    ridge_te_buf   = {lam: np.zeros((nd, n_trials)) for lam in lambdas}
    ridge_norm_buf = {lam: np.zeros((nd, n_trials)) for lam in lambdas}

    if verbose:
        print(f"  n={n}, sigma={sigma}, trials={n_trials}, "
              f"d=[{d_values[0]},{d_values[-1]}], lambdas={lambdas}")

    for i, d in enumerate(d_values):
        for t in range(n_trials):
            seed = base_seed + i * n_trials + t
            rng  = np.random.default_rng(seed)
            X_tr, y_tr, X_te, y_te, _ = generate_data(n, d, sigma, rng)

            # ols
            w_ols = ols_solution(X_tr, y_tr)
            ols_tr_buf[i, t]   = compute_mse(w_ols, X_tr, y_tr)
            ols_te_buf[i, t]   = compute_mse(w_ols, X_te, y_te)
            ols_norm_buf[i, t] = compute_weight_norm(w_ols)

            # ridge
            for lam in lambdas:
                w_r = ridge_solution(X_tr, y_tr, lam)
                ridge_tr_buf[lam][i, t]   = compute_mse(w_r, X_tr, y_tr)
                ridge_te_buf[lam][i, t]   = compute_mse(w_r, X_te, y_te)
                ridge_norm_buf[lam][i, t] = compute_weight_norm(w_r)

        if verbose and ((i + 1) % 10 == 0 or (i + 1) == nd):
            print(f"    d={d_values[i]:4d}  "
                  f"OLS test={ols_te_buf[i].mean():.4f} "
                  f"±{ols_te_buf[i].std():.4f}")

    results = {
        'ols_train_mean': ols_tr_buf.mean(axis=1),
        'ols_train_std':  ols_tr_buf.std(axis=1),
        'ols_test_mean':  ols_te_buf.mean(axis=1),
        'ols_test_std':   ols_te_buf.std(axis=1),
        'ols_norm_mean':  ols_norm_buf.mean(axis=1),
        'ols_norm_std':   ols_norm_buf.std(axis=1),
        'ridge_train_mean': {lam: ridge_tr_buf[lam].mean(axis=1) for lam in lambdas},
        'ridge_train_std':  {lam: ridge_tr_buf[lam].std(axis=1)  for lam in lambdas},
        'ridge_test_mean':  {lam: ridge_te_buf[lam].mean(axis=1) for lam in lambdas},
        'ridge_test_std':   {lam: ridge_te_buf[lam].std(axis=1)  for lam in lambdas},
        'ridge_norm_mean':  {lam: ridge_norm_buf[lam].mean(axis=1) for lam in lambdas},
        'ridge_norm_std':   {lam: ridge_norm_buf[lam].std(axis=1)  for lam in lambdas},
    }
    return results


def run_noise_sweep(n, d_values, sigma_list, lambdas, n_trials, base_seed=77, verbose=True):
    noise_results = {}
    for sigma in sigma_list:
        if verbose:
            print(f"\n[Noise sweep] sigma={sigma}")
        noise_results[sigma] = run_experiment( n, d_values, sigma, lambdas=lambdas, n_trials=n_trials,
                                              base_seed=base_seed, verbose=verbose)
    return noise_results


def gd_ols_solution(X, y, n_epochs=800, lr=0.01):
    n, d = X.shape
    w = np.zeros(d)
    XtX = X.T @ X
    Xty = X.T @ y
    for _ in range(n_epochs):
        grad = (2/n)* X.T @ (X @ w - y)
        w = w - lr * grad
    return w


def run_gd_experiment(n, d_values, sigma, n_trials=10, n_epochs=800, lr=0.01, base_seed=55, verbose=True):

    #compare closed-form OLS with gradient descent

    nd = len(d_values)
    gd_buf = np.zeros((nd, n_trials))
    cf_buf = np.zeros((nd, n_trials))

    if verbose:
        print(f"\n[GD] n_epochs={n_epochs}, n_trials={n_trials}, lr={lr}")

    for i, d in enumerate(d_values):
        for t in range(n_trials):
            seed = base_seed + i * n_trials + t
            rng  = np.random.default_rng(seed)
            X_tr, y_tr, X_te, y_te, _ = generate_data(n, d, sigma, rng)

            w_gd = gd_ols_solution(X_tr, y_tr, n_epochs=n_epochs, lr=lr)
            gd_buf[i, t] = compute_mse(w_gd, X_te, y_te)

            w_cf = ols_solution(X_tr, y_tr)
            cf_buf[i, t] = compute_mse(w_cf, X_te, y_te)

    return (gd_buf.mean(axis=1), gd_buf.std(axis=1),
            cf_buf.mean(axis=1), cf_buf.std(axis=1))
