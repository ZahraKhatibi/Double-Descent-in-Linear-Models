# Double Descent in Linear Models

An empirical and from-scratch study of the **double descent** phenomenon in linear models, where test error follows a two-descent curve rather than the classical U-shape.

---

## Overview

Classical statistical says more parameters lead to overfitting. Double descent breaks that rule: as the number of features $d$ grows past the number of training samples $n$, the test error spikes at $d = n$ then descends again. This project investigates why that happens and how regularisation and optimisation method affect it.

All solvers are implemented from scratch in pure NumPy, including Gaussian elimination, Householder bidiagonalisation, a full SVD, and the Moore–Penrose pseudoinverse, with no dependence on `scipy` or sklearn.

---

## Repository Structure

```
.
├── notebook.ipynb       # Main notebook: runs all experiments and produces all figures
├── src/
│   ├── data.py          # Synthetic dataset generation
│   ├── ols.py           # Minimum-norm OLS solver (normal equations + dual form)
│   ├── ridge.py         # Ridge regression solver
│   ├── solvers.py       # Custom linear algebra: Gaussian elimination, SVD, pseudoinverse
│   ├── metrics.py       # MSE and weight-norm computation
│   ├── experiments.py   # Experiment runners: OLS/Ridge sweep, noise sweep, GD comparison
│   └── plotting.py      # All figure-generation functions
└── figures/             # Output plots 
```

---

## Experiments

### 1. OLS Train/Test Sweep
Sweeps $d$ from 10 to 300 with $n = 100$ fixed, plotting both train and test MSE to visualise the full double descent curve and the interpolation threshold spike.

### 2. OLS vs Ridge
Compares OLS against Ridge regression at $\lambda \in \{0.1, 1, 10, 100\}$. Ridge eliminates the interpolation peak by keeping the weight norm bounded, trading a small amount of bias for dramatically reduced variance near the threshold.

### 3. Noise Sweep
Repeats the OLS sweep for $\sigma \in \{0.1, 0.5, 1.0\}$ to show that the double descent shape is universal across noise levels.

### 4. Weight Norm Analysis
Tracks $\|\hat{w}\|_2$ alongside test MSE to show that the test-error spike is directly caused by the norm blowing up near $d = n$, and how Ridge regularisation controls it.

### 5. Gradient Descent vs Closed Form
Compares gradient descent at various learning rates and epoch counts against the closed-form OLS solution. GD converges to the same minimum-norm solution; early stopping acts as implicit Ridge regularisation.

---

## Experiment Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n` | 100 | Training samples |
| `sigma` | 0.5 | Label noise standard deviation |
| `n_trials` | 30 | Independent repetitions per $(d, \lambda)$ pair |
| `lambdas` | [0.1, 1, 10, 100] | Ridge regularisation strengths |
| `d_values` | 10, 15, …, 300 | Feature dimensions swept (step = 5) |

Each $(d, \text{trial})$ pair uses a unique deterministic seed for full reproducibility. Plots show mean ± 1 standard deviation bands across trials.

---

## Implementation Highlights

**`data.py`**: The true weight vector is $w^* = \mathbf{1}/\sqrt{d}$, so every feature carries equal signal and $\|w^*\|_2 = 1$ regardless of $d$, making comparisons across dimensions fair.

**`ols.py`**: Implements two branches, normal equations $(X^\top X)^{-1} X^\top y$ when $d < n$, and the dual form $X^\top (XX^\top)^{-1} y$ (minimum-norm solution) when $d \geq n$.

**`ridge.py`**: Same two-branch logic; adding $\lambda I$ makes the system strictly positive definite, eliminating the singularity at the threshold.

**`solvers.py`**: Implements the full numerical stack from scratch:
- `solve(A, b)`: Gaussian elimination with back-substitution
- `_bidiagonalise(A)`: Householder reflections to upper bidiagonal form
- `_householder_svd(A)`: Full SVD via bidiagonalisation
- `pinv(A)`: Moore-Penrose pseudoinverse $A^+ = V\Sigma^+ U^\top$
- `fit_weights(X, y)`: Combines the above into a regression solver

---

## Key Findings

| Experiment | Finding |
|------------|---------|
| OLS sweep | Clear double descent: test MSE peaks sharply at $d = n$ then recovers |
| OLS vs Ridge | Ridge eliminates the peak; optimal $\lambda$ beats OLS everywhere |
| Noise sweep | Double descent is universal; peak height scales with $\sigma^2$ |
| Weight norm | The test-error peak is directly caused by norm blow-up at $d = n$ |
| GD comparison | GD converges to the minimum-norm solution; early stopping ≈ Ridge |

---

## Getting Started

**Requirements:** Python 3.8+, NumPy, Matplotlib

```bash
git clone https://github.com/ZahraKhatibi/Double-Descent-in-Linear-Models
cd Double-Descent-in-Linear-Models
pip install numpy matplotlib jupyter
jupyter notebook notebook.ipynb
```

Run all cells in order. Figures are saved to the `figures/` directory automatically.

---

## Reference

- Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). [Reconciling modern machine-learning practice and the classical bias–variance trade-off.](https://www.pnas.org/doi/10.1073/pnas.1903070116) *PNAS*.

