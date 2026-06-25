import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from src.experiments import run_experiment, run_noise_sweep, run_gd_experiment

CLIP = 20.0

RIDGE = {0.1:   ('#e31a1c', r'Ridge $\lambda=0.1$'), 1.0:   ('#ff7f00', r'Ridge $\lambda=1$'),
         10.0:  ('#1f78b4', r'Ridge $\lambda=10$'), 100.0: ('#6a3d9a', r'Ridge $\lambda=100$'),}

NOISE_COLORS = {0.1: '#1b7837', 0.5: '#4dac26', 1.0: '#d73027'}


def _clip(arr):
    return np.minimum(arr, CLIP)

def _band(ax, x, mean, std, color, alpha=0.15):
    ax.fill_between(x, np.maximum(0, mean - std), np.minimum(CLIP, mean + std), color=color, alpha=alpha)

def _vline(ax, n):
    ax.axvline(n, color='crimson', ls='--', lw=1.4, label=f'interpolation threshold $d={n}$', zorder=10)

def _save(fig, ax, title, xlabel, ylabel, outpath, **legend_kw):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, framealpha=0.85, **legend_kw)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.show()

def fig1_ols_train_test(d_values, n, res, sigma, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mean_key, std_key, color, label, ls in [('ols_test_mean',  'ols_test_std',  'steelblue', 'Test MSE',  '-'),
                                                ('ols_train_mean', 'ols_train_std', 'coral', 'Train MSE', '--'),]:
        m = _clip(res[mean_key])
        ax.plot(d_values, m, color=color, lw=2, ls=ls, label=label)
        _band(ax, d_values, m, res[std_key], color)

    peak = np.argmax(res['ols_test_mean'])
    ax.annotate(f'Peak\n$d={d_values[peak]}$',
                xy=(d_values[peak], _clip(res['ols_test_mean'])[peak]),
                xytext=(d_values[peak] + 10, _clip(res['ols_test_mean'])[peak] * 0.85),
                fontsize=8, arrowprops=dict(arrowstyle='->', color='black'))
    _vline(ax, n)
    _save(fig, ax,
          title=f'OLS Train/Test Error  ($n={n}$, $\\sigma={sigma}$, {res.get("n_trials", 30)} trials)',
          xlabel='Number of features $d$',
          ylabel='MSE (clipped at 20)',
          outpath=outpath,
          loc='upper right')

def fig2_test_comparison(d_values, n, res, sigma, lambdas, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    m = _clip(res['ols_test_mean'])
    ax.plot(d_values, m, color='black', lw=2.2, label='OLS', zorder=6)
    _band(ax, d_values, m, res['ols_test_std'], 'black', alpha=0.10)

    for lam in lambdas:
        color, label = RIDGE[lam]
        m = _clip(res['ridge_test_mean'][lam])
        ax.plot(d_values, m, color=color, lw=1.8, label=label)
        _band(ax, d_values, m, res['ridge_test_std'][lam], color, alpha=0.10)
    _vline(ax, n)
    _save(fig, ax,
          title=f'Test Error: OLS vs Ridge  ($n={n}$, $\\sigma={sigma}$)',
          xlabel='Number of features $d$',
          ylabel='Test MSE (clipped at 20)',
          outpath=outpath,
          ncol=2, loc='upper right')
    
def fig3_noise(d_values, n, noise_results, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    for sigma, res in noise_results.items():
        color = NOISE_COLORS.get(sigma, 'gray')
        m = _clip(res['ols_test_mean'])
        ax.plot(d_values, m, color=color, lw=2, label=f'$\\sigma={sigma}$')
        _band(ax, d_values, m, res['ols_test_std'], color)
    _vline(ax, n)
    _save(fig, ax, title=f'Effect of Noise on Double Descent  ($n={n}$)',
          xlabel='Number of features $d$', ylabel='OLS Test MSE (clipped at 20)',
          outpath=outpath, loc='upper right')
    
def fig3_noise_individual(d_values, n, noise_results, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (sigma, res) in zip(axes, noise_results.items()):
        color = NOISE_COLORS.get(sigma, 'gray')
        m = res['ols_test_mean']
        ax.plot(d_values, m, color=color, lw=2)
        _band(ax, d_values, m, res['ols_test_std'], color)
        _vline(ax, n)
        ax.set(title=f'σ = {sigma}', xlabel='$d$', ylim=(0, np.max(m) * 1.05))
        ax.grid(True, alpha=0.3, linestyle=':')
    axes[0].set_ylabel('Test MSE')
    fig.suptitle(f'OLS Test Error for Different Noise Levels  ($n={n}$)', fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)