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
