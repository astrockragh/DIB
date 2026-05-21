#!/usr/bin/env python
"""
plot_chi2_grid_C2v_15272.py

Visualize chi^2 landscape from grid_search_C2v_15272.npz.

Rotational constants recovered from saved (A, Bf, Cf):
    B = Bf * A
    C = Cf * B = Cf * Bf * A
so  A > B > C  by construction.

Usage:
    python plot_chi2_grid_C2v_15272.py --T 10 --chi2 total
    python plot_chi2_grid_C2v_15272.py --T 10 --chi2 all   # 3-row figure
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _norm(chi2):
    vmin, vmax = np.nanmin(chi2), np.nanmax(chi2)
    if vmax / max(vmin, 1e-30) > 10:
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _add_forbidden(ax, x, y):
    """Shade the forbidden region y >= x on log-log axes."""
    lo = min(x.min(), y.min()) * 0.3
    hi = max(x.max(), y.max()) * 3
    diag = np.logspace(np.log10(lo), np.log10(hi), 400)
    ax.plot(diag, diag, 'k--', lw=1, zorder=2, label='A = B boundary')
    ax.fill_between(diag, diag, hi * 10, color='gray', alpha=0.10, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def _marginal_min(x, y, chi2):
    """Return unique (x, y) pairs with minimum chi2 over the third dimension."""
    # Use rounded log-values as keys to handle float precision
    best = {}
    for xi, yi, c in zip(x, y, chi2):
        if not np.isfinite(c):
            continue
        key = (round(np.log10(xi), 8), round(np.log10(yi), 8))
        if key not in best or c < best[key][2]:
            best[key] = (xi, yi, c)
    vals = np.array(list(best.values()))
    return vals[:, 0], vals[:, 1], vals[:, 2]


def plot_row(axes, A, B, C, chi2, chi2_label, cmap='plasma_r'):
    projections = [
        (axes[0], A, B, r'$A$ (cm$^{-1}$)', r'$B$ (cm$^{-1}$)', '($A$, $B$)'),
        (axes[1], A, C, r'$A$ (cm$^{-1}$)', r'$C$ (cm$^{-1}$)', '($A$, $C$)'),
        (axes[2], B, C, r'$B$ (cm$^{-1}$)', r'$C$ (cm$^{-1}$)', '($B$, $C$)'),
    ]
    scatters = []
    for ax, x_all, y_all, xl, yl, title in projections:
        # Marginalize over the third constant: keep min chi2 at each (x, y) pair
        x, y, c2 = _marginal_min(x_all, y_all, chi2)
        norm = _norm(c2)

        _add_forbidden(ax, x, y)

        sc = ax.scatter(x, y, c=c2, cmap=cmap, norm=norm,
                        s=90, edgecolors='k', linewidths=0.4, zorder=3)
        scatters.append(sc)

        ibest = np.argmin(c2)
        ax.scatter(x[ibest], y[ibest], marker='*', s=220, c='red',
                   edgecolors='darkred', linewidths=0.6, zorder=5,
                   label=f'best  {c2[ibest]:.1f}')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc='upper left')
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(chi2_label, fontsize=8)
    return scatters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=str(REPO_ROOT / 'grid_search_C2v_15272.npz'))
    ap.add_argument('--T', type=float, default=None,
                    help='Temperature slice in K (picks nearest available)')
    ap.add_argument('--chi2', choices=['direct', 'dT', 'total', 'all'],
                    default='total',
                    help='Which chi2 to visualize (all → 3-row figure)')
    args = ap.parse_args()

    d = np.load(args.npz)
    T_arr   = d['T']
    A_arr   = d['A']
    Bf_arr  = d['Bf']
    Cf_arr  = d['Cf']
    c2_dir  = d['chi2_direct']
    c2_dT   = d['chi2_dT']
    c2_tot  = d['chi2_total']

    B_arr = Bf_arr * A_arr
    C_arr = Cf_arr * B_arr

    T_vals = np.unique(T_arr)
    if args.T is None:
        T_pick = T_vals[0]
        print(f"No --T supplied; using T = {T_pick:.0f} K.  "
              f"Available: {T_vals.tolist()}")
    else:
        T_pick = T_vals[np.argmin(np.abs(T_vals - args.T))]
        print(f"Using T = {T_pick:.0f} K")

    mask = T_arr == T_pick
    A, B, C = A_arr[mask], B_arr[mask], C_arr[mask]
    sets = {
        'direct': (c2_dir[mask],  r'$\chi^2_\mathrm{direct}$'),
        'dT':     (c2_dT[mask],   r'$\chi^2_\mathrm{dT}$'),
        'total':  (c2_tot[mask],  r'$\chi^2_\mathrm{total}$'),
    }

    if args.chi2 == 'all':
        keys = ['direct', 'dT', 'total']
        fig, axes = plt.subplots(3, 3, figsize=(14, 13))
        fig.suptitle(f'C2v 15272 — $T = {T_pick:.0f}$ K', fontsize=13, y=1.01)
        for row, key in enumerate(keys):
            chi2, lbl = sets[key]
            plot_row(axes[row], A, B, C, chi2, lbl)
    else:
        chi2, lbl = sets[args.chi2]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        fig.suptitle(f'C2v 15272  {lbl} — $T = {T_pick:.0f}$ K', fontsize=13)
        plot_row(axes, A, B, C, chi2, lbl)

    plt.tight_layout()
    tag = f'T{T_pick:.0f}_{args.chi2}'
    out = str(REPO_ROOT / f'chi2_landscape_C2v_15272_{tag}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    plt.show()


if __name__ == '__main__':
    main()
