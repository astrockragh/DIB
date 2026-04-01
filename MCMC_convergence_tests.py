import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde, ks_2samp
import corner, h5py, os, io

# ======================
# MATPLOTLIB STYLE
# ======================
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["figure.facecolor"] = "white"
plt.rc("text", usetex=False)
plt.rc("font", family="serif", size=20)
plt.rc("axes", linewidth=1.5)
plt.rc("axes", labelsize=18)
plt.rc("xtick", labelsize=12, direction="in")
plt.rc("ytick", labelsize=12, direction="in")
plt.rc("xtick", top=True)
plt.rc("ytick", right=True)
plt.rc("xtick.minor", visible=True)
plt.rc("ytick.minor", visible=True)
plt.rc("xtick.major", size=8, pad=4)
plt.rc("xtick.minor", size=4, pad=4)
plt.rc("ytick.major", size=8)
plt.rc("ytick.minor", size=4)
plt.rc("legend", fontsize=10)

# ======================
# DIAGNOSTICS
# ======================

def detect_multimodal(samples_1d):
    kde0 = gaussian_kde(samples_1d)
    kde  = gaussian_kde(samples_1d, bw_method=kde0.factor * 2)
    xs   = np.linspace(samples_1d.min(), samples_1d.max(), 200)
    ys   = kde(xs)
    peaks      = np.where((ys[1:-1] > ys[:-2]) & (ys[1:-1] > ys[2:]))[0]
    true_peaks = peaks[ys[peaks] > 0.05 * np.max(ys)]
    return len(true_peaks)

def ks_stability_test(chain, flat_samples, tau=100):
    """
    Returns per-parameter p-values comparing:
      1) 4 consecutive chunks of the second half of flat_samples
      2) last tau vs previous tau samples (tail test)
    """
    ndim   = flat_samples.shape[1]
    splits = np.array_split(flat_samples[len(flat_samples) // 2:], 4)

    pvals_chunks = []
    for i in range(ndim):
        pvals = [ks_2samp(splits[j][:, i], splits[j+1][:, i]).pvalue
                 for j in range(len(splits) - 1)]
        pvals_chunks.append(np.max(pvals))

    tail = flat_samples[-2 * tau:]
    if len(tail) < 2 * tau:
        pvals_tail = [np.nan] * ndim
    else:
        pvals_tail = [ks_2samp(tail[:tau, i], tail[tau:, i]).pvalue
                      for i in range(ndim)]

    return np.array(pvals_chunks), np.array(pvals_tail)

def compute_rhat(chain):
    """
    Gelman-Rubin R-hat statistic.
    chain: (nsteps, nwalkers, ndim)
    R-hat ~ 1.0 means converged; values > 1.01 suggest lack of convergence.
    """
    nsteps, nwalkers, ndim = chain.shape
    rhat = np.zeros(ndim)
    for i in range(ndim):
        chains_i   = chain[:, :, i].T          # (nwalkers, nsteps)
        n          = nsteps
        W          = np.mean(np.var(chains_i, axis=1, ddof=1))
        B          = n * np.var(np.mean(chains_i, axis=1), ddof=1)
        var_hat    = (n - 1) / n * W + B / n
        rhat[i]    = np.sqrt(var_hat / W) if W > 0 else np.nan
    return rhat

def format_diagnostics(n, tau, old_tau, ess, af, stuck, multimodal_peaks,
                        ks_chunks, ks_tail, rhat, labels=None):
    """Build the full diagnostic string; both printed and saved to HDF5."""
    def pname(i):
        return labels[i] if labels and i < len(labels) else f"param_{i}"

    lines = [
        f"\n{'='*54} DIAGNOSTICS {'='*54}",
        f"Iteration : {n}",
        f"{'-'*120}",
        f"Autocorrelation time (tau)         : {np.round(tau, 2)}",
        f"Tau change since last checkpoint   : {np.round(100 * np.abs(tau - old_tau) / np.where(old_tau == np.inf, 1, old_tau), 2)} %",
        f"  -> Want stable tau (change <5%) and N > 40*tau",
        f"ESS                                : {np.round(ess, 1)}",
        f"  -> Want ESS > 100 for reliable posteriors",
        f"Acceptance fraction (mean)         : {af.mean():.3f}",
        f"  -> Ideal range ~0.2-0.5",
        f"Stuck walkers                      : {np.sum(stuck)}",
        f"Multimodal flags                   : {multimodal_peaks > 1}",
    ]
    if np.any(multimodal_peaks > 1):
        lines.append(f"  Number of peaks                  : {multimodal_peaks}")
    lines += [
        f"  -> True indicates possible multiple posterior modes",
        f"",
        f"KS p-values (4-chunk stability)    : { {pname(i): float(f'{ks_chunks[i]:.3f}') for i in range(len(ks_chunks))} }",
        f"KS p-values (tail test)            : { {pname(i): float(f'{ks_tail[i]:.3f}') for i in range(len(ks_tail))} }",
        f"  -> p > 0.05 suggests the distribution is stationary across chunks / the tail",
        f"",
        f"R-hat (Gelman-Rubin)               : { {pname(i): float(f'{rhat[i]:.4f}') for i in range(len(rhat))} }",
        f"  -> R-hat < 1.01 indicates good convergence; values > 1.1 suggest chains have not mixed",
        f"{'='*120}",
    ]
    return "\n".join(lines)

# ======================
# LAYOUT HELPER
# ======================

layouts = {
    1:  (1, 1),  2:  (2, 1),  3:  (2, 2),  4:  (2, 2),
    5:  (3, 2),  6:  (3, 2),  7:  (4, 2),  8:  (4, 2),
    9:  (3, 3),  10: (4, 3),  11: (4, 3),  12: (4, 3),
    13: (4, 4),  14: (4, 4),  15: (4, 4),  16: (4, 4),
}

# ======================
# SUMMARY FIGURE
# ======================

def make_summary_figure(flat_samples, chain, filename,
                        labels=None, iter_hist=None, tau_hist=None,
                        ks_chunks=None, ks_tail=None):

    nsteps, nwalkers, ndim = chain.shape
    nrows, ncols = layouts[ndim]

    size = 4 * np.sqrt(ndim)

    fig = plt.figure(figsize=(2 * size, 2 * size), constrained_layout=True)
    subfigs = fig.subfigures(2, 2, wspace=0.05, hspace=0.05)

    # ---- Upper-left: corner plot ----
    corner.corner(
        flat_samples,
        labels=labels,
        titles=labels,
        quantiles=[0.16, 0.50, 0.84],
        title_quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt=".3f",
        fig=subfigs[0, 0],
        label_kwargs={"fontsize": 12},
        title_kwargs={"fontsize": 12},
    )
    subfigs[0, 0].suptitle("Posterior corner", fontsize=11, y=1.03, x=0.55)

    # ---- Upper-right: traces ----
    axs_trace = subfigs[0, 1].subplots(nrows, ncols, squeeze=False)
    subfigs[0, 1].suptitle("Walker traces", fontsize=12)
    for idx in range(ndim):
        r, c = divmod(idx, ncols)
        ax   = axs_trace[r, c]
        ax.plot(chain[:, :, idx], "k", alpha=0.2 / np.sqrt(nwalkers), rasterized=True)
        ax.set_title(labels[idx] if labels else f"param {idx}", fontsize=8)
        ax.set_xlabel("Iteration", fontsize=7)
        ax.tick_params(labelsize=7)
        lo, hi = np.percentile(chain[-1000:, :, idx], [0.1, 99.9])
        ax.set_ylim(lo, hi)
    for idx in range(ndim, nrows * ncols):
        r, c = divmod(idx, ncols)
        axs_trace[r, c].set_visible(False)

    # ---- Lower-left: posterior stability + KS annotations ----
    axs_stab = subfigs[1, 0].subplots(nrows, ncols, squeeze=False)
    subfigs[1, 0].suptitle("Posterior stability (2nd-half chunks)", fontsize=12)
    splits = np.array_split(flat_samples[len(flat_samples) // 2:], 4)
    for idx in range(ndim):
        r, c = divmod(idx, ncols)
        ax   = axs_stab[r, c]
        xs   = np.linspace(flat_samples[:, idx].min(), flat_samples[:, idx].max(), 200)
        for j, s in enumerate(splits):
            kde = gaussian_kde(s[:, idx])
            ax.plot(xs, kde(xs), alpha=0.6, label=f"chunk {j}")
        ax.set_title(labels[idx] if labels else f"param {idx}", fontsize=12)
        ax.tick_params(labelsize=7)
        if ks_chunks is not None and ks_tail is not None:
            ann = (f"KS-chunk p={ks_chunks[idx]:.3f}\n"
                   f"KS-tail  p={ks_tail[idx]:.3f}")
            ax.text(0.02, 0.97, ann,
                    transform=ax.transAxes, fontsize=6, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        ax.legend(fontsize=12, loc="upper right")
    for idx in range(ndim, nrows * ncols):
        r, c = divmod(idx, ncols)
        axs_stab[r, c].set_visible(False)

    # ---- Lower-right: tau history ----
    ax_tau = subfigs[1, 1].subplots(1, 1)
    subfigs[1, 1].suptitle("Autocorrelation time", fontsize=12)
    if tau_hist is not None and iter_hist is not None:
        tau_hist_arr  = np.array(tau_hist)   # (n_checkpoints, ndim)
        iter_hist_arr = np.array(iter_hist)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        max_taus = []
        for i in range(ndim):
            lbl = labels[i] if labels else f"param {i}"
            col = colors[i % len(colors)]
            ax_tau.plot(iter_hist_arr, tau_hist_arr[:, i], label=lbl, color=col)
            max_tau = tau_hist_arr[:, i].max()
            max_taus.append(max_tau)
            ax_tau.axhline(max_tau, color=col, linestyle=":", linewidth=1.0, alpha=0.7)

        ax_tau.plot(iter_hist_arr, iter_hist_arr / 50, "k--",
                    label=r"Convergence criterion (N = 50$\tau$)")
        ax_tau.set_xlabel("Iteration", fontsize=12)
        ax_tau.set_ylabel(r"$\tau$", fontsize=12)
        if not np.any(np.isnan(max_taus)) and not np.any(np.isinf(max_taus)):
            ax_tau.set_ylim(2, np.nanmax(max_taus) * 1.05)
        ax_tau.tick_params(labelsize=12)
        ax_tau.legend(fontsize=12)

    # ---- Save to HDF5 ----
    with h5py.File(filename, "a") as f:
        grp = f.require_group(f"figures_{nsteps}")
        buf_main = io.BytesIO()
        fig.savefig(buf_main, format="png", dpi=120, bbox_inches="tight")
        if "summary" in grp:
            del grp["summary"]
        grp.create_dataset("summary", data=np.frombuffer(buf_main.getvalue(), dtype="uint8"))

    plt.close(fig)
