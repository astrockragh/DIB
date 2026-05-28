#!/usr/bin/env python
"""
ame_ground_ground_fit.py

Draws posterior samples from 15272-Cs and 15672-Cs MCMC chains, simulates
the ground→ground pure-rotational spectrum (axis a and axis b) with PGOPHER
for each sample, applies tunable post-simulation broadening, averages the
spectra over samples, then fits a NNLS linear combination of the four mean
profiles (15272-a, 15272-b, 15672-a, 15672-b) to Planck AME data.

Posterior selection follows the same burn-in / finite-mask pattern as
emcee_analysis_joint_with_geometry_plots_update_units_update_alpha.ipynb.

Usage
-----
python ame_ground_ground_fit.py \
    --file_15272 /scratch/.../15272_..._Cs_....h5 \
    --file_15672 /scratch/.../15672_..._Cs_....h5 \
    --n_samples 20 \
    --gauss_broad_MHz 3000 \
    --lorentz_broad_MHz 0 \
    --savefig figs/ame_gg_fit.png

If --file_15272 / --file_15672 are omitted the script auto-selects the most
recently modified matching HDF5 in --scratch_dir.
"""

import argparse, glob, os, shutil, subprocess
import numpy as np
import emcee
from pathlib import Path
from scipy.optimize import nnls
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import matplotlib as mpl

REPO_ROOT      = Path(__file__).resolve().parent
PGO_TEMPLATE   = str(REPO_ROOT / 'pgo_files' / 'ground_ground.pgo')
PGO_BIN        = str(REPO_ROOT / 'pgo')
SCRATCH_DEFAULT = os.path.expanduser('~/../../scratch/gpfs/MELCHIOR/cj1223/DIB')
BURNIN_MULT    = 5   # discard first BURNIN_MULT * max(tau) steps

# ── Planck 2011 AME data (from Planck 2011 XI, Table 1) ──────────────────────
AME_DATASETS = {
    'Perseus Cloud': dict(
        nu=np.array([
            0.408, 0.82, 1.42, 10.9, 12.7, 14.7, 16.3, 22.8, 28.5, 33.0,
            40.9, 44.1, 61.3, 70.3, 93.8, 100, 143, 217, 353, 545, 857,
            1250, 2143, 2997
        ]),
        flux=np.array([
            0.2, 0.5, -0.5, 9.4, 13.3, 21.8, 29.3, 33.4, 31.5, 29.7,
            22.6, 19.6, 10.9, 8.9, 7, 10, -25, 120, -600, -800, -1000,
            -2400, 7000, -2000
        ]),
        err=np.array([
            3.4, 4.8, 1.8, 2.0, 2.4, 3.3, 4.2, 4.5, 4.4, 4.4,
            4.0, 4.0, 6.4, 9.0, 21, 29, 80, 320, 1400, 4900, 14000,
            28000, 35000, 20000
        ]),
    ),
    r'$\rho$ Ophiuchi Cloud': dict(
        nu=np.array([
            0.4, 1.4, 2.3, 22.7, 28.5, 33.0, 40.7, 44.1, 60.6, 70.3,
            93.4, 100, 143, 217, 353, 545, 857, 1249, 2141, 2997
        ]),
        flux=np.array([
            -7, -1.0, 1.2, 24.8, 27.3, 27.2, 21.9, 19.9, 9.8, 6.5,
            6, 13, 5, 130, -180, 400, 500, -5000, 20000, -9000
        ]),
        err=np.array([
            11, 8.3, 8.3, 6.6, 6.6, 6.3, 5.8, 5.7, 6.5, 7.8,
            15, 19, 49, 200, 780, 3200, 10000, 25000, 41000, 32000
        ]),
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--file_15272', default=None,
                   help='HDF5 backend for 15272 Cs (auto-selected if omitted)')
    p.add_argument('--file_15672', default=None,
                   help='HDF5 backend for 15672 Cs (auto-selected if omitted)')
    p.add_argument('--scratch_dir', default=SCRATCH_DEFAULT,
                   help='Scratch directory for auto-detection of backend files')
    p.add_argument('--n_samples', type=int, default=20,
                   help='Number of posterior samples per DIB (default 20)')
    p.add_argument('--gauss_broad_MHz', type=float, default=3000.0,
                   help='Additional Gaussian broadening FWHM [MHz] applied after simulation (default 3000)')
    p.add_argument('--lorentz_broad_MHz', type=float, default=0.0,
                   help='Additional Lorentzian broadening FWHM [MHz] applied after simulation (default 0)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--savefig', default=str(REPO_ROOT / 'figs' / 'ame_gg_fit.png'),
                   help='Output figure path')
    p.add_argument('--save_spectra', default=None,
                   help='If set, save mean spectra to this .npz path for inspection')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def find_backend(scratch_dir, dib, sym='Cs'):
    pattern = os.path.join(os.path.expanduser(scratch_dir), '*.h5')
    matches = [
        f for f in glob.glob(pattern)
        if f'{dib}_run_' in os.path.basename(f)
        and f'Symmetry{sym}' in os.path.basename(f)
        and 'temppy' not in os.path.basename(f)
    ]
    if not matches:
        raise FileNotFoundError(f'No {dib} Cs backend found in {scratch_dir}')
    best = sorted(matches, key=os.path.getmtime, reverse=True)[0]
    return best


def load_flat_samples(h5_path):
    h5_path = os.path.expanduser(h5_path)
    backend  = emcee.backends.HDFBackend(h5_path, read_only=True)
    chain    = backend.get_chain()       # (nsteps, nwalkers, ndim)
    log_prob = backend.get_log_prob()    # (nsteps, nwalkers)

    tau     = emcee.autocorr.integrated_time(chain, tol=0)
    burnin  = int(BURNIN_MULT * np.max(tau))
    burnin  = max(1, min(burnin, chain.shape[0] - 1))

    flat_samples  = chain[burnin:].reshape(-1, chain.shape[2])
    flat_log_prob = log_prob[burnin:].reshape(-1)

    finite_mask  = np.isfinite(flat_log_prob)
    flat_samples = flat_samples[finite_mask]

    print(f'    burn-in: {burnin} steps  (= {BURNIN_MULT} × max(τ) = {BURNIN_MULT:.0f} × {np.max(tau):.1f})')
    print(f'    flat samples available: {len(flat_samples)}')
    return flat_samples


# ─────────────────────────────────────────────────────────────────────────────
def _run_pgopher(T, A, B, C, axis, temp_dir):
    """Run PGOPHER for the ground→ground rotational spectrum.

    Returns (freq_MHz, intensity) as 1-D numpy arrays on the native grid.
    """
    tag      = f'T{T:.4f}_A{A:.7f}_B{B:.7f}_C{C:.7f}_ax{axis}'
    pgo_file = os.path.join(temp_dir, f'gg_{tag}.pgo')
    spec_txt = os.path.join(temp_dir, f'gg_{tag}.txt')

    # Patch the template: Temperature, A, B, C, and transition-moment axis.
    # The ground_ground.pgo has a single manifold so no inside/outside-manifold
    # guards are needed — just match the parameter names directly.
    awk = f"""awk \\
  -v temp="{T}" \\
  -v A_val="{A}" -v B_val="{B}" -v C_val="{C}" \\
  -v axis="{axis}" '
/<Parameter Name="Temperature" Value=/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\\\"" temp "\\\\"") }}
/<Parameter Name="A" Value=/           {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\\\"" A_val "\\\\"") }}
/<Parameter Name="B" Value=/           {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\\\"" B_val "\\\\"") }}
/<Parameter Name="C" Value=/           {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\\\"" C_val "\\\\"") }}
/<CartesianTransitionMoment Axis=/     {{ sub(/Axis="[^"]+"/, "Axis=\\\\"" axis "\\\\"") }}
{{ print }}
' {PGO_TEMPLATE} > {pgo_file}"""

    subprocess.run(awk, shell=True, check=True, executable='/bin/bash')
    subprocess.run([PGO_BIN, '--plot', pgo_file, spec_txt],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    data = np.loadtxt(spec_txt)
    os.remove(pgo_file)
    os.remove(spec_txt)

    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, 0].copy(), data[:, 1].copy()   # freq_MHz, intensity


def _apply_broadening(freq_MHz, intensity, gauss_fwhm_MHz, lorentz_fwhm_MHz):
    """Convolve a spectrum with an additional Gaussian and/or Lorentzian kernel.

    Both FWHMs are in MHz.  The input spectrum must be on a regular grid
    (which PGOPHER --plot always produces).
    """
    if gauss_fwhm_MHz <= 0 and lorentz_fwhm_MHz <= 0:
        return freq_MHz, intensity.copy()

    df = freq_MHz[1] - freq_MHz[0]   # grid spacing [MHz]
    spec = intensity.copy()

    if gauss_fwhm_MHz > 0:
        sigma = gauss_fwhm_MHz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        hw    = int(np.ceil(5.0 * sigma / df))
        t     = np.arange(-hw, hw + 1) * df
        kern  = np.exp(-t**2 / (2.0 * sigma**2))
        kern /= kern.sum()
        spec  = fftconvolve(spec, kern, mode='same')

    if lorentz_fwhm_MHz > 0:
        gamma = lorentz_fwhm_MHz / 2.0
        hw    = int(np.ceil(20.0 * gamma / df))
        t     = np.arange(-hw, hw + 1) * df
        kern  = (gamma / np.pi) / (t**2 + gamma**2)
        kern /= kern.sum()
        spec  = fftconvolve(spec, kern, mode='same')

    return freq_MHz, spec


def simulate_and_average(flat_samples, n_samples, dib_label, axes,
                         temp_dir, rng, gauss_fwhm_MHz, lorentz_fwhm_MHz):
    """Draw n_samples from flat_samples, simulate ground→ground spectra,
    apply broadening, and return the mean spectrum for each axis.

    Cs parameter vector (B_not_equal_C=True):
      [T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, (center_offset)]
    B_not_equal_C=False:
      [T, A, C, frac_A, frac_C, lorentz_width, ...]  where B = C
    """
    n_avail = len(flat_samples)
    if n_samples > n_avail:
        print(f'    Warning: requested {n_samples} samples but only {n_avail} available; using all.')
        n_samples = n_avail
    idx     = rng.choice(n_avail, size=n_samples, replace=False)
    samples = flat_samples[idx]

    ndim = samples.shape[1]
    # Detect B_not_equal_C from ndim (Cs without center-offset: 8 → True, 6 → False)
    b_neq_c = (ndim >= 8)

    # Accumulate spectra; common_freq set on first successful run
    common_freq = None
    accum = {ax: None for ax in axes}
    counts = {ax: 0 for ax in axes}

    for i, s in enumerate(samples):
        T = s[0]
        A = s[1]
        if b_neq_c:
            B, C = s[2], s[3]
        else:
            C = s[2]
            B = C   # B = C when B_not_equal_C=False

        print(f'  [{dib_label}] sample {i+1:3d}/{n_samples}: '
              f'T={T:.1f} K  A={A:.5f}  B={B:.5f}  C={C:.5f}', flush=True)

        for ax in axes:
            try:
                freq, spec = _run_pgopher(T, A, B, C, ax, temp_dir)
                freq, spec = _apply_broadening(freq, spec, gauss_fwhm_MHz, lorentz_fwhm_MHz)

                if common_freq is None:
                    common_freq = freq
                    for a in axes:
                        accum[a] = np.zeros_like(spec)

                # Interpolate onto the common grid (all PGOPHER runs use the
                # same Fmax / nDF so grids are identical; interp is a safety net)
                spec_on_grid = np.interp(common_freq, freq, spec, left=0.0, right=0.0)
                accum[ax]   += spec_on_grid
                counts[ax]  += 1

            except Exception as e:
                print(f'    ERROR [{dib_label}] axis={ax} sample {i+1}: {e}', flush=True)

    mean_spectra = {}
    for ax in axes:
        if counts[ax] > 0:
            mean_spectra[ax] = accum[ax] / counts[ax]
        else:
            mean_spectra[ax] = np.zeros_like(common_freq) if common_freq is not None else np.array([0.0])

    return common_freq, mean_spectra


# ─────────────────────────────────────────────────────────────────────────────
def build_model_matrix(profiles_GHz, nu_data_GHz):
    """Evaluate each mean profile at the AME data frequencies (GHz).

    profiles_GHz : list of (freq_GHz, mean_spec) tuples, one per template
    nu_data_GHz  : 1-D array of AME data frequencies [GHz]

    Returns (N_data × N_profiles) design matrix, normalised so each column
    peaks at 1 (same convention as AME_plot.ipynb).
    """
    M = np.zeros((len(nu_data_GHz), len(profiles_GHz)))
    for j, (freq_GHz, spec) in enumerate(profiles_GHz):
        peak = np.max(np.abs(spec))
        if peak == 0:
            continue
        spec_norm = spec / peak
        M[:, j]   = np.interp(nu_data_GHz, freq_GHz, spec_norm, left=0.0, right=0.0)
    return M


def fit_nnls(M, flux, err):
    Mw    = M    / err[:, None]
    fw    = flux / err
    amps, _ = nnls(Mw, fw)
    residuals = flux - M @ amps
    chi2 = np.sum((residuals / err)**2)
    return amps, chi2


# ─────────────────────────────────────────────────────────────────────────────
def plot_results(profiles_GHz, profile_labels, results, savefig):
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rc('font', family='serif', size=13)
    plt.rc('axes', linewidth=1.5)
    plt.rc('xtick', labelsize=13, direction='in', top=True)
    plt.rc('ytick', labelsize=13, direction='in', right=True)
    plt.rc('xtick.minor', visible=True)
    plt.rc('ytick.minor', visible=True)

    colors = ['tomato', 'firebrick', 'cornflowerblue', 'steelblue']
    ls_map = ['-', '--', '-', '--']
    nu_fine = np.logspace(np.log10(0.3), np.log10(200), 3000)

    n_ds = len(results)
    fig, axes = plt.subplots(2, n_ds, figsize=(9 * n_ds, 7), sharex=True,
                             gridspec_kw={'height_ratios': [6, 1], 'hspace': 0.0,
                                          'wspace': 0.22})
    if n_ds == 1:
        axes = axes[:, np.newaxis]

    for col, (ds_name, ds) in enumerate(results.items()):
        ax_top = axes[0, col]
        ax_res = axes[1, col]
        nu      = ds['nu']
        flux    = ds['flux']
        err     = ds['err']
        amps    = ds['amps']
        chi2    = ds['chi2']
        M_data  = ds['M_data']
        flux_model_at_data = M_data @ amps

        # Fine-grid model
        M_fine     = build_model_matrix(profiles_GHz, nu_fine)
        flux_model = M_fine @ amps

        dof = int(np.sum(np.logical_and(flux > 0,
                                        np.logical_and(nu > 5, nu < 200)))) \
              - int(np.sum(amps > 0.1))
        dof = max(dof, 1)

        ax_top.errorbar(nu, flux, yerr=err, fmt='o', color='k',
                        label='Data (Planck 2011)', ms=5, zorder=5)
        ax_top.plot(nu_fine, flux_model, c='purple', lw=2, label='Total fit')

        for j, (label, (freq_GHz, spec)) in enumerate(zip(profile_labels, profiles_GHz)):
            if amps[j] < 1e-1:
                continue
            peak = np.max(np.abs(spec))
            if peak == 0:
                continue
            spec_fine = np.interp(nu_fine, freq_GHz, spec / peak, left=0.0, right=0.0)
            ax_top.fill_between(nu_fine, amps[j] * spec_fine, alpha=0.3,
                                color=colors[j], label=label,
                                linestyle=ls_map[j])

        ax_top.set_xscale('log')
        ax_top.set_yscale('log')
        ax_top.set_xlim(5, 200)
        ax_top.set_ylim(0.4, 50)
        ax_top.set_ylabel('Flux (Jy)')
        ax_top.set_title(ds_name)
        ax_top.legend(fontsize=10, ncol=2, loc='lower left', framealpha=0.9)
        ax_top.text(0.03, 0.97,
                    rf'$\chi^2 = {chi2:.1f}$' + '\n'
                    rf'dof $= {dof}$' + '\n'
                    rf'$\chi^2_\nu = {chi2/dof:.1f}$',
                    transform=ax_top.transAxes, fontsize=12,
                    ha='left', va='top')

        ax_res.scatter(nu, (flux - flux_model_at_data) / err,
                       color='k', s=20, zorder=5)
        ax_res.axhline(0, color='gray', lw=1)
        ax_res.set_xlabel('Frequency (GHz)')
        ax_res.set_ylabel(r'$\chi$')
        m = max(np.max(np.abs(ax_res.get_ylim())), 1)
        ax_res.set_ylim(-1.3 * m, 1.3 * m)

    plt.suptitle('Ground–Ground Rotational Profiles vs. AME\n'
                 '(posterior-averaged, 15272 & 15672 Cs)',
                 y=0.98, fontsize=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(savefig)), exist_ok=True)
    plt.savefig(savefig, dpi=200, bbox_inches='tight')
    print(f'\nFigure saved → {savefig}')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    # ── Locate backend files ──────────────────────────────────────────────────
    file_15272 = args.file_15272 or find_backend(args.scratch_dir, '15272')
    file_15672 = args.file_15672 or find_backend(args.scratch_dir, '15672')
    print(f'15272 backend: {os.path.basename(file_15272)}')
    print(f'15672 backend: {os.path.basename(file_15672)}')

    # ── Load posteriors ───────────────────────────────────────────────────────
    print('\nLoading 15272 Cs posterior...')
    flat_15272 = load_flat_samples(file_15272)
    print('Loading 15672 Cs posterior...')
    flat_15672 = load_flat_samples(file_15672)

    # ── PGOPHER temp directory ────────────────────────────────────────────────
    temp_dir = os.path.expanduser(
        '~/../../scratch/gpfs/MELCHIOR/cj1223/DIB/pgo_temppy_ame_gg')
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir)

    axes = ['a', 'b']

    # ── Simulate spectra ──────────────────────────────────────────────────────
    print(f'\nSimulating 15272 ground→ground ({args.n_samples} samples × 2 axes)...')
    freq_15272, specs_15272 = simulate_and_average(
        flat_15272, args.n_samples, '15272', axes, temp_dir, rng,
        args.gauss_broad_MHz, args.lorentz_broad_MHz)

    print(f'\nSimulating 15672 ground→ground ({args.n_samples} samples × 2 axes)...')
    freq_15672, specs_15672 = simulate_and_average(
        flat_15672, args.n_samples, '15672', axes, temp_dir, rng,
        args.gauss_broad_MHz, args.lorentz_broad_MHz)

    shutil.rmtree(temp_dir, ignore_errors=True)

    # Convert native MHz grid → GHz
    freq_GHz_15272 = freq_15272 / 1000.0
    freq_GHz_15672 = freq_15672 / 1000.0

    # 4 profiles: (15272-a, 15272-b, 15672-a, 15672-b)
    profiles_GHz = [
        (freq_GHz_15272, specs_15272['a']),
        (freq_GHz_15272, specs_15272['b']),
        (freq_GHz_15672, specs_15672['a']),
        (freq_GHz_15672, specs_15672['b']),
    ]
    profile_labels = ['15272  a-axis', '15272  b-axis',
                      '15672  a-axis', '15672  b-axis']

    # ── Optional: save spectra ────────────────────────────────────────────────
    if args.save_spectra:
        np.savez(args.save_spectra,
                 freq_GHz_15272=freq_GHz_15272,
                 freq_GHz_15672=freq_GHz_15672,
                 spec_15272_a=specs_15272['a'], spec_15272_b=specs_15272['b'],
                 spec_15672_a=specs_15672['a'], spec_15672_b=specs_15672['b'])
        print(f'Spectra saved → {args.save_spectra}')

    # ── NNLS fit to each AME dataset ──────────────────────────────────────────
    print('\nFitting NNLS to AME data...')
    results = {}
    for ds_name, ds in AME_DATASETS.items():
        M_data = build_model_matrix(profiles_GHz, ds['nu'])
        amps, chi2 = fit_nnls(M_data, ds['flux'], ds['err'])
        results[ds_name] = dict(nu=ds['nu'], flux=ds['flux'], err=ds['err'],
                                amps=amps, chi2=chi2, M_data=M_data)

        print(f'\n  {ds_name}')
        print(f'  {"Profile":<20} {"Amplitude":>10}')
        print('  ' + '-' * 32)
        for lbl, amp in zip(profile_labels, amps):
            print(f'  {lbl:<20} {amp:10.3f}')
        print(f'  chi2 = {chi2:.2f}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(profiles_GHz, profile_labels, results, args.savefig)


if __name__ == '__main__':
    main()
