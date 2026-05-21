#!/usr/bin/env python
"""
grid_search_C2v_15272.py

Coarse grid evaluation of chi^2 for DIB 15272, C2v symmetry.

User-facing rotational constant convention (A is the dominant C2-axis constant):
    pgopher_C (largest) = A
    pgopher_B           = B_f * A
    pgopher_A (smallest)= C_f * A

Fixed parameters:
    frac_A = frac_B = frac_C = 0.99
    lorentz_width (tau)       = 0.7 cm^-1
    center_offset              = 0.0 cm^-1

Grids:
    T    = [5, 10, 15, 20]
    A    = [0.001, 0.003, 0.01, 0.03, 0.1]
    B_f  = [0.9, 0.5, 0.1]
    C_f  = [0.9, 0.5, 0.1]

Combinations where B_f <= C_f violate pgopher_B > pgopher_A and are skipped.

Saves chi2_direct, chi2_dT, chi2_total and the grid coords to
grid_search_C2v_15272.npz in the repo root.
"""

import os, shutil, subprocess, itertools
import numpy as np
import h5py
from pathlib import Path
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parent

# ── Constants ──────────────────────────────────────────────────────────────────
CENTRAL_INVCM = 1e8 / 15272.27178113337   # ~6544.44 cm^-1
LSF_FILE     = str(REPO_ROOT / 'LSFs'     / 'lsf_15272.h5')
DATA_FILE    = str(REPO_ROOT / 'all_errs' / 'res_dib_15272.h5')
PGO_TEMPLATE = str(REPO_ROOT / 'pgo_files' / 'asym_top_15272_C2v.pgo')
PGO_BIN      = str(REPO_ROOT / 'pgo')

CROP  = 15    # default err_model crop for 15272
FUDGE = 1.0   # noise inflation on direct spectrum

FRAC_A = FRAC_B = FRAC_C = 0.99
TAU       = 0.7
DELTA_CEN = 0.0

# ── Grid definition ────────────────────────────────────────────────────────────
T_grid  = [5, 10, 15, 20]
A_grid  = [0.001, 0.003, 0.01, 0.03, 0.1]
Bf_grid = [0.9, 0.5, 0.1]
Cf_grid = [0.9, 0.5, 0.1]

# ── Temp directory ─────────────────────────────────────────────────────────────
TEMP_DIR = os.path.expanduser(
    '~/../../scratch/gpfs/MELCHIOR/cj1223/DIB/pgo_temppy_grid_C2v_15272')
shutil.rmtree(TEMP_DIR, ignore_errors=True)
os.makedirs(TEMP_DIR)

# ── Data loading ───────────────────────────────────────────────────────────────
with h5py.File(DATA_FILE, 'r') as df:
    data_wavelength = df['wav'][:]
    data_flux       = df['mean'][:, 0]
    data_flux_dT    = df['mean'][:, 1]
    noise_std       = FUDGE * np.sqrt(df['var'][:, 0])
    noise_std_dT    =         np.sqrt(df['var'][:, 1])

with h5py.File(LSF_FILE, 'r') as f:
    wav_grid = f['wav'][:]

# ── PGOPHER call ───────────────────────────────────────────────────────────────
def run_pgopher(T, pgopher_A, pgopher_B, pgopher_C):
    """
    Run PGOPHER for C2v 15272 and return LSF-convolved flux on the data wavelength grid.

    Parameters follow PGOPHER C2v 15272 convention: pgopher_C > pgopher_B > pgopher_A.
    The excited-state constants are scaled by FRAC_* (fixed at 0.99).
    """
    A_e = pgopher_A * FRAC_A
    B_e = pgopher_B * FRAC_B
    C_e = pgopher_C * FRAC_C

    tag = (f"T{T:.4f}_A{pgopher_A:.7f}_B{pgopher_B:.7f}_C{pgopher_C:.7f}")
    pgo_file = os.path.join(TEMP_DIR, f"temp_{tag}.pgo")
    spec_txt = os.path.join(TEMP_DIR, f"spec_{tag}.txt")

    awk = f'''awk \
  -v temp="{T}" \
  -v A_ground="{pgopher_A}" -v B_ground="{pgopher_B}" -v C_ground="{pgopher_C}" \
  -v A_excited="{A_e}"      -v B_excited="{B_e}"      -v C_excited="{C_e}" \
  -v lorentz_width="{TAU}" '
BEGIN {{ in_ground=0; in_excited=0; }}
/<AsymmetricTop Name="v=0"/  {{ in_ground=1; }}
/<AsymmetricTop Name="v=1"/  {{ in_excited=1; }}
/<\\/AsymmetricTop>/          {{ in_ground=0; in_excited=0; }}
in_ground  && /<Parameter Name="A" Value=/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" A_ground "\\"") }}
in_ground  && /<Parameter Name="B" Value=/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" B_ground "\\"") }}
in_ground  && /<Parameter Name="C" Value=/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" C_ground "\\"") }}
in_excited && /<Parameter Name="A" Value=/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" A_excited "\\"") }}
in_excited && /<Parameter Name="B" Value=/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" B_excited "\\"") }}
in_excited && /<Parameter Name="C" Value=/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" C_excited "\\"") }}
/<CartesianTransitionMoment Bra="v=1" Ket="v=0"/ {{ sub(/Axis="[^"]+"/, "Axis=\\"a\\"") }}
/<Parameter Name="Temperature" Value=/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" temp "\\"") }}
/<Parameter Name="Lorentzian"  Value=/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" lorentz_width "\\"") }}
{{ print }}
' {PGO_TEMPLATE} > {pgo_file}'''

    subprocess.run(awk, shell=True, check=True, executable='/bin/bash')
    subprocess.run([PGO_BIN, '--plot', pgo_file, spec_txt],
                   check=True, stdout=subprocess.DEVNULL)

    inv_cm, flux = np.loadtxt(spec_txt).T
    wav_pgo = 1e8 / inv_cm

    # Convolve with three-Gaussian APOGEE LSF
    wavc   = 1e8 / CENTRAL_INVCM
    window, dlam = 8.0, 0.01
    wav_reg = np.arange(wavc - window, wavc + window, dlam)
    flux_reg = interp1d(wav_pgo, flux, bounds_error=False, fill_value=0.0)(wav_reg)

    sig1, sig2, sig3 = 0.3, 1.85 * 0.3, 9.5 * 0.3
    f1, f2, c0 = 0.895, 0.1, 1.3e-3
    rel = wav_reg - wavc
    lsf  = (f1 / np.sqrt(2*np.pi*sig1**2) * np.exp(-rel**2 / (2*sig1**2))
           + f2 / np.sqrt(2*np.pi*sig2**2) * np.exp(-rel**2 / (2*sig2**2))
           + (1-f1-f2) / np.sqrt(2*np.pi*sig3**2) * np.exp(-rel**2 / (2*sig3**2))
           + c0)
    lsf /= lsf.sum()
    convolved = convolve(flux_reg, lsf, mode='same')
    flux_on_grid = interp1d(wav_reg, convolved, bounds_error=False, fill_value=0.0)(wav_grid)

    os.remove(spec_txt)
    os.remove(pgo_file)
    return flux_on_grid


# ── chi^2 evaluation ───────────────────────────────────────────────────────────
def eval_chi2(flux, flux_dT):
    """
    Given model spectra at T and T+0.05, compute chi2_direct and chi2_dT.

    Direct fit: gamma * spec + offset
    dT fit    : alpha * spec + beta * (spec_dT - spec) + offset_dT
    """
    c  = CROP
    gf = 0.01   # mild Gaussian filter (matches MCMC code)

    spec = gaussian_filter(flux[c:-c], gf)
    meas = data_flux[c:-c]
    ns   = noise_std[c:-c]

    M = np.vstack([spec, np.ones_like(spec)]).T
    gamma, off = np.linalg.lstsq(M, meas, rcond=None)[0]
    chi2_direct = float(np.sum(((meas - gamma * spec - off) / ns) ** 2))

    spec_dT   = gaussian_filter(flux_dT[c:-c], gf) - spec
    meas_dT   = data_flux_dT[c:-c]
    ns_dT     = noise_std_dT[c:-c]
    M_dT      = np.vstack([spec, spec_dT, np.ones_like(spec)]).T
    a, b, o   = np.linalg.lstsq(M_dT, meas_dT, rcond=None)[0]
    chi2_dT   = float(np.sum(((meas_dT - a * spec - b * spec_dT - o) / ns_dT) ** 2))

    return chi2_direct, chi2_dT


# ── Grid loop ──────────────────────────────────────────────────────────────────
rows  = []   # list of (T, A, Bf, Cf, chi2_direct, chi2_dT, chi2_total)
total = len(T_grid) * len(A_grid) * len(Bf_grid) * len(Cf_grid)
done  = 0

print(f"Grid search: {total} combinations "
      f"({len(T_grid)} T × {len(A_grid)} A × {len(Bf_grid)} Bf × {len(Cf_grid)} Cf)\n")

for T, A, Bf, Cf in itertools.product(T_grid, A_grid, Bf_grid, Cf_grid):
    done += 1
    tag = f"T={T:4.1f} A={A:.3f} Bf={Bf:.1f} Cf={Cf:.1f}"

    # Map to PGOPHER C2v convention: pgopher_C (largest) = A
    pgopher_C = A
    pgopher_B = Bf * A
    pgopher_A = Cf * A

    if not (pgopher_C > pgopher_B > pgopher_A):
        print(f"[{done:3d}/{total}] {tag}  → skipped (Bf <= Cf violates hierarchy)")
        continue

    try:
        flux    = run_pgopher(T,        pgopher_A, pgopher_B, pgopher_C)
        flux_dT = run_pgopher(T + 0.05, pgopher_A, pgopher_B, pgopher_C)
        c2s, c2dT = eval_chi2(flux, flux_dT)
        c2tot = c2s + c2dT
        rows.append((T, A, Bf, Cf, c2s, c2dT, c2tot))
        print(f"[{done:3d}/{total}] {tag}  "
              f"chi2_direct={c2s:8.1f}  chi2_dT={c2dT:8.1f}  chi2_total={c2tot:8.1f}")
    except Exception as e:
        print(f"[{done:3d}/{total}] {tag}  → ERROR: {e}")

# ── Save ───────────────────────────────────────────────────────────────────────
if rows:
    arr = np.array(rows, dtype=float)
    out = str(REPO_ROOT / 'grid_search_C2v_15272.npz')
    np.savez(out,
             T=arr[:, 0], A=arr[:, 1], Bf=arr[:, 2], Cf=arr[:, 3],
             chi2_direct=arr[:, 4], chi2_dT=arr[:, 5], chi2_total=arr[:, 6])
    print(f"\nSaved {len(rows)} results → {out}")

    best = np.argsort(arr[:, 6])[:5]
    print("\nTop 5 by chi2_total:")
    for i in best:
        T, A, Bf, Cf, c2s, c2dT, c2tot = arr[i]
        print(f"  T={T:.1f}  A={A:.3f}  Bf={Bf:.1f}  Cf={Cf:.1f}  "
              f"chi2_direct={c2s:.1f}  chi2_dT={c2dT:.1f}  chi2_total={c2tot:.1f}")
else:
    print("No valid results to save.")

shutil.rmtree(TEMP_DIR, ignore_errors=True)
