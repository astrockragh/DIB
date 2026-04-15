"""
submit_grid.py — Generate and submit a grid of SLURM jobs for run_emcee_asym_both_newerrs.py.

Usage:
    python submit_grid.py [--dry-run] [--outdir OUTDIR]

Options:
    --dry-run       Write slurm scripts but do not submit them.
    --outdir DIR    Directory to write slurm scripts into (default: slurm_grid/<timestamp>).
"""

import argparse
import itertools
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Time limits ───────────────────────────────────────────────────────────────
TIME_C2V = "2-23:59:59"   # just under 3 days (C2v converges faster, fewer free params)
TIME_CS  = "3-23:59:59"   # just under 4 days (Cs has more walkers / slower mixing)

# ── Cluster / SBATCH constants ─────────────────────────────────────────────────
SBATCH_DEFAULTS = {
    "nodes":         1,       # always a single node — emcee uses multiprocessing, not MPI
    "account":       "spergel",  # billing account on Della
    "cpus-per-task": 64,      # number of CPU cores; emcee parallelises over walkers with Pool(64)
    "mem-per-cpu":   "2G",    # per-core memory; 64 × 2G = 128 GB total per node
    "mail-type":     "all",   # send email on job start, end, and failure
    "mail-user":     "cj1223@princeton.edu",  # address for those notifications
}

MODULE_LINES = """\
module purge
module load anaconda3/2022.5
source activate torch-env"""

PYTHON_SCRIPT = "$HOME/DIB/run_emcee_asym_both_newerrs.py"

# ── Baseline keyword values (applied to every job) ───────────────────────────
# Every argument accepted by run_emcee_asym_both_newerrs.py must appear here
# OR in GRID (grid values override baseline for that run). Keeping all keys
# explicit means the generated slurm scripts are fully self-documenting.
BASELINE = {
    # nsteps: total MCMC steps per walker; set per symmetry group via NSTEPS below
    # (populated dynamically in build_slurm, not hardcoded here)

    "fudge":                 1,     # multiplicative inflation of the DIB profile flux noise only
                                    # (1 = no inflation; higher = softer likelihood on the profile)

    "use_direct":            1,     # 1 = use directly measured profile/dT; 0 = use PCA reconstruction

    "old_errs":              0,     # 1 = use Andrew's original overly tight jackknife errors (jackknife_dib.h5)
                                    # mutually exclusive with plate_errs

    "plate_errs":            0,     # 1 = use errors jackknifed over plates (jackknife_plates_*.h5), 0 = use our preferred errors, 
                                    # which are jackknifed over plates in a different way that should take into account more sources of uncertainty
                                    # mutually exclusive with old_errs

    "flat_prior":            0,     # 1 = flat (uniform) priors on all parameters;
                                    # 0 = use the exponential/gaussian physics-motivated priors

    "fit_spec":              1,     # 1 = include the main DIB absorption profile in the likelihood

    "fit_dT":                1,     # 1 = include the dT (temperature-derivative) profile in the likelihood

    "cov":                   0,     # 1 = use the full pixel-pixel covariance matrix;
                                    # 0 = treat pixels as independent (diagonal covariance)

    "nonlinear_fit":         0,     # 1 = fit the dipole ratio between B/C axis with a non-linear optimiser (L-BFGS-B);
                                    # 0 = do so with a linear optimiser. Only relevant for Cs runs 

    "use_scalar_prior":      0,     # 1 = place a prior on the B/C ratio for dT instead of fitting both profile and dT directly

    "tau_prior":             0.05,  # slope of the exponential prior on lifetime/velocity broadening τ [cm⁻¹];
                                    # larger = softer (allows broader lines)

    "alpha_prior":           0.05,  # std of the half-Gaussian prior on the rotational-vibrational coupling coefficients
                                    # larger = allows stronger coupling,which tends to make the profile more assymmetric and heavy[tailed]
                                    # (overridden per DIB via DIB_OVERRIDES)

    "extra_truncation":      0,     # additional pixels to crop from each edge of the data beyond
                                    # the model's default crop (useful to exclude noisy wings)

    "balance_errs":          0,     # 1 = rescale errors so the relative uncertainty on the DIB profile
                                    # and the dT profile are equal, preventing one from dominating

    "err_factor":            1.0,   # divide both noise_std and noise_std_dT by this factor;
                                    # >1 tightens the likelihood (useful for sensitivity tests)

    "fit_peakcenter_offset": 0.01,  # Gaussian prior width [cm⁻¹] on the peak centre offset from the
                                    # nominal rest wavelength; False disables the offset entirely
                                    # (overridden per DIB via DIB_OVERRIDES)

    "title":                 "sweep",    # arbitrary string appended to the output directory name
                                    # for human-readable run labelling
}

# ── Per-symmetry-group overrides ──────────────────────────────────────────────
# These are applied on top of BASELINE before the grid combo, so they can be
# further overridden by GRID entries if needed.
NSTEPS = {
    "C2v": 20000,  # C2v is faster per-step because it has only one dipole moment
    "Cs":  25000,  # Cs has a large space due to the two dipole moments; needs more steps to converge
}

# ── Per-DIB fixed overrides ────────────────────────────────────────────────────
# Applied after the grid combo so these values are never swept — they are
# physically motivated and fixed per molecule.
DIB_OVERRIDES = {
    "15272": {"alpha_prior": 0.05, "fit_peakcenter_offset": 0.01},
    "15672": {"alpha_prior": 0.15, "fit_peakcenter_offset": 0.10},
}

# ── Grid axes (each key maps to a list of values to sweep over) ───────────────
# The full Cartesian product of all axes is generated, then crossed with every
# (DIB, symmetry_group) combination. Keys here override BASELINE for that run.
GRID = {
    "tau_prior":     [0.05, 0.15],  # sweep soft vs. tight lifetime-broadening prior
    "plate_errs": [0, 1],         # 0 = do not use plate-jackknife errors; 1 = use smaller plate-jackknife errors 
    "balance_errs": [0, 1],         # 0 = do not balance errors; 1 = balance errors between profile and dT
    # "use_direct": [0, 1],        # 0 = pca reconstruction; 1 = direct measurement (only relevant for dT, since profile is always direct)
    "extra_truncation": [0, 10],  # additional pixels to crop from each edge of the data beyond the model's default crop
    # "fit_dT": [0, 1],          # 0 = do not fit dT; 1 = include dT in the fit (only relevant for Cs, since dT is not available for C2v)
    "emphasize_dT_center": [False, True],  # False = disabled; True = look up width/factor/offset
                                           # from emphasize_dT_centers[dib][err_model] below
}

# Which DIBs and symmetry groups to run
# DIBS            = ["15272", "15672"]  # the two DIBs being fitted (rest wavelengths in Å)
DIBS            = ["15672"]  # the two DIBs being fitted (rest wavelengths in Å)
# DIBS            = ["15272"]  # the two DIBs being fitted (rest wavelengths in Å)
SYMMETRY_GROUPS = ["Cs", "C2v"]      # molecular symmetry; controls which band types are active

emphasize_dT_centers = {'15272': {
    'old':         None,
    'plate':       {'width': 10, 'factor': 3, 'offset': 2},
    'default':     {'width': 10, 'factor': 3, 'offset': 2},
    'pca':         {'width': 20, 'factor': 2, 'offset': 0},
    'pca_default': {'width': 20, 'factor': 2, 'offset': 0}},
                       '15672': {
    'old':         None,
    'plate':       {'width': 10, 'factor': 3, 'offset': -2},
    'default':     {'width': 10, 'factor': 3, 'offset': -2},
    'pca':         {'width': 10, 'factor': 3, 'offset': 0},
    'pca_default': {'width': 10, 'factor': 3, 'offset': 0}}}  

# ── Short inline comments for every argument (written into generated scripts) ─
PARAM_COMMENTS = {
    "dib":                   "target DIB rest wavelength [Å]",
    "symmetry_group":        "molecular symmetry (Cs or C2v)",
    "nsteps":                "MCMC steps per walker",
    "B_not_equal_C":         "0=symmetric top (B=C); 1=asymmetric top",
    "fudge":                 "flux noise inflation factor (1=none)",
    "use_direct":            "1=direct dT; 0=PCA reconstruction",
    "old_errs":              "use original jackknife errors",
    "plate_errs":            "use plate-jackknife errors",
    "flat_prior":            "0=physics priors; 1=flat/uniform priors",
    "fit_spec":              "include DIB absorption profile in fit",
    "fit_dT":                "include dT profile in fit",
    "cov":                   "0=diagonal covariance; 1=full pixel-pixel",
    "nonlinear_fit":         "nonlinear B/C dipole ratio optimiser (Cs only)",
    "use_scalar_prior":      "prior on B/C scalar ratio for dT",
    "tau_prior":             "exp prior slope on lifetime broadening [/cm]",
    "alpha_prior":           "Gaussian prior width on vib-rot coupling",
    "extra_truncation":      "extra pixels cropped from each data edge",
    "balance_errs":          "equalise relative DIB/dT uncertainties",
    "err_factor":            "noise divisor (>1 tightens likelihood)",
    "fit_peakcenter_offset": "Gaussian prior width on peak centre [/cm]",
    "emphasize_dT_center":   "WIDTH FACTOR OFFSET of dT centre emphasis window",
    "title":                 "label appended to output directory name",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_err_model(params):
    """Mirror the err_model derivation in run_emcee_asym_both_newerrs.py."""
    if params.get("old_errs", 0):
        return "old"
    elif params.get("use_direct", 1) and params.get("plate_errs", 0):
        return "plate"
    elif params.get("use_direct", 1):
        return "default"
    elif params.get("plate_errs", 0):
        return "pca"
    else:
        return "pca_default"


def make_job_name(dib, sym, combo):
    """Build a short, unique SLURM job name from the run parameters."""
    parts = [f"dib{dib}", sym]
    for k, v in combo.items():
        short_k = k.replace("_prior", "").replace("_", "")
        parts.append(f"{short_k}{v}")
    return "_".join(parts)


def format_arg(v):
    """Format a Python value as a shell argument string.
    Lists become space-separated tokens (for argparse nargs='+' arguments)."""
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v)


def build_slurm(job_name, dib, sym, params, time_limit):
    """Assemble and return the full text of a SLURM batch script."""
    sbatch_lines = [f"#SBATCH --job-name={job_name}"]
    for k, v in SBATCH_DEFAULTS.items():
        sbatch_lines.append(f"#SBATCH --{k}={v}")
    sbatch_lines.append(f"#SBATCH --time={time_limit}")

    # Merge: baseline → symmetry nsteps → grid combo → per-DIB overrides → dib/sym fixed args
    all_params = dict(BASELINE)
    all_params["nsteps"] = NSTEPS[sym]
    all_params.update(params)
    all_params.update(DIB_OVERRIDES.get(dib, {}))  # per-DIB values always win over grid
    all_params["dib"]            = dib
    all_params["symmetry_group"] = sym

    # Resolve emphasize_dT_center: True → look up width/factor/offset for this DIB+err_model;
    # False → omit the argument entirely (script default is None/disabled).
    emphasize = all_params.pop("emphasize_dT_center", False)
    if emphasize:
        err_model = get_err_model(all_params)
        spec = emphasize_dT_centers[dib].get(err_model)
        if spec is not None:
            all_params["emphasize_dT_center"] = [spec["width"], spec["factor"], spec["offset"]]
        # if spec is None (e.g. old errors have no centre emphasis), leave the arg omitted

    # Align argument columns for readability in both the comment header and the command.
    # NOTE: inline # comments after \ continuations are not valid bash (# swallows the \),
    # so comments live in a header block above the python command.
    max_klen = max(len(k) for k in all_params)
    max_vlen = max(len(format_arg(v)) for v in all_params.values())

    comment_lines = ["# Parameters:"]
    arg_lines     = []
    for k, v in all_params.items():
        padded_k = f"--{k}".ljust(max_klen + 3)
        padded_v = format_arg(v).ljust(max_vlen)
        comment  = PARAM_COMMENTS.get(k, "")
        comment_lines.append(f"#   {padded_k} {padded_v}  {comment}")
        arg_lines.append(f"    {padded_k} {format_arg(v)}")

    python_block = (
        "\n".join(comment_lines) + "\n\n"
        + f"SCRIPT={PYTHON_SCRIPT}\n\n"
        + "python $SCRIPT \\\n"
        + " \\\n".join(arg_lines)
    )

    return "\n".join([
        "#!/bin/bash",
        *sbatch_lines,
        "",
        MODULE_LINES,
        "",
        python_block,
        "",
    ])


def main():
    ap = argparse.ArgumentParser(description="Generate and submit a SLURM grid.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write scripts but do not call sbatch.")
    ap.add_argument("--outdir", default=None,
                    help="Directory for generated slurm scripts.")
    ap.add_argument("--N", type=int, default=None, metavar="N",
                    help="Randomly sample N jobs from the full grid. If omitted or larger than "
                         "the total number of jobs, all jobs are run.")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else Path(f"slurm_grid/{timestamp}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing slurm scripts to: {outdir}")

    # Build full Cartesian product of grid axes
    grid_keys   = list(GRID.keys())
    grid_values = list(GRID.values())
    combos      = [dict(zip(grid_keys, vals)) for vals in itertools.product(*grid_values)]

    # Flatten all (dib, sym, combo) triples, then optionally subsample
    all_jobs = [
        (dib, sym, combo)
        for dib in DIBS
        for sym in SYMMETRY_GROUPS
        for combo in combos
    ]
    total = len(all_jobs)
    if args.N is not None and args.N < total:
        all_jobs = random.sample(all_jobs, args.N)
        print(f"Sampling {args.N} of {total} total jobs.")
    else:
        print(f"Running all {total} jobs.")

    submitted = 0
    for dib, sym, combo in all_jobs:
        time_limit  = TIME_C2V if sym == "C2v" else TIME_CS
        job_name    = make_job_name(dib, sym, combo)
        script_txt  = build_slurm(job_name, dib, sym, combo, time_limit)
        script_path = outdir / f"{job_name}.slurm"
        script_path.write_text(script_txt)

        if args.dry_run:
            print(f"  [dry-run] {script_path}")
        else:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  submitted {script_path.name}: {result.stdout.strip()}")
            else:
                print(f"  FAILED {script_path.name}: {result.stderr.strip()}",
                      file=sys.stderr)
        submitted += 1

    noun = "scripts written" if args.dry_run else "jobs submitted"
    print(f"\nDone — {submitted} {noun}.")


if __name__ == "__main__":
    main()
