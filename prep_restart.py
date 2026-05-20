#!/usr/bin/env python
"""
prep_restart.py

Prepare an emcee HDF5 chain for use as a restart file by:
  1. Optionally swapping pairs of parameter indices in the chain.
  2. Optionally appending a new dimension sampled from N(loc, scale).

Output is saved as <input_name>_prep_restart.h5.

Examples
--------
# Swap dims 1 and 3:
python prep_restart.py run.h5 --swap_dims 1 3

# Swap dims 1↔3 and 4↔5, then add a new dimension:
python prep_restart.py run.h5 --swap_dims 1 3 4 5 --add_dim

# Add a new dim only, with custom distribution:
python prep_restart.py run.h5 --add_dim --new_dim_loc 0.01 --new_dim_scale 0.005
"""

import argparse, os, shutil
import numpy as np
import h5py


def parse_args():
    p = argparse.ArgumentParser(description="Prepare an emcee HDF5 chain for restart")
    p.add_argument("input",
                   help="Path to input HDF5 file")
    p.add_argument("--swap_dims", type=int, nargs='+', metavar='IDX',
                   default=None,
                   help="Pairs of indices to swap. e.g. --swap_dims 1 3 swaps dims 1 and 3; "
                        "--swap_dims 1 3 4 5 swaps 1↔3 and 4↔5.")
    p.add_argument("--add_dim", action='store_true',
                   help="Append a new dimension to the chain, sampled from N(new_dim_loc, new_dim_scale).")
    p.add_argument("--new_dim_loc", type=float, default=0.0,
                   help="Mean of the new dimension distribution (default: 0.0)")
    p.add_argument("--new_dim_scale", type=float, default=0.01,
                   help="Std of the new dimension distribution (default: 0.01)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for the new dimension samples (default: None)")
    return p.parse_args()


args = parse_args()

# ── Validate ──────────────────────────────────────────────────────────────────
src = os.path.abspath(os.path.expanduser(args.input))
assert os.path.exists(src), f"Input file not found: {src}"

if args.swap_dims is not None:
    assert len(args.swap_dims) % 2 == 0, \
        "--swap_dims must be an even number of indices (pairs), e.g. --swap_dims 1 3"
    swap_pairs = [(args.swap_dims[i], args.swap_dims[i + 1])
                  for i in range(0, len(args.swap_dims), 2)]
else:
    swap_pairs = []

assert args.swap_dims is not None or args.add_dim, \
    "Nothing to do: pass --swap_dims and/or --add_dim"

# ── Output path ───────────────────────────────────────────────────────────────
base, ext = os.path.splitext(src)
dst = base + '_prep_restart' + ext
assert not os.path.exists(dst), f"Output already exists: {dst}"

print(f"Input:  {src}")
print(f"Output: {dst}")

# ── Copy then modify ──────────────────────────────────────────────────────────
shutil.copy2(src, dst)
print("Copied.")

if args.seed is not None:
    np.random.seed(args.seed)

with h5py.File(dst, 'r+') as f:
    chain = f['mcmc/chain'][:]           # (nsteps, nwalkers, ndim)
    nsteps, nwalkers, ndim = chain.shape
    print(f"Chain shape: {chain.shape}")

    # ── Swap pairs ────────────────────────────────────────────────────────────
    for i, j in swap_pairs:
        assert 0 <= i < ndim and 0 <= j < ndim, \
            f"Swap index out of range: ({i}, {j}) — ndim={ndim}"
        chain[:, :, [i, j]] = chain[:, :, [j, i]]
        print(f"  Swapped dim {i} ↔ dim {j}")

    # ── Add new dimension ─────────────────────────────────────────────────────
    if args.add_dim:
        new_col = np.random.normal(
            loc=args.new_dim_loc,
            scale=args.new_dim_scale,
            size=(nsteps, nwalkers, 1),
        )
        chain = np.concatenate([chain, new_col], axis=2)
        print(f"  Appended dim {ndim} ~ N(loc={args.new_dim_loc}, scale={args.new_dim_scale})")
        print(f"  New chain shape: {chain.shape}")

    # ── Write back ────────────────────────────────────────────────────────────
    del f['mcmc/chain']
    f.create_dataset('mcmc/chain', data=chain)

print(f"Done → {os.path.basename(dst)}")
