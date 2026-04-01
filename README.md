# Constraints on the Geometric Structure and Physical Parameters of two DIB Carriers in APOGEE

**Christian Kragh Jespersen & Andrew K. Saydjari** (Princeton University)

*Code accompanying Jespersen & Saydjari (in prep.)*

---

## Overview

The diffuse interstellar bands (DIBs) are among the oldest unsolved problems in astrophysics: over 500 absorption features imprinted on stellar spectra by interstellar molecules whose identities remain largely unknown. This repository contains the full analysis pipeline used to place the first simultaneous Bayesian constraints on the molecular geometry, rotational constants, and excitation temperature of the carriers of the two strongest near-infrared DIBs in the APOGEE spectral window—the 15,273 Å and 15,672 Å bands.

The central methodological advance exploited here is the simultaneous use of two observables derived from the APOGEE/SDSS-V archive by Saydjari & Green (2025):

1. **The mean DIB profile** ∂f(λ)/∂E — the absorption profile per unit dust column.
2. **Its derivative with respect to the dust extinction parameter R(V)** — ∂f(λ)/∂R(V), which encodes how the DIB profile shape changes as a function of interstellar environment.

The R(V)-response function is physically interpreted as a linear combination of the profile itself (reflecting environmentally driven changes in carrier column density) and the *temperature derivative* of the profile ∂p(λ)/∂T (reflecting changes in rotational level populations as the carrier's excitation temperature varies with dust environment). This second observable breaks the classical degeneracy between rotational constants and excitation temperature that has limited all previous profile-fitting analyses of DIBs.

---

## Scientific Motivation

The rotational envelope of a DIB encodes the moments of inertia—and thereby the size, geometry, and symmetry—of the carrier molecule, but this envelope is controlled by *both* the rotational constants (A, B, C) and the excitation temperature T. A colder, lighter molecule can produce a profile nearly indistinguishable from a hotter, heavier one at any achievable spectral resolution. Previous studies were forced to either fix T from independent tracers (C₂, H₂ rotational temperatures) or fix the rotational constants to those of a candidate molecule.

The physical link between R(V) and dust grain temperature provides a natural lever arm: sightlines with higher R(V) probe larger, colder grains in denser environments with weaker interstellar radiation fields. The corresponding change in the DIB carrier's rotational temperature produces a *shape* change in the absorption profile that is orthogonal in parameter space to the dependence of the profile shape on the rotational constants. Fitting both ∂f/∂E and ∂f/∂R(V) simultaneously therefore yields posteriors that are, for the first time, simultaneously informative about all three principal rotational constants, the excitation temperature, and the rate of change of DIB strength and temperature with environment.

---

## Methods

Rotational profiles are synthesized using [PGOPHER](https://pgopher.chm.bris.ac.uk/) (Western 2017), a comprehensive spectral simulation package. We consider molecular symmetry groups of increasing complexity:

- **C∞v (linear rigid rotor)** — single rotational constant B; useful as a baseline but insufficient for the broad, asymmetric NIR profiles.
- **C₂v (asymmetric top, one transition dipole axis)** — three rotational constants (A, B, C) plus ro-vibrational coupling terms (αA, αB, αC) and a Lorentzian lifetime/velocity-broadening width.
- **Cs (asymmetric top, two transition dipole axes)** — as above, plus the ratio r of the squared transition dipole moments along the two allowed axes.

Each synthesized profile is convolved with the APOGEE instrumental line-spread function (R ≈ 22,500, averaged across the APO and LCO spectrographs). The temperature derivative ∂p(λ)/∂T is computed by finite difference. Given the nonlinear molecular parameters (T, A, B, C, αA, αB, αC, τ, and optionally r), the remaining linear scaling coefficients (γ, α, β, d, e) are solved analytically by weighted least squares at each likelihood evaluation, making the MCMC sampling tractable.

Posterior distributions are sampled with the affine-invariant ensemble sampler `emcee` (Foreman-Mackey et al. 2013), distributing walkers across many CPU cores. Each likelihood call invokes PGOPHER (≈1–2 s per evaluation); convergence requires ≳10,000 steps, for a total cost of approximately **1,500 CPU-hours per fit**.

### Prior distributions

| Parameter | Prior | Motivation |
|-----------|-------|------------|
| T | Log-normal, μ = 25 K, σ = 0.4 dex | Typical diffuse cloud excitation temperatures |
| αi = Ae/Ag | Half-Gaussian, σ = 0.1 | Small geometry changes between ground and excited states |
| τ (Lorentzian width) | Exponential, slope τ₀ = 0.05 cm⁻¹ | Small intrinsic linewidths |
| A, B, C | Hard bound A ≥ B ≥ C | Eliminates computational degeneracies |

---

## Key Results (Jespersen & Saydjari, in prep.)

- **First simultaneous Bayesian constraints on all three principal rotational constants** (A, B, C) and excitation temperature T for two NIR DIB carriers, enabled by the joint fit to the profile and its R(V)-response function.
- The 15,273 Å carrier is consistent with a **planar polycyclic aromatic hydrocarbon (PAH)** geometry; the moment-of-inertia ratios place it among fused-ring structures containing tens of carbon atoms.
- The 15,672 Å carrier occupies a **distinct region of molecular geometry space**, demonstrating that the two strongest NIR DIBs arise from chemically different species.
- Rotational excitation temperatures of order **T ≈ 22 K**, consistent with diffuse ISM thermal conditions and independent estimates from C₂ rotational populations.
- The implied **dT/dR(V) ≈ 27.8 K per unit R(V)**, quantifying for the first time the environmental temperature sensitivity of a DIB carrier.
- The inferred grain sizes and rotation rates are **compatible with the spinning-dust mechanism** responsible for Anomalous Microwave Emission (AME), connecting the DIB carriers to one of the other major open problems in interstellar medium physics.

---

## Repository Structure

| File / Directory | Description |
|-----------------|-------------|
| `run_emcee_asym_more_options.py` | Main MCMC driver for asymmetric-top fits (C₂v and Cs) with configurable priors, symmetry, and data |
| `run_emcee_asym_more_options_fast*.py` | Optimized variants for the 15,672 Å band |
| `ingest_pgo_asymtop_emcee*.jl` | Julia interface to PGOPHER; constructs forward model spectra and temperature derivatives |
| `MCMC_convergence_tests.py` | Convergence diagnostics (autocorrelation times, Gelman–Rubin statistics) |
| `pca_version.txt` | Principal-component decomposition of the R(V)-response function |
| `pgo_files/` | PGOPHER input templates (.pgo) for each symmetry group and DIB |
| `LSFs/` | APOGEE instrumental line-spread functions (APO + LCO) |
| `pgo_outputs_*.h5` | Cached PGOPHER output spectra |
| `jackknife_dib.h5` | Jackknife covariance matrices for the observed profiles |
| `emcee_analysis*.ipynb` | Posterior analysis and figure generation notebooks |
| `slurm*/` | HPC job submission scripts (Princeton Della cluster) |

---

## Dependencies

**Python**: `emcee`, `numpy`, `scipy`, `pandas`, `h5py`, `matplotlib`

**Julia**: `FITSIO.jl`, `HDF5.jl`

**External**: [PGOPHER](https://pgopher.chm.bris.ac.uk/) (Western 2017) — must be installed and accessible as `./pgopher` or via the path set in the run scripts.

---

## Reproducing the Fits

```bash
# C2v fit for the 15272 DIB (default settings)
python run_emcee_asym_more_options.py

# Cs fit with flat priors for the 15672 DIB
python run_emcee_asym_more_options_fast_15672.py --flat_prior True

# Fit profile only (no temperature derivative)
python run_emcee_asym_more_options.py --fit_dT False

# Fit temperature derivative only
python run_emcee_asym_more_options.py --fit_spec False
```

For HPC submission on a SLURM system, see the scripts in `slurm_della/`.

---

## Data

The observed DIB profiles ∂f(λ)/∂E and R(V)-response functions ∂f(λ)/∂R(V) are taken from **Saydjari & Green (2025)**, who derived them from the full APOGEE/SDSS-V spectroscopic archive using a hierarchical profile-fitting framework. Please cite that work when using those data products. See also their Figure 10 and Figure 15 for the specific profiles fitted here.

---

## Citation

If you use this code, please cite the companion paper:

> Jespersen, C. K. & Saydjari, A. K. (in prep.). *Constraints on the Geometric Structure and Physical Parameters of two DIB Carriers in APOGEE.*

and the underlying observational dataset:

> Saydjari, A. K. & Green, G. M. (2025). *[DIB profiles and R(V)-response functions from APOGEE/SDSS-V.]*

---

## Contact

Questions, bug reports, and requests for collaboration are welcome. Please open a GitHub issue or contact the authors directly:

- Christian Kragh Jespersen — ckragh@princeton.edu
- Andrew K. Saydjari — aksaydjari@gmail.com
