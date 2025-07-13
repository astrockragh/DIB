from multiprocessing import get_context
import subprocess
import os
import emcee
import time
from scipy.ndimage import gaussian_filter
import numpy as np
import pandas as pd
import h5py
import os.path as osp
from pathlib import Path
import argparse

# Read measurements
measurements = pd.read_csv('pca_version.txt', sep='\s+', names=['wavelength', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'])

# Constants
PGO_TEMPLATE = "asym_top_0.pgo"
JULIA_SCRIPT = "ingest_pgo_asymtop_emcee_more_options.jl"

def parse_args():
    parser = argparse.ArgumentParser(description="Run emcee for asymmetric top spectra fitting")

    parser.add_argument("-B_not_equal_C", "--B_not_equal_C",
                        type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=False,
                        help="Allow A and B to be different (default: False)")

    parser.add_argument("-fudge", "--fudge",
                        type=float,
                        default=5.0,
                        help="Fudge factor for noise errors (default: 5.0)")

    parser.add_argument("-use_direct", "--use_direct",
                        type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=True,
                        help="Use direct measured data (True) or PCA data (False) (default: True)")

    parser.add_argument("-flat_prior", "--flat_prior",
                        type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=False,
                        help="Use flat priors (default: False)")

    parser.add_argument("-fit_spec", "--fit_spec",
                        type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=True,
                        help="Fit main spectrum (default: True)")

    parser.add_argument("-fit_dT", "--fit_dT",
                        type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=True,
                        help="Fit temperature derivative spectrum (default: True)")
    
    parser.add_argument("-cov", "--cov",
                        type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=False,
                        help="Fit temperature derivative spectrum (default: True)")

    return parser.parse_args()

args = parse_args()

# Compose TEMP_SUFFIX and TEMP_DIR depending on all args
def val_to_str(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    elif isinstance(v, float):
        return str(v).replace('.', 'p')
    return str(v)

TEMP_SUFFIX = f"BC{val_to_str(args.B_not_equal_C)}_F{val_to_str(args.fudge)}_D{val_to_str(args.use_direct)}_" + \
              f"Flat{val_to_str(args.flat_prior)}_Spec{val_to_str(args.fit_spec)}_dT{val_to_str(args.fit_dT)}_cov{val_to_str(args.cov)}"

print(TEMP_SUFFIX)

TEMP_DIR = osp.expanduser(f"~/../../scratch/gpfs/cj1223/DIB/pgo_temppy_{TEMP_SUFFIX}")
os.makedirs(TEMP_DIR, exist_ok=True)

def filename_base(T, A_base, B_base, C_base, frac_A, frac_B, frac_C):
    return f"T{T:.3f}_A{A_base:.7f}_B{B_base:.7f}_C{C_base:.7f}_FA{frac_A:.5f}_FB{frac_B:.5f}_FC{frac_C:.5f}"

def generate_pgopher_input(T, A_base, B_base, C_base, frac_A, frac_B, frac_C):
    A_g, B_g, C_g = A_base, B_base, C_base
    A_e, B_e, C_e = A_base * frac_A, B_base * frac_B, C_base * frac_C

    base = filename_base(T, A_base, B_base, C_base, frac_A, frac_B, frac_C)
    pgo_file = os.path.join(TEMP_DIR, f"temp_{base}.pgo")
    spec_txt = os.path.join(TEMP_DIR, f"spec_{base}.txt")

    awk_script = f'''
    awk -v temp="{T}" \\
        -v A_ground="{A_g}" -v B_ground="{B_g}" -v C_ground="{C_g}" \\
        -v A_excited="{A_e}" -v B_excited="{B_e}" -v C_excited="{C_e}" '
    BEGIN {{ inside_ground = 0; inside_excited = 0; }}
    /<Parameter Name="Temperature" Value="/ {{
        sub(/Value="[0-9.eE+-]+"/, "Value=\\"" temp "\\"")
    }}
    /<AsymmetricManifold Name="Ground"/ {{ inside_ground = 1 }}
    /<AsymmetricManifold Name="Excited"/ {{ inside_excited = 1 }}
    /<\/AsymmetricManifold>/ {{ inside_ground = 0; inside_excited = 0 }}
    inside_ground && /<Parameter Name="A" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" A_ground "\\"") }}
    inside_ground && /<Parameter Name="B" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" B_ground "\\"") }}
    inside_ground && /<Parameter Name="C" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" C_ground "\\"") }}
    inside_excited && /<Parameter Name="A" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" A_excited "\\"") }}
    inside_excited && /<Parameter Name="B" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" B_excited "\\"") }}
    inside_excited && /<Parameter Name="C" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" C_excited "\\"") }}
    {{ print }}
    ' {PGO_TEMPLATE} > {pgo_file}
    '''

    subprocess.run(awk_script, shell=True, check=True, executable="/bin/bash")
    subprocess.run(["./pgo", pgo_file, "-o", spec_txt], check=True, stdout=subprocess.DEVNULL)
    return spec_txt, base

def run_julia_convolution(spec_txt, base):
    output_h5 = os.path.join(TEMP_DIR, f"convolved_{base}")
    result = subprocess.run([
        "julia", JULIA_SCRIPT, spec_txt, output_h5
    ], check=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Julia failed!")
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)
        raise RuntimeError("Julia call failed")

    return output_h5 + ".h5"

def read_h5_spectrum(h5_file):
    with h5py.File(h5_file, "r") as f:
        spectrum = f['spectra'][:]
        wavelength = f['wavelengths'][:]
    return wavelength, spectrum

def log_prior(params):
    if args.B_not_equal_C:
        if len(params) != 7:
            return -np.inf
        T, A, B, C, frac_A, frac_B, frac_C = params
    else:
        if len(params) != 5:
            return -np.inf
        T, A, C, frac_A, frac_C = params
        B = A
        frac_B = frac_A

    if args.flat_prior:
        if not (3 <= T <= 100): return -np.inf
        if not (0.0001 <= C <= 0.02): return -np.inf
        if not (0.0001 <= B <= 0.02): return -np.inf
        if not (0.0001 <= A <= 0.2): return -np.inf
        if not (0.9 <= frac_A <= 1.0): return -np.inf
        if not (0.9 <= frac_B <= 1.0): return -np.inf
        if not (0.9 <= frac_C <= 1.0): return -np.inf
        return 0.0
    else:
        if T <= 2 or T > 100: return -np.inf
        ## params for log-normal Temp prior
        mu = np.log(30)
        sigma = 1
        temp_logprior = -np.log(T * sigma * np.sqrt(2 * np.pi)) - ((np.log(T) - mu) ** 2) / (2 * sigma ** 2)

        if C < 0.0005 or C > 0.1: return -np.inf
        if B < 0.0005 or B > 0.1: return -np.inf
        ab_logprior = 0.0

        if A < 0.001 or A > 0.2: return -np.inf
        c_logprior = 0.0

        if frac_A > 1: return -np.inf
        frac_a_logprior = -10 * (frac_A - 1) ** 2

        if frac_B > 1: return -np.inf
        frac_b_logprior = -10 * (frac_B - 1) ** 2

        if frac_C > 1: return -np.inf
        frac_c_logprior = -10 * (frac_C - 1) ** 2

        return temp_logprior + ab_logprior + c_logprior + frac_a_logprior + frac_b_logprior + frac_c_logprior


def compute_loglikelihood(model_flux, model_flux_dT, data_flux, data_flux_dT, noise_std, noise_std_dT, fit_spec_flag, fit_dT_flag):
    chi2 = 0.0
    c = 10 #edge factor, don't fit to the nothing around the edges
    gf = 0.1

    if fit_spec_flag:
        model_spec = gaussian_filter(model_flux[c:-c], gf)
        measurement = data_flux[c:-c]
        noise = noise_std[c:-c]

        M = np.vstack([model_spec, np.ones_like(model_spec)]).T
        coeffs, _, _, _ = np.linalg.lstsq(M, measurement, rcond=None)
        scalar, offset = coeffs
        fit = scalar * model_spec + offset
        chi = (measurement - fit) / noise
        chi2 += np.sum(chi ** 2)
    else:
        scalar = offset = 0

    if fit_dT_flag:
        model_spec = gaussian_filter(model_flux[c:-c], gf)
        model_spec_dT = gaussian_filter(model_flux_dT[c:-c], gf) - model_spec
        measurement_dT = data_flux_dT[c:-c]
        noise_dT = noise_std_dT[c:-c]
        M = np.vstack([model_spec, model_spec_dT, np.ones_like(model_spec)]).T
        coeffs, _, _, _ = np.linalg.lstsq(M, measurement_dT, rcond=None)
        scalar1, scalar2, offset_dT = coeffs
        fit_dT = scalar1 * model_spec + scalar2 * model_spec_dT + offset_dT
        chi = (measurement_dT - fit_dT) / noise_dT
        chi2 += np.sum(chi ** 2)
    else:
        scalar1 = scalar2 = offset_dT = 0

    return -0.5 * chi2, np.array([scalar, scalar1, scalar2])

def compute_loglikelihood_cov(model_flux, model_flux_dT, data_flux, data_flux_dT,
                          cov_spec, cov_dT, fit_spec_flag, fit_dT_flag):
    chi2 = 0.0
    c = 35  # edge factor
    gf = 0.1

    if fit_spec_flag:
        model_spec = gaussian_filter(model_flux[c:-c], gf)
        measurement = data_flux[c:-c]
        cov = cov_spec[c:-c, c:-c]  # crop covariance matrix

        M = np.vstack([model_spec, np.ones_like(model_spec)]).T
        coeffs, _, _, _ = np.linalg.lstsq(M, measurement, rcond=None)
        scalar, offset = coeffs
        fit = scalar * model_spec + offset
        delta = measurement - fit

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return -np.inf, np.array([0, 0, 0])  # Bail out on singular matrix

        chi2 += (delta @ cov_inv @ delta)/args.fudge
    else:
        scalar = offset = 0

    if fit_dT_flag:
        model_spec = gaussian_filter(model_flux[c:-c], gf)
        model_spec_dT = gaussian_filter(model_flux_dT[c:-c], gf) - model_spec
        measurement_dT = data_flux_dT[c:-c]
        cov_dT_crop = cov_dT[c:-c, c:-c]

        M = np.vstack([model_spec, model_spec_dT, np.ones_like(model_spec)]).T
        coeffs, _, _, _ = np.linalg.lstsq(M, measurement_dT, rcond=None)
        scalar1, scalar2, offset_dT = coeffs
        fit_dT = scalar1 * model_spec + scalar2 * model_spec_dT + offset_dT
        delta_dT = measurement_dT - fit_dT

        try:
            cov_dT_inv = np.linalg.inv(cov_dT_crop)
        except np.linalg.LinAlgError:
            return -np.inf, np.array([0, 0, 0])  # Bail out on singular matrix

        chi2 += delta_dT @ cov_dT_inv @ delta_dT
    else:
        scalar1 = scalar2 = offset_dT = 0

    return -0.5 * chi2, np.array([scalar, scalar1, scalar2])

def model_log_likelihood(params, data_wavelength, data_flux, data_flux_dT, noise_std, noise_std_dT):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf, np.array([0.0, 0.0, 0.0])

    try:
        if args.B_not_equal_C:
            T, A, B, C, frac_A, frac_B, frac_C = params
        else:
            T, A, C, frac_A, frac_C = params
            B = A
            frac_B = frac_A
        
        if args.fit_spec:
            spec_txt, base = generate_pgopher_input(T, A, B, C, frac_A, frac_B, frac_C)
            h5_file = run_julia_convolution(spec_txt, base)
            _, model_flux = read_h5_spectrum(h5_file)
        else:
            model_flux = np.zeros_like(data_flux)

        if args.fit_dT:
            spec_txt_dT, base_dT = generate_pgopher_input(T + 0.05, A, B, C, frac_A, frac_B, frac_C)
            h5_file_dT = run_julia_convolution(spec_txt_dT, base_dT)
            _, model_flux_dT = read_h5_spectrum(h5_file_dT)
        else:
            model_flux_dT = np.zeros_like(data_flux)

        if args.cov:
            lnlike, scalars = compute_loglikelihood_cov(
            model_flux, model_flux_dT,
            data_flux, data_flux_dT,
            noise_std, noise_std_dT,
            args.fit_spec, args.fit_dT)

        else:
            lnlike, scalars = compute_loglikelihood(
                model_flux, model_flux_dT,
                data_flux, data_flux_dT,
                noise_std, noise_std_dT,
                args.fit_spec, args.fit_dT
            )

        if args.fit_spec:
            # Extract parameter string
            base = osp.basename(spec_txt)  # e.g. spec_T20.005_A0.0026984_...txt

            if base.startswith("spec_") and base.endswith(".txt"):
                param_str = base[len("spec_"):-len(".txt")]  # strip prefix/suffix
                temp_pgo_file = os.path.join(TEMP_DIR, f'temp_{param_str}.pgo')
            
            # Clean up
            os.remove(spec_txt)
            os.remove(temp_pgo_file)
            os.remove(h5_file)

        if args.fit_dT:
            ### Now the same for the dT files
            # Extract parameter string
            base_dT = osp.basename(spec_txt_dT)  # e.g. spec_T20.005_A0.0026984_...txt

            if base_dT.startswith("spec_") and base_dT.endswith(".txt"):
                param_str = base_dT[len("spec_"):-len(".txt")]  # strip prefix/suffix
                temp_pgo_file_dT = os.path.join(TEMP_DIR, f'temp_{param_str}.pgo')
            
            # Clean up
            os.remove(spec_txt_dT)
            os.remove(temp_pgo_file_dT)
            os.remove(h5_file_dT)

        return lnlike + lp, scalars

    except Exception as e:
        print(f"Error for params {params}: {e}")
        return -np.inf, np.array([0.0, 0.0, 0.0])

# Clear TEMP_DIR on start
for file in Path(TEMP_DIR).iterdir():
    if file.is_file():
        file.unlink()

if args.B_not_equal_C:
    ndim = 7
    p0_center = [20, 0.02, 0.003, 0.003, 0.99, 0.99, 0.99] # T, A, B, C, frac_A, frac_B, frac_C
    step_scales = [15, 0.005, 0.0015, 0.0015, 0.0005, 0.0005, 0.0005]
else:
    ndim = 5
    p0_center = [20, 0.02, 0.003, 0.99, 0.99]  # T, AB, C, frac_AB, frac_C
    step_scales = [15, 0.005, 0.0015, 0.001, 0.001]

nsteps = 5000
ncpu_to_use = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else max(1, os.cpu_count())
nwalkers = ncpu_to_use
print(f"Using {ncpu_to_use} CPUs")

errs = h5py.File('jackknife_dib.h5', "r")

fudge = float(args.fudge) #how much to inflate errors that we may not believe in

if args.cov:
    noise_std1 = errs['cov'][:, :, 0]
    noise_std2 = errs['cov'][:, :, 1]
else:
    noise_std1 = fudge * np.sqrt(errs['var'][:, 0])
    noise_std2 = fudge * np.sqrt(errs['var'][:, 1])

data_wavelength = measurements['wavelength'].values

if args.use_direct:
    data_flux = errs['mean'][0, :, 0]
    data_flux_dT = errs['mean'][0, :, 1]
else:
    data_flux = measurements['PC1_1'].values
    data_flux_dT = measurements['PC2_2'].values

backend_file = osp.expanduser(f"~/../../scratch/gpfs/cj1223/DIB/full_run_{TEMP_SUFFIX}.h5")
if osp.exists(backend_file):
    os.remove(backend_file)  # Ensure clean start

backend = emcee.backends.HDFBackend(backend_file)

with get_context("fork").Pool(processes=ncpu_to_use) as pool:
    
    print(
    f"Running MCMC with the following settings:\n"
    f"- B and C treated as {'different' if args.B_not_equal_C else 'equal'}\n"
    f"- Fudge factor on noise: {args.fudge}\n"
    f"- Using {'direct' if args.use_direct else 'PCA'} spectrum data\n"
    f"- {'Flat priors' if args.flat_prior else 'Priors chosen by Andrew and I'}\n"
    f"- {'Using diagonal of covariance only' if not args.cov else 'Using full covariance'}\n"
    f"- Fitting main spectrum: {args.fit_spec}\n"
    f"- Fitting temperature derivative: {args.fit_dT}") 

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        model_log_likelihood,
        args=(data_wavelength, data_flux, data_flux_dT, noise_std1, noise_std2),
        pool=pool,
        backend=backend
    )

    p0 = np.array([
        p0_center + np.array(step_scales) / np.sqrt(nwalkers) * np.random.normal(size=ndim)
        for _ in range(nwalkers)
    ])

    startm = time.time()
    sampler.run_mcmc(p0, nsteps, progress=True)
    end = time.time()
    print(f"Multiprocessing took {end - startm:.1f} seconds")
