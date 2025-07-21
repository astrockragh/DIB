from multiprocessing import get_context
import subprocess, os, emcee, time, shutil, h5py, argparse
from scipy.ndimage import gaussian_filter
import numpy as np
import pandas as pd
import os.path as osp
from pathlib import Path
from scipy.signal import fftconvolve
from scipy.signal import convolve
from scipy.interpolate import interp1d

# Constants
PGO_TEMPLATE = "asym_top_15272_Cs.pgo"

def parse_args():
    parser = argparse.ArgumentParser(description="Run emcee for asymmetric top spectra fitting")

    parser.add_argument("-B_not_equal_C", "--B_not_equal_C",
                        type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=False,
                        help="Allow B and C to be different (default: False)")

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
                        help="Fit with full covariance (default: False)")
    
    parser.add_argument("-nsteps", "--nsteps",
                        type=int,
                        default=5000,
                        help="How many steps to take (default: 5000)")
    
    parser.add_argument("-title", "--title",
                        type=str,
                        default='',
                        help="Anything to add to the title? (default: Nothing)")

    return parser.parse_args()

args = parse_args()

# Compose TEMP_SUFFIX and TEMP_DIR depending on all args
def val_to_str(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    elif isinstance(v, float):
        return str(v).replace('.', 'p')
    return str(v)

TEMP_SUFFIX = f"Cs_BC{val_to_str(args.B_not_equal_C)}_F{val_to_str(args.fudge)}_D{val_to_str(args.use_direct)}_" + \
              f"Flat{val_to_str(args.flat_prior)}_Spec{val_to_str(args.fit_spec)}_dT{val_to_str(args.fit_dT)}_cov{val_to_str(args.cov)}"+f'_{args.title}'

print(TEMP_SUFFIX)

TEMP_DIR = osp.expanduser(f"~/../../scratch/gpfs/cj1223/DIB/pgo_temppy_{TEMP_SUFFIX}")
os.makedirs(TEMP_DIR, exist_ok=True)
shutil.rmtree(TEMP_DIR, ignore_errors=False, onerror=None)
os.makedirs(TEMP_DIR, exist_ok=True)

def filename_base(T, A_base, B_base, C_base, frac_A, frac_B, frac_C, axis = 'b'):
    return f"T{T:.3f}_A{A_base:.7f}_B{B_base:.7f}_C{C_base:.7f}_FA{frac_A:.5f}_FB{frac_B:.5f}_FC{frac_C:.5f}_ax{axis}"

def generate_pgopher_input(T, A_base, B_base, C_base, frac_A, frac_B, frac_C, axis="b"):
    A_g, B_g, C_g = A_base, B_base, C_base
    A_e, B_e, C_e = A_base * frac_A, B_base * frac_B, C_base * frac_C

    base = filename_base(T, A_base, B_base, C_base, frac_A, frac_B, frac_C, axis = axis)
    pgo_file = os.path.join(TEMP_DIR, f"temp_{base}.pgo")
    spec_txt = os.path.join(TEMP_DIR, f"spec_{base}.txt")

    awk_script = f'''
    awk -v temp="{T}" \\
        -v A_ground="{A_g}" -v B_ground="{B_g}" -v C_ground="{C_g}" \\
        -v A_excited="{A_e}" -v B_excited="{B_e}" -v C_excited="{C_e}" \\
        -v axis="{axis}" '
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
    /<CartesianTransitionMoment Axis="/ {{
        sub(/Axis="[^"]+"/, "Axis=\\"" axis "\\"")
    }}
    {{ print }}
    ' {PGO_TEMPLATE} > {pgo_file}
    '''

    subprocess.run(awk_script, shell=True, check=True, executable="/bin/bash")
    subprocess.run(["./pgo", "--plot", pgo_file, spec_txt], check=True, stdout=subprocess.DEVNULL)
    return spec_txt, base

def convolve_pgopher_spectrum(spectrum_file, center_wav, lsf_key='LCO+APO', dlam=0.01, window=8):
    """
    Convolve PGOPHER output using LSF evaluated on a regular wavelength grid.

    Parameters:
        spectrum_file (str): PGOPHER output file with [wavenumber (1/cm), flux].
        center_wav (float): Central wavelength for the LSF (Å).
        lsf_key (str): Instrument profile selector (placeholder).
        dlam (float): Spacing (Å) for the regular wavelength grid.
        window (float): Half-width (Å) of the wavelength region to define the regular grid.

    Returns:
        wav_lsf (np.ndarray): Wavelength grid used for final output.
        flux_on_lsf_grid (np.ndarray): Flux convolved and resampled to wav_lsf.
        lsf_kernel (np.ndarray): LSF evaluated on regular grid.
        wav_pgo (np.ndarray): Original PGOPHER wavelength grid.
        convolved_flux (np.ndarray): Full-resolution convolved flux (on regular grid).
    """
    # === Load PGOPHER spectrum ===
    inv_cm, flux = np.loadtxt(spectrum_file).T
    wav_pgo = 1e8 / inv_cm

    wavc = 15272.27178113337
    # === Define regular wavelength grid around center ===
    wav_reg = np.arange(wavc - window, wavc + window, dlam)

    # === Interpolate PGOPHER flux onto this regular grid ===
    flux_interp = interp1d(wav_pgo, flux, bounds_error=False, fill_value=0.0)
    flux_reg = flux_interp(wav_reg)

    # === Construct LSF kernel on regular grid ===
    sig1 = 0.3
    sig2 = 1.85 * sig1
    sig3 = 9.5 * sig1
    f1 = 0.895
    f2 = 0.1
    c0 = 1.3e-3

    rel_grid = wav_reg - wavc  # Center the kernel
    p1 = 1/np.sqrt(2*np.pi*sig1**2)*np.exp( - rel_grid**2 / (2 * sig1**2) )
    p2 = 1/np.sqrt(2*np.pi*sig2**2)*np.exp( - rel_grid**2 / (2 * sig2**2) )
    p3 = 1/np.sqrt(2*np.pi*sig3**2)*np.exp( - rel_grid**2 / (2 * sig3**2) )

    lsf_kernel = f1 * p1 + f2 * p2 + (1 - f1 - f2) * p3 + c0
    lsf_kernel /= np.sum(lsf_kernel)

    # === Convolve on regular grid ===
    convolved_flux = convolve(flux_reg, lsf_kernel, mode='same')

    # === Interpolate back to original PGOPHER (or LSF) grid ===
    out_interp = interp1d(wav_reg, convolved_flux, bounds_error=False, fill_value=0.0)

    lsf_file = f'LSFs/lsf_15272.h5'
    # Load LSF and its wavelength grid
    with h5py.File(lsf_file, 'r') as f:
        wav_load = f['wav'][:]
      # Or replace with another grid if desired
    flux_on_lsf_grid = out_interp(wav_load)

    return wav_load, flux_on_lsf_grid

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
        if not (1 <= T <= 100): return -np.inf
        if not (0.0001 <= C <= 0.04): return -np.inf
        if not (0.0001 <= B <= 0.04): return -np.inf
        if not (0.0001 <= A <= 0.3): return -np.inf
        if not (0.9 <= frac_A <= 1.0): return -np.inf
        if not (0.9 <= frac_B <= 1.0): return -np.inf
        if not (0.9 <= frac_C <= 1.0): return -np.inf
        return 0.0
    else:
        if T <= 1 or T > 100: return -np.inf
        ## params for log-normal Temp prior
        mu = np.log(30)
        sigma = 1
        temp_logprior = -np.log(T * sigma * np.sqrt(2 * np.pi)) - ((np.log(T) - mu) ** 2) / (2 * sigma ** 2)

        if C < 0.0005 or C > 0.04: return -np.inf
        if B < 0.0005 or B > 0.04: return -np.inf
        if C>=B: return np.inf
        else:
            if args.B_not_equal_C:
                C0 = (1/A+1/B)**(-1)
                CB_logprior = - ( (C-C0)/( np.sqrt(2) * 0.3 * C0 ) )**2
            else:
                CB_logprior = 0.0

        if A < 0.001 or A > 0.3: return -np.inf
        A_logprior = 0.0

        if frac_A > 1: return -np.inf
        frac_a_logprior = -10 * (frac_A - 1) ** 2

        if frac_B > 1: return -np.inf
        frac_b_logprior = -10 * (frac_B - 1) ** 2

        if frac_C > 1: return -np.inf
        frac_c_logprior = -10 * (frac_C - 1) ** 2

        return temp_logprior + CB_logprior + A_logprior + frac_a_logprior + frac_b_logprior + frac_c_logprior

def compute_loglikelihood(
    model_flux_b, model_flux_c,
    model_flux_dT_b, model_flux_dT_c,
    data_flux, data_flux_dT,
    noise_std, noise_std_dT
):
    chi2 = 0.0
    c = 10  # edge crop
    gf = 0.01  # Gaussian filter width
    b_frac = c_frac = offset = 0
    b_frac_dT = c_frac_dT = offset_dT = 0
    base_frac_dT = 0

    if args.fit_spec:
        # Apply Gaussian filter and crop edges
        spec_b = gaussian_filter(model_flux_b[c:-c], gf)
        spec_c = gaussian_filter(model_flux_c[c:-c], gf)
        measurement = data_flux[c:-c]
        noise = noise_std[c:-c]

        # Fit linear combination: b_frac * spec_b + c_frac * spec_c + offset
        M = np.vstack([spec_b, spec_c, np.ones_like(spec_b)]).T
        coeffs, _, _, _ = np.linalg.lstsq(M, measurement, rcond=None)
        b_frac, c_frac, offset = coeffs

        # Evaluate fit
        fit = b_frac * spec_b + c_frac * spec_c + offset
        chi = (measurement - fit) / noise
        chi2 += np.sum(chi ** 2)

    if args.fit_dT:
        # Apply Gaussian filter and crop edges
        spec_b = gaussian_filter(model_flux_b[c:-c], gf)
        spec_c = gaussian_filter(model_flux_c[c:-c], gf)
        spec_dT_b = gaussian_filter(model_flux_dT_b[c:-c], gf) - spec_b
        spec_dT_c = gaussian_filter(model_flux_dT_c[c:-c], gf) - spec_c
        measurement_dT = data_flux_dT[c:-c]
        noise_dT = noise_std_dT[c:-c]

        # Estimate direct ratio to construct the base spectrum
        if args.fit_spec:
            ratio_direct = b_frac / (c_frac + 1e-10)
        else:
            ratio_direct = 1.0  # fallback

        # Form matrix: linear combo of original + delta spectra
        base_spec = spec_b + ratio_direct * spec_c
        M_dT = np.vstack([base_spec, spec_dT_b, spec_dT_c, np.ones_like(base_spec)]).T
        coeffs_dT, _, _, _ = np.linalg.lstsq(M_dT, measurement_dT, rcond=None)
        base_frac_dT, b_frac_dT, c_frac_dT, offset_dT = coeffs_dT

        # Evaluate fit
        fit_dT = (
            base_frac_dT * base_spec +
            b_frac_dT * spec_dT_b +
            c_frac_dT * spec_dT_c +
            offset_dT
        )
        chi_dT = (measurement_dT - fit_dT) / noise_dT
        chi2 += np.sum(chi_dT ** 2)

        # Optional: ratio constraint on the shape of the delta contributions
        ratio_dT = b_frac_dT / (c_frac_dT + 1e-10)
        ratio_tol = 0.5 * ratio_direct
        ratio_deviation = ((ratio_dT - ratio_direct) / ratio_tol) ** 2
        chi2 += ratio_deviation
    
    # scalars = np.array([
    #     b_frac, c_frac, offset,
    #     base_frac_dT, b_frac_dT, c_frac_dT, offset_dT
    # ])

    scalars = np.array([
    float(b_frac), float(c_frac), float(offset),
    float(base_frac_dT), float(b_frac_dT), float(c_frac_dT), float(offset_dT)])

    return -0.5 * chi2, scalars

def model_log_likelihood(params, data_wavelength, data_flux, data_flux_dT, noise_std, noise_std_dT, central_wav = 15272):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf, np.zeros(7)

    try:
        if args.B_not_equal_C:
            T, A, B, C, frac_A, frac_B, frac_C = params
        else:
            T, A, C, frac_A, frac_C = params
            B = A
            frac_B = frac_A
        
        
        spec_txt_b, base_b = generate_pgopher_input(T, A, B, C, frac_A, frac_B, frac_C, axis='b')
        _, model_flux_b = convolve_pgopher_spectrum(spec_txt_b, central_wav)

        spec_txt_c, base_c = generate_pgopher_input(T, A, B, C, frac_A, frac_B, frac_C, axis='c')
        _, model_flux_c = convolve_pgopher_spectrum(spec_txt_c, central_wav)

        if args.fit_dT:
            spec_txt_dT_b, base_dT_b = generate_pgopher_input(T + 0.05, A, B, C, frac_A, frac_B, frac_C, axis='b')
            _, model_flux_dT_b = convolve_pgopher_spectrum(spec_txt_dT_b, central_wav)

            spec_txt_dT_c, base_dT_c = generate_pgopher_input(T + 0.05, A, B, C, frac_A, frac_B, frac_C, axis='c')
            _, model_flux_dT_c = convolve_pgopher_spectrum(spec_txt_dT_c, central_wav)
        else:
            model_flux_dT_b = np.zeros_like(data_flux)
            model_flux_dT_c = np.zeros_like(data_flux)

        # if args.cov:
        #     lnlike, scalars = compute_loglikelihood_cov(
        #     model_flux, model_flux_dT,
        #     data_flux, data_flux_dT,
        #     noise_std, noise_std_dT,
        #     args.fit_spec, args.fit_dT) 
        # else:
            # lnlike, scalars = compute_loglikelihood(
            #     model_flux, model_flux_dT,
            #     data_flux, data_flux_dT,
            #     noise_std, noise_std_dT,
            #     args.fit_spec, args.fit_dT
            # )

        lnlike, scalars = compute_loglikelihood(
                model_flux_b, model_flux_c,
                model_flux_dT_b, model_flux_dT_c,
                data_flux, data_flux_dT,
                noise_std, noise_std_dT)

        if args.fit_spec:
            for spec_txt in [spec_txt_b, spec_txt_c]:
                base = osp.basename(spec_txt)  # e.g. spec_T20.005_A0.0026984_...txt
                if base.startswith("spec_") and base.endswith(".txt"):
                    param_str = base[len("spec_"):-len(".txt")]  # strip prefix/suffix
                    temp_pgo_file = os.path.join(TEMP_DIR, f'temp_{param_str}.pgo')

                    # Clean up
                    os.remove(spec_txt)
                    os.remove(temp_pgo_file)

        if args.fit_dT:
            for spec_txt_dT in [spec_txt_dT_b, spec_txt_dT_c]:
                base_dT = osp.basename(spec_txt_dT)
                if base_dT.startswith("spec_") and base_dT.endswith(".txt"):
                    param_str = base_dT[len("spec_"):-len(".txt")]
                    temp_pgo_file_dT = os.path.join(TEMP_DIR, f'temp_{param_str}.pgo')

                    # Clean up
                    os.remove(spec_txt_dT)
                    os.remove(temp_pgo_file_dT)

        return lnlike + lp, scalars

    except Exception as e:
        print(f"Error for params {params}: {e}")
        return -np.inf, np.zeros(7)

# Clear TEMP_DIR on start
for file in Path(TEMP_DIR).iterdir():
    if file.is_file():
        file.unlink()

if args.B_not_equal_C:
    ndim = 7
    p0_center = [20, 0.02, 0.003, 0.003, 0.99, 0.99, 0.99] # T, A, B, C, frac_A, frac_B, frac_C
    step_scales = [15, 0.005, 0.0015, 0.0015, 0.001, 0.001, 0.001]
else:
    ndim = 5
    p0_center = [20, 0.02, 0.003, 0.99, 0.99]  # T, AB, C, frac_AB, frac_C
    step_scales = [15, 0.005, 0.0015, 0.001, 0.001]

nsteps = args.nsteps
ncpu_to_use = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else max(1, os.cpu_count())
nwalkers = ncpu_to_use
print(f"Using {ncpu_to_use} CPUs")

fudge = float(args.fudge) #how much to inflate errors that we may not believe in

DIB_15272 = h5py.File('new_errs/res_dib_15272.h5', "r")
data_wavelength = DIB_15272['wav'][:]
data_flux = DIB_15272['mean'][:][:,0]
data_flux_dT = DIB_15272['mean'][:][:,1]

noise_std = fudge*np.sqrt(DIB_15272['var'][:][:,0])
noise_std_dT = fudge*np.sqrt(DIB_15272['var'][:][:,1])

if args.use_direct:
    data_flux = DIB_15272['mean'][:][:,0]
    data_flux_dT = DIB_15272['mean'][:][:,1]
else:
    errs0 = h5py.File('jackknife_dib.h5', "r")
    measurements = pd.read_csv('pca_version.txt', sep='\s+', names=['wavelength', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'])
    data_flux = measurements['PC1_1'].values
    data_flux_dT = measurements['PC2_2'].values
    if args.cov:
        noise_std1 = errs0['cov'][:, :, 0]
        noise_std2 = errs0['cov'][:, :, 1]
    else:
        noise_std1 = fudge * np.sqrt(errs0['var'][:, 0])
        noise_std2 = np.sqrt(errs0['var'][:, 1])



backend_file = osp.expanduser(f"~/../../scratch/gpfs/cj1223/DIB/bc_run_{TEMP_SUFFIX}.h5")
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
        args=(data_wavelength, data_flux, data_flux_dT, noise_std, noise_std_dT),
        pool=pool,
        backend=backend
    )

    p0 = np.array([
        p0_center + np.array(step_scales) / np.sqrt(nwalkers) * np.random.normal(size=ndim)
        for _ in range(nwalkers)
    ])

    p0[:,3] = (1/p0[:,2]+1/p0[:,1])**(-1)

    startm = time.time()
    sampler.run_mcmc(p0, nsteps, progress=True)
    end = time.time()
    print(f"Multiprocessing took {end - startm:.1f} seconds")


##### Former

# def convolve_pgopher_spectrum(
#     spectrum_file,
#     center_wav,
#     lsf_key='LCO+APO'
# ):
#     """
#     Convolve a PGOPHER output spectrum directly on its own wavelength grid, then
#     interpolate the result onto the LSF wavelength grid.

#     Parameters:
#         spectrum_file (str): Path to PGOPHER output file.
#         lsf_file (str): Path to LSF .h5 file.
#         lsf_key (str): 'LCO', 'APO', or 'LCO+APO'.
#         center_on (str or float): Center of the LSF in Å. Use 'auto' for mean of PGOPHER grid.
#         normalize_lsf (bool): Normalize LSF before convolution.

#     Returns:
#         (wav_lsf, y_interp): Tuple of LSF wavelength grid and convolved+interpolated spectrum.
#     """
#     # Load PGOPHER spectrum
#     inv_cm, flux = np.loadtxt(spectrum_file).T
#     wav_pgo = 1e8 / inv_cm  # cm⁻¹ → Å
    
#     lsf_file=f'LSFs/lsf_{center_wav}.h5'
    
#     # Load LSF and its wavelength grid
#     with h5py.File(lsf_file, 'r') as f:
#         wav_lsf = f['wav'][:]
#         lsf = f['LCO'][:] if lsf_key == 'LCO' else f['APO'][:]
#         if lsf_key == 'LCO+APO':
#             lsf = f['LCO'][:] + f['APO'][:]
#     # Resample LSF onto PGOPHER wavelength grid
#     lsf_interp = interp1d(wav_lsf, lsf, bounds_error=False, fill_value=0.0)
#     lsf_on_pgo_grid = lsf_interp(wav_pgo)
    
#     # print(np.max(lsf))
#     # print(np.max(wav_pgo), np.max(wav_lsf))
#     # print(np.max(lsf_on_pgo_grid))
    
#     lsf_on_pgo_grid /= np.sum(lsf_on_pgo_grid)

#     # Convolve on native grid
#     convolved_flux = fftconvolve(flux, lsf_on_pgo_grid, mode='same')

#     # Interpolate final result onto wav_lsf grid
#     final_interp = interp1d(wav_pgo, convolved_flux, bounds_error=False, fill_value=0.0)
#     flux_on_lsf_grid = final_interp(wav_lsf)

#     return wav_lsf, flux_on_lsf_grid

# from scipy.interpolate import interp1d
# from scipy.signal import convolve

# def generate_pgopher_input(T, A_base, B_base, C_base, frac_A, frac_B, frac_C):
#     A_g, B_g, C_g = A_base, B_base, C_base
#     A_e, B_e, C_e = A_base * frac_A, B_base * frac_B, C_base * frac_C

#     base = filename_base(T, A_base, B_base, C_base, frac_A, frac_B, frac_C)
#     pgo_file = os.path.join(TEMP_DIR, f"temp_{base}.pgo")
#     spec_txt = os.path.join(TEMP_DIR, f"spec_{base}.txt")

#     awk_script = f'''
#     awk -v temp="{T}" \\
#         -v A_ground="{A_g}" -v B_ground="{B_g}" -v C_ground="{C_g}" \\
#         -v A_excited="{A_e}" -v B_excited="{B_e}" -v C_excited="{C_e}" '
#     BEGIN {{ inside_ground = 0; inside_excited = 0; }}
#     /<Parameter Name="Temperature" Value="/ {{
#         sub(/Value="[0-9.eE+-]+"/, "Value=\\"" temp "\\"")
#     }}
#     /<AsymmetricManifold Name="Ground"/ {{ inside_ground = 1 }}
#     /<AsymmetricManifold Name="Excited"/ {{ inside_excited = 1 }}
#     /<\/AsymmetricManifold>/ {{ inside_ground = 0; inside_excited = 0 }}
#     inside_ground && /<Parameter Name="A" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" A_ground "\\"") }}
#     inside_ground && /<Parameter Name="B" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" B_ground "\\"") }}
#     inside_ground && /<Parameter Name="C" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" C_ground "\\"") }}
#     inside_excited && /<Parameter Name="A" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" A_excited "\\"") }}
#     inside_excited && /<Parameter Name="B" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" B_excited "\\"") }}
#     inside_excited && /<Parameter Name="C" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" C_excited "\\"") }}
#     {{ print }}
#     ' {PGO_TEMPLATE} > {pgo_file}
#     '''

#     subprocess.run(awk_script, shell=True, check=True, executable="/bin/bash")
#     subprocess.run(["./pgo", "--plot", pgo_file, spec_txt], check=True, stdout=subprocess.DEVNULL)
#     return spec_txt, base

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