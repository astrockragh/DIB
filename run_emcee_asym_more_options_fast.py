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

    parser.add_argument("-symmetry_group", "--symmetry_group",
                        type=str,
                        default='Cs',
                        help="Which symmetry group to use (default: Cs (only other option is currently C2v))")
    
    parser.add_argument("-nonlinear_fit", "--nonlinear_fit",
                        type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=False,
                        help="Fit the constants in a fully non-linear way")
    
    parser.add_argument("-use_scalar_prior", "--use_scalar_prior",
                        type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=False,
                        help="Use a prior on the b/c ratio instead of a direct ")
    
    parser.add_argument("-tau_prior", "--tau_prior",
                        type=float,
                        default=0.05,
                        help="Slope on the exponential prior for lifetime broadening [cm^{-1}]")
    
    parser.add_argument("-alpha_prior", "--alpha_prior",
                        type=float,
                        default=0.14,
                        help="Standard deviation on the gaussian prior for the alpha coefficients")
    
    parser.add_argument("-old_errs", "--old_errs",
                        type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=False,
                        help="If True, use the former errors that Andrew had computed which are very tight")
    
    parser.add_argument("-extra_truncation", "--extra_truncation",
                        type=int,
                        default=0,
                        help="How much further should the data be truncated? Measured in wavelength bins (default: 0)")
    
    parser.add_argument("-plate_errs", "--plate_errs",
                        type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=False,
                        help="If True, use the new errors which were jackknifed over plates, it ends up somewhere between old/new errs")
    
    return parser.parse_args()

args = parse_args()

assert not args.old_errs and args.plate_errs, "Cannot both want to use old errors and new plate jackknifed errors. Set either --old_errs OR --plate_errs to True."
# Compose TEMP_SUFFIX and TEMP_DIR depending on all args
def val_to_str(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    elif isinstance(v, float):
        return str(v).replace('.', 'p')
    return str(v)

errType = 'Default'
if args.old_errs:
    errType = 'Old'
if args.plate_errs:
    errType = 'Plate'

TEMP_SUFFIX = f"Symmetry{val_to_str(args.symmetry_group)}_BC{val_to_str(args.B_not_equal_C)}_F{val_to_str(args.fudge)}_D{val_to_str(args.use_direct)}_" + \
              f"Flat{val_to_str(args.flat_prior)}_Spec{val_to_str(args.fit_spec)}_dT{val_to_str(args.fit_dT)}_cov{val_to_str(args.cov)}_nonlin{val_to_str(args.nonlinear_fit)}"+ \
              f'_tauSlope{val_to_str(args.tau_prior)}_alphaSlope{val_to_str(args.alpha_prior)}_err{errType}_trunc{args.extra_truncation}_{args.title}'

# Constants
if args.symmetry_group == 'Cs':
    PGO_TEMPLATE = osp.expanduser("~/DIB/pgo_files/asym_top_15272_Cs.pgo")
if args.symmetry_group == 'C2v':
    PGO_TEMPLATE = osp.expanduser("~/DIB/pgo_files/asym_top_15272_C2v.pgo")

TEMP_DIR = osp.expanduser(f"~/../../scratch/gpfs/cj1223/DIB/pgo_temppy_{TEMP_SUFFIX}")

os.makedirs(TEMP_DIR, exist_ok=True)
shutil.rmtree(TEMP_DIR, ignore_errors=False, onerror=None)
os.makedirs(TEMP_DIR, exist_ok=True)

def filename_base(T, A_base, B_base, C_base, frac_A, frac_B, frac_C, lorentz_width=0.01, axis = 'b'):
    return f"T{T:.3f}_A{A_base:.7f}_B{B_base:.7f}_C{C_base:.7f}_FA{frac_A:.5f}_FB{frac_B:.5f}_FC{frac_C:.5f}_ax{axis}_lifetime{lorentz_width:.3f}"

def generate_pgopher_input_Cs(T, A_base, B_base, C_base, frac_A, frac_B, frac_C, lorentz_width=0.01, axis="b"):
    A_g, B_g, C_g = A_base, B_base, C_base
    A_e, B_e, C_e = A_base * frac_A, B_base * frac_B, C_base * frac_C

    base = filename_base(T, A_base, B_base, C_base, frac_A, frac_B, frac_C, axis = axis)
    pgo_file = os.path.join(TEMP_DIR, f"temp_{base}.pgo")
    spec_txt = os.path.join(TEMP_DIR, f"spec_{base}.txt")

    awk_script = f'''
    awk -v temp="{T}" \\
        -v A_ground="{A_g}" -v B_ground="{B_g}" -v C_ground="{C_g}" \\
        -v A_excited="{A_e}" -v B_excited="{B_e}" -v C_excited="{C_e}" \\
        -v axis="{axis}" -v lorentz_width="{lorentz_width}" '
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
    /<Parameter Name="Lorentzian" Value="/ {{
        sub(/Value="[0-9.eE+-]+"/, "Value=\\"" lorentz_width "\\"")
    }}
    {{ print }}
    ' {PGO_TEMPLATE} > {pgo_file}
    '''

    subprocess.run(awk_script, shell=True, check=True, executable="/bin/bash")
    subprocess.run([osp.expanduser("~/DIB/./pgo"), "--plot", pgo_file, spec_txt], check=True, stdout=subprocess.DEVNULL)
    return spec_txt, base

def generate_pgopher_input_C2v(T, A_base, B_base, C_base, frac_A, frac_B, frac_C,
                              lorentz_width=0.01, axis="a"):
    A_g, B_g, C_g = A_base, B_base, C_base
    A_e, B_e, C_e = A_base * frac_A, B_base * frac_B, C_base * frac_C

    base = filename_base(T, A_base, B_base, C_base, frac_A, frac_B, frac_C, axis=axis)
    pgo_file = os.path.join(TEMP_DIR, f"temp_{base}.pgo")
    spec_txt = os.path.join(TEMP_DIR, f"spec_{base}.txt")

    awk_script = f'''
    awk -v temp="{T}" \\
        -v A_ground="{A_g}" -v B_ground="{B_g}" -v C_ground="{C_g}" \\
        -v A_excited="{A_e}" -v B_excited="{B_e}" -v C_excited="{C_e}" \\
        -v axis="{axis}" -v lorentz_width="{lorentz_width}" '
    BEGIN {{
        in_ground = 0; in_excited = 0;
    }}
    /<AsymmetricTop Name="v=0"/ {{
        in_ground = 1;
    }}
    /<AsymmetricTop Name="v=1"/ {{
        in_excited = 1;
    }}
    /<\/AsymmetricTop>/ {{
        in_ground = 0;
        in_excited = 0;
    }}
    in_ground && /<Parameter Name="A" Value=/ {{
        sub(/Value="[0-9.eE+-]+"/, "Value=\\"" A_ground "\\"")
    }}
    in_ground && /<Parameter Name="B" Value=/ {{
        sub(/Value="[0-9.eE+-]+"/, "Value=\\"" B_ground "\\"")
    }}
    in_ground && /<Parameter Name="C" Value=/ {{
        sub(/Value="[0-9.eE+-]+"/, "Value=\\"" C_ground "\\"")
    }}
    in_excited && /<Parameter Name="A" Value=/ {{
        sub(/Value="[0-9.eE+-]+"/, "Value=\\"" A_excited "\\"")
    }}
    in_excited && /<Parameter Name="B" Value=/ {{
        sub(/Value="[0-9.eE+-]+"/, "Value=\\"" B_excited "\\"")
    }}
    in_excited && /<Parameter Name="C" Value=/ {{
        sub(/Value="[0-9.eE+-]+"/, "Value=\\"" C_excited "\\"")
    }}
    /<CartesianTransitionMoment Bra="v=1" Ket="v=0"/ {{
        sub(/Axis="[^"]+"/, "Axis=\\"" axis "\\"")
    }}
    /<Parameter Name="Temperature" Value=/ {{
        sub(/Value="[0-9.eE+-]+"/, "Value=\\"" temp "\\"")
    }}
    /<Parameter Name="Lorentzian" Value=/ {{
        sub(/Value="[0-9.eE+-]+"/, "Value=\\"" lorentz_width "\\"")
    }}
    {{ print }}
    ' {PGO_TEMPLATE} > {pgo_file}
    '''

    subprocess.run(awk_script, shell=True, check=True, executable="/bin/bash")
    subprocess.run([osp.expanduser("~/DIB/./pgo"), "--plot", pgo_file, spec_txt], check=True, stdout=subprocess.DEVNULL)
    return spec_txt, base

if args.symmetry_group == 'Cs':
    generate_pgopher_input = generate_pgopher_input_Cs
if args.symmetry_group == 'C2v':
    generate_pgopher_input = generate_pgopher_input_C2v

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

    if args.use_direct and not args.old_errs:
        lsf_file = osp.expanduser('~/DIB/LSFs/lsf_15272.h5')
        # Load LSF and its wavelength grid
        with h5py.File(lsf_file, 'r') as f:
            wav_load = f['wav'][:]
        
    else:
        measurements = pd.read_csv(osp.expanduser('~/DIB/pca_version.txt'), sep='\s+', names=['wavelength', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'])
        wav_load = measurements['wavelength']
    flux_on_lsf_grid = out_interp(wav_load)

    return wav_load, flux_on_lsf_grid

def log_prior_Cs(params):
    if args.B_not_equal_C:
        if len(params) != 8:
            return -np.inf
        T, A, B, C, frac_A, frac_B, frac_C, lorentz_width  = params
    else:
        if len(params) != 6:
            return -np.inf
        T, A, C, frac_A, frac_C, lorentz_width = params
        B = C
        frac_B = frac_C

    if args.flat_prior:
        if not (3 <= T <= 100): return -np.inf
        if not (0.0001 <= C <= 0.04): return -np.inf
        if not (0.0001 <= B <= 0.04): return -np.inf
        if not (0.0001 <= A <= 0.3): return -np.inf
        if not (0.9 <= frac_A <= 1.0): return -np.inf
        if not (0.9 <= frac_B <= 1.0): return -np.inf
        if not (0.9 <= frac_C <= 1.0): return -np.inf
        if not (0.0 <= lorentz_width <= 1.0): return -np.inf
        return 0.0
    else:
        if T <= 3 or T > 100: return -np.inf
        ## params for log-normal Temp prior
        mu = np.log(25)
        sigma = 0.4
        temp_logprior = -np.log(T * sigma * np.sqrt(2 * np.pi)) - ((np.log(T) - mu) ** 2) / (2 * sigma ** 2)

        if C < 0.0005 or C > 0.3: return -np.inf
        if B < 0.0005 or B > 0.3: return -np.inf
        if not (0.0 <= lorentz_width <= 1.0):
            return -np.inf
        else:
            lorentz_width_prior = - (lorentz_width/args.tau_prior) #exponential
        
        if B>A:
            return np.inf # if enforcing hierarchy
        if args.B_not_equal_C:
            if C>=B: return np.inf # if enforcing hierarchy
        
        if args.B_not_equal_C:
            C0 = (1/A+1/B)**(-1)
            CB_logprior = - ( (C-C0)/( np.sqrt(2) * 1 * C0 ) )**2
        else:
            CB_logprior = 0.0

        if A < 0.001 or A > 0.3: return -np.inf
        A_logprior = - ((0.013 - C)**2/(2*0.02**2))

        if frac_A > 1: return -np.inf
        alpha_sig = args.alpha_prior
        frac_a_logprior = - (frac_A - 1) ** 2 / (2*alpha_sig**2)

        if frac_B > 1: return -np.inf
        frac_b_logprior = - (frac_B - 1) ** 2 / (2*alpha_sig**2)

        if frac_C > 1: return -np.inf
        frac_c_logprior = - (frac_C - 1) ** 2 / (2*alpha_sig**2)
        frac_c_logprior = -100 * (frac_C - 1) ** 2

        return temp_logprior + CB_logprior + A_logprior + frac_a_logprior + frac_b_logprior + frac_c_logprior + lorentz_width_prior

def log_prior_C2v(params):
    if args.B_not_equal_C:
        if len(params) != 8:
            return -np.inf
        T, A, B, C, frac_A, frac_B, frac_C, lorentz_width  = params
    else:
        if len(params) != 6:
            return -np.inf
        T, A, C, frac_A, frac_C, lorentz_width = params
        B = A
        frac_B = frac_A

    if args.flat_prior:
        if not (3 <= T <= 100): return -np.inf
        if not (0.0001 <= C <= 0.3): return -np.inf
        if not (0.0001 <= B <= 0.3): return -np.inf
        if not (0.0001 <= A <= 0.3): return -np.inf
        if not (0.9 <= frac_A <= 1.0): return -np.inf
        if not (0.9 <= frac_B <= 1.0): return -np.inf
        if not (0.9 <= frac_C <= 1.0): return -np.inf
        if not (0.0 <= lorentz_width <= 1.0): return -np.inf
        return 0.0
    else:
        if T <= 3 or T > 100: return -np.inf
        ## params for log-normal Temp prior
        mu = np.log(25)
        sigma = 0.4
        temp_logprior = -np.log(T * sigma * np.sqrt(2 * np.pi)) - ((np.log(T) - mu) ** 2) / (2 * sigma ** 2)

        if C < 0.0005 or C > 0.3: return -np.inf
        C_logprior = - ((0.013 - C)**2/(2*0.02**2))
        if B < 0.0005 or B > 0.3: return -np.inf
        if not (0.0 <= lorentz_width <= 1.0):
            return -np.inf
        
        lorentz_width_prior = - (lorentz_width/args.tau_prior) #exponential
        
        if C<=B: return np.inf # if enforcing hierarchy
        if args.B_not_equal_C:
            if B<=A: return np.inf # if enforcing hierarchy

        if A < 0.0005 or A > 0.3: return -np.inf
        A_logprior = 0.0

        if frac_A > 1: return -np.inf

        alpha_sig = args.alpha_prior
        frac_a_logprior = - (frac_A - 1) ** 2 / (2*alpha_sig**2)

        if frac_B > 1: return -np.inf
        frac_b_logprior = - (frac_B - 1) ** 2 / (2*alpha_sig**2)

        if frac_C > 1: return -np.inf
        frac_c_logprior = - (frac_C - 1) ** 2 / (2*alpha_sig**2)

        return temp_logprior + C_logprior + A_logprior + frac_a_logprior + frac_b_logprior + frac_c_logprior + lorentz_width_prior

if args.symmetry_group == 'Cs':
    log_prior = log_prior_Cs
if args.symmetry_group == 'C2v':
    log_prior = log_prior_C2v

def compute_loglikelihood_Cs(
    model_flux_b, model_flux_c,
    model_flux_dT_b, model_flux_dT_c,
    data_flux, data_flux_dT,
    noise_std, noise_std_dT
):
    chi2 = 0.0

    if args.use_direct and not args.old_errs:
        c = 10 + args.extra_truncation # edge crop
    else:
        c = 30 + args.extra_truncation

    gf = 0.01  # Gaussian filter width
    b_frac = c_frac = offset = 0
    b_frac_dT = c_frac_dT = offset_dT = 0
    base_frac_dT = 0
    if args.use_scalar_prior:
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

            # Optional: ratio constraint on the shape of the dT contributions
            ratio_dT = b_frac_dT / (c_frac_dT + 1e-10)
            ratio_tol = 0.01 * ratio_direct
            ratio_deviation = ((ratio_dT - ratio_direct) / ratio_tol) ** 2
            chi2 += ratio_deviation
        
        scalars = np.array([
        float(b_frac), float(c_frac), float(offset),
        float(base_frac_dT), float(b_frac_dT), float(c_frac_dT), float(offset_dT)])
    else:
        ## New as of August 4th, doing joint fit for the ratio between b- and c-type transitions
        ## I think that this has to be non-linear, so switching to scipy.optimize
        if args.fit_spec and not args.fit_dT :
            from scipy.optimize import least_squares
            
            # Apply Gaussian filters and crop
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

            scalars = np.array([
                float(b_frac), float(c_frac), float(offset),
                np.nan, np.nan, np.nan])
            

        if not args.fit_spec and args.fit_dT:
            from scipy.optimize import minimize
            # Apply Gaussian filters and crop
            spec_b = gaussian_filter(model_flux_b[c:-c], gf)
            spec_c = gaussian_filter(model_flux_c[c:-c], gf)
            measurement = data_flux[c:-c]
            noise = noise_std[c:-c]

            spec_dT_b = gaussian_filter(model_flux_dT_b[c:-c]-model_flux_b[c:-c], gf)
            spec_dT_c = gaussian_filter(model_flux_dT_c[c:-c]-model_flux_c[c:-c], gf)
            measurement_dT = data_flux_dT[c:-c]
            noise_dT = noise_std_dT[c:-c]

            # Residual function for minimize (only depends on ratio_bc)
            def objective(ratio_bc):
                # Build design matrix for spec fit: gamma * (spec_b + ratio_bc * spec_c) + offset_spec

                base_spec = spec_b + ratio_bc * spec_c
                delta_spec = spec_dT_b + ratio_bc * spec_dT_c
                X_spec = np.vstack([
                    base_spec,
                    np.ones_like(base_spec)
                ]).T
                y_spec = measurement

                # Weighted linear least squares for spec
                W_spec = 1.0 / noise
                Xw_spec = X_spec * W_spec[:, None]
                yw_spec = y_spec * W_spec
                coeffs_spec, _, _, _ = np.linalg.lstsq(Xw_spec, yw_spec, rcond=None)
                gamma, offset_spec = coeffs_spec

                # Build design matrix for dT fit:            
                X_dT = np.vstack([
                    base_spec,
                    delta_spec,
                    np.ones_like(base_spec)
                ]).T
                y_dT = measurement_dT

                # Weighted linear least squares for dT
                W_dT = 1.0 / noise_dT
                Xw_dT = X_dT * W_dT[:, None]
                yw_dT = y_dT * W_dT
                coeffs_dT, _, _, _ = np.linalg.lstsq(Xw_dT, yw_dT, rcond=None)
                alpha_dT, beta_dT, offset_dT = coeffs_dT

                # Compute total chi-squared
                fit_spec = gamma * base_spec + offset_spec
                fit_dT = alpha_dT * base_spec + beta_dT * delta_spec + offset_dT
                chi2_spec = np.sum(((measurement - fit_spec) / noise) ** 2)
                chi2_dT = np.sum(((measurement_dT - fit_dT) / noise_dT) ** 2)

                return chi2_spec + chi2_dT

            # Run outer optimization over ratio_bc
            opt_result = minimize(objective, x0=[1.0], method='L-BFGS-B')

            # Optimal ratio_bc
            ratio_bc = opt_result.x[0]

            # Final linear fits with optimal ratio_bc
            # Main spectrum
            X_spec = np.vstack([
                spec_b + ratio_bc * spec_c,
                np.ones_like(spec_b)
            ]).T
            y_spec = measurement
            W_spec = 1.0 / noise
            # W_dT = np.ones_like(noise)
            Xw_spec = X_spec * W_spec[:, None]
            yw_spec = y_spec * W_spec
            gamma, offset_spec = np.linalg.lstsq(Xw_spec, yw_spec, rcond=None)[0]

            # dT spectrum
            base_spec = spec_b + ratio_bc * spec_c
            delta_spec = spec_dT_b + ratio_bc * spec_dT_c
            X_dT = np.vstack([
                base_spec,
                delta_spec,
                np.ones_like(base_spec)
            ]).T
            y_dT = measurement_dT
            W_dT = 1.0 / noise_dT
            # W_dT = np.ones_like(noise_dT)

            Xw_dT = X_dT * W_dT[:, None]
            yw_dT = y_dT * W_dT
            alpha_dT, beta_dT, offset_dT = np.linalg.lstsq(Xw_dT, yw_dT, rcond=None)[0]

            # Evaluate fits
            fit = gamma * (spec_b + ratio_bc * spec_c) + offset_spec
            fit_dT = alpha_dT * base_spec + beta_dT * delta_spec + offset_dT
            chi2 += np.sum(((measurement - fit) / noise) ** 2)
            chi2 += np.sum(((measurement_dT - fit_dT) / noise_dT) ** 2)

            # print( ratio_bc )
            # Output scalar parameters
            scalars = np.array([
                float(gamma), float(ratio_bc), float(offset_spec),
                float(alpha_dT), float(beta_dT), float(offset_dT)
            ])

        if args.fit_spec and args.fit_dT and args.nonlinear_fit: 
            from scipy.optimize import least_squares
            
            # Apply Gaussian filters and crop
            spec_b = gaussian_filter(model_flux_b[c:-c], gf)
            spec_c = gaussian_filter(model_flux_c[c:-c], gf)
            measurement = data_flux[c:-c]
            noise = noise_std[c:-c]

            spec_dT_b = gaussian_filter(model_flux_dT_b[c:-c]-model_flux_b[c:-c], gf)
            spec_dT_c = gaussian_filter(model_flux_dT_c[c:-c]-model_flux_c[c:-c], gf)
            measurement_dT = data_flux_dT[c:-c]
            noise_dT = noise_std_dT[c:-c]

            # Define residuals function
            def residuals(params):
                # Unpack parameters
                gamma, ratio_bc, offset_spec, alpha_dT, beta_dT, offset_dT = params

                # First fit residuals
                fit = gamma * ( spec_b + ratio_bc * spec_c ) + offset_spec
                chi = (measurement - fit) / noise

                # dT fit residuals
                base_spec = spec_b + ratio_bc * spec_c
                fit_dT = (
                    alpha_dT * base_spec + #primary spectrum
                    beta_dT * ( spec_dT_b  +  ratio_bc * spec_dT_c ) +  offset_dT #dT spectrum
                )
                chi_dT = (measurement_dT - fit_dT) / noise_dT

                return np.concatenate([chi, chi_dT])

            # Initial guess
            x0 = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]

            # Perform nonlinear least squares fit
            result = least_squares(residuals, x0)

            # Unpack results
            gamma, ratio_bc, offset_spec, alpha_dT, beta_dT, offset_dT = result.x
            # b_frac, c_frac, offset, base_frac_dT, b_frac_dT, c_frac_dT, offset_dT = gamma, gamma*ratio_bc, offset_spec, alpha_dT, beta_dT, beta_dT*ratio_bc, offset_dT 

            # Evaluate fits if needed
            fit = gamma * spec_b + gamma*ratio_bc * spec_c + offset_spec
            fit_dT = (
                alpha_dT * (spec_b + ratio_bc * spec_c) +
                beta_dT * ( spec_dT_b  +  ratio_bc * spec_dT_c ) +  offset_dT
            )
            chi2 += np.sum( ( (measurement_dT - fit_dT) / noise_dT )**2 )
            chi2 += np.sum( ( (measurement - fit) / noise )**2 )
            
            scalars = np.array([
            float(gamma), float(ratio_bc), float(offset_spec),
            float(alpha_dT), float(beta_dT), float(offset_dT)])

        ## this is now a function that only does ratio_bc non-linearly
        elif args.fit_spec and args.fit_dT and not args.nonlinear_fit:
            from scipy.optimize import minimize

            # Apply Gaussian filters and crop
            spec_b = gaussian_filter(model_flux_b[c:-c], gf)
            spec_c = gaussian_filter(model_flux_c[c:-c], gf)
            measurement = data_flux[c:-c]
            noise = noise_std[c:-c]

            spec_dT_b = gaussian_filter(model_flux_dT_b[c:-c]-model_flux_b[c:-c], gf)
            spec_dT_c = gaussian_filter(model_flux_dT_c[c:-c]-model_flux_c[c:-c], gf)
            measurement_dT = data_flux_dT[c:-c]
            noise_dT = noise_std_dT[c:-c]

            # Residual function for minimize (only depends on ratio_bc)
            def objective(ratio_bc):
                # Build design matrix for spec fit: gamma * (spec_b + ratio_bc * spec_c) + offset_spec

                base_spec = spec_b + ratio_bc * spec_c
                delta_spec = spec_dT_b + ratio_bc * spec_dT_c
                X_spec = np.vstack([
                    base_spec,
                    np.ones_like(base_spec)
                ]).T
                y_spec = measurement

                # Weighted linear least squares for spec
                W_spec = 1.0 / noise
                Xw_spec = X_spec * W_spec[:, None]
                yw_spec = y_spec * W_spec
                coeffs_spec, _, _, _ = np.linalg.lstsq(Xw_spec, yw_spec, rcond=None)
                gamma, offset_spec = coeffs_spec

                # Build design matrix for dT fit:            
                X_dT = np.vstack([
                    base_spec,
                    delta_spec,
                    np.ones_like(base_spec)
                ]).T
                y_dT = measurement_dT

                # Weighted linear least squares for dT
                W_dT = 1.0 / noise_dT
                Xw_dT = X_dT * W_dT[:, None]
                yw_dT = y_dT * W_dT
                coeffs_dT, _, _, _ = np.linalg.lstsq(Xw_dT, yw_dT, rcond=None)
                alpha_dT, beta_dT, offset_dT = coeffs_dT

                # Compute total chi-squared
                fit_spec = gamma * base_spec + offset_spec
                fit_dT = alpha_dT * base_spec + beta_dT * delta_spec + offset_dT
                chi2_spec = np.sum(((measurement - fit_spec) / noise) ** 2)
                chi2_dT = np.sum(((measurement_dT - fit_dT) / noise_dT) ** 2)

                return chi2_spec + chi2_dT

            # Run outer optimization over ratio_bc
            opt_result = minimize(objective, x0=[1.0], method='L-BFGS-B')

            # Optimal ratio_bc
            ratio_bc = opt_result.x[0]

            # Final linear fits with optimal ratio_bc
            # Main spectrum
            X_spec = np.vstack([
                spec_b + ratio_bc * spec_c,
                np.ones_like(spec_b)
            ]).T
            y_spec = measurement
            W_spec = 1.0 / noise
            # W_dT = np.ones_like(noise)
            Xw_spec = X_spec * W_spec[:, None]
            yw_spec = y_spec * W_spec
            gamma, offset_spec = np.linalg.lstsq(Xw_spec, yw_spec, rcond=None)[0]

            # dT spectrum
            base_spec = spec_b + ratio_bc * spec_c
            delta_spec = spec_dT_b + ratio_bc * spec_dT_c
            X_dT = np.vstack([
                base_spec,
                delta_spec,
                np.ones_like(base_spec)
            ]).T
            y_dT = measurement_dT
            W_dT = 1.0 / noise_dT
            # W_dT = np.ones_like(noise_dT)

            Xw_dT = X_dT * W_dT[:, None]
            yw_dT = y_dT * W_dT
            alpha_dT, beta_dT, offset_dT = np.linalg.lstsq(Xw_dT, yw_dT, rcond=None)[0]

            # Evaluate fits
            fit = gamma * (spec_b + ratio_bc * spec_c) + offset_spec
            fit_dT = alpha_dT * base_spec + beta_dT * delta_spec + offset_dT
            chi2 += np.sum(((measurement - fit) / noise) ** 2)
            chi2 += np.sum(((measurement_dT - fit_dT) / noise_dT) ** 2)

            save = False
            if save:
                import random
                import string

                def generate_random_string(length):
                    """Generates a random string of specified length using letters and digits."""
                    characters = string.ascii_letters + string.digits
                    random_string = ''.join(random.choice(characters) for i in range(length))
                    return random_string

                # Example usage:
                random_str = generate_random_string(5)
                allf = np.vstack([fit, spec_b, spec_c, fit_dT, spec_dT_b, spec_dT_c])
                np.savetxt(f'temp_outputs/fit_spec_{random_str}.csv', allf)

            # print( ratio_bc )
            # Output scalar parameters
            scalars = np.array([
                float(gamma), float(ratio_bc), float(offset_spec),
                float(alpha_dT), float(beta_dT), float(offset_dT)
            ])
    
    return -0.5 * chi2, scalars

def model_log_likelihood_Cs(params, data_wavelength, data_flux, data_flux_dT, noise_std, noise_std_dT, central_wav = 15272):
    lp = log_prior(params)
    if not np.isfinite(lp):
        if args.use_scalar_prior:
            return -np.inf, np.zeros(7)
        else:
            return -np.inf, np.zeros(6)

    try:
        if args.B_not_equal_C:
            T, A, B, C, frac_A, frac_B, frac_C, lorentz_width = params
        else:
            T, A, C, frac_A, frac_C, lorentz_width = params
            B = C
            frac_B = frac_C
        
        spec_txt_b, base_b = generate_pgopher_input(T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, axis='b')
        _, model_flux_b = convolve_pgopher_spectrum(spec_txt_b, central_wav)

        spec_txt_c, base_c = generate_pgopher_input(T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, axis='c')
        _, model_flux_c = convolve_pgopher_spectrum(spec_txt_c, central_wav)

        if args.fit_dT:
            spec_txt_dT_b, base_dT_b = generate_pgopher_input(T + 0.05, A, B, C, frac_A, frac_B, frac_C, lorentz_width, axis='b')
            _, model_flux_dT_b = convolve_pgopher_spectrum(spec_txt_dT_b, central_wav)

            spec_txt_dT_c, base_dT_c = generate_pgopher_input(T + 0.05, A, B, C, frac_A, frac_B, frac_C, lorentz_width, axis='c')
            _, model_flux_dT_c = convolve_pgopher_spectrum(spec_txt_dT_c, central_wav)
        else:
            model_flux_dT_b = np.zeros_like(data_flux)
            model_flux_dT_c = np.zeros_like(data_flux)

        lnlike, scalars = compute_loglikelihood_Cs(
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
        if args.use_scalar_prior:
            return -np.inf, np.zeros(7)
        else:
            return -np.inf, np.zeros(6)

def compute_loglikelihood_C2v(
    model_flux, model_flux_dT, data_flux, data_flux_dT, noise_std, noise_std_dT ):
    chi2 = 0.0
    if args.use_direct and not args.old_errs:
        c = 10 + args.extra_truncation # edge crop
    else:
        c = 30 + args.extra_truncation

    gf = 0.01  # Gaussian filter width
    gamma = offset = 0
    alpha_dT = beta_dT = offset_dT = 0
    ratio_bc = np.nan

    if args.fit_spec:
        # Apply Gaussian filter and crop edges
        spec = gaussian_filter(model_flux[c:-c], gf) 
        measurement = data_flux[c:-c]
        noise = noise_std[c:-c]

        # Fit linear combination: b_frac * spec_b + c_frac * spec_c + offset
        M = np.vstack([spec, np.ones_like(spec)]).T
        coeffs, _, _, _ = np.linalg.lstsq(M, measurement, rcond=None)
        gamma, offset = coeffs

        # Evaluate fit
        fit = gamma * spec + offset
        chi = (measurement - fit) / noise
        chi2 += np.sum(chi ** 2)

    if args.fit_dT:
        # Apply Gaussian filter and crop edges
        spec = gaussian_filter(model_flux[c:-c], gf)
        spec_dT = gaussian_filter(model_flux_dT[c:-c], gf) - spec
        measurement_dT = data_flux_dT[c:-c]
        noise_dT = noise_std_dT[c:-c]

        # Form matrix: linear combo of original + delta spectra
        M_dT = np.vstack([spec, spec_dT, np.ones_like(spec)]).T
        coeffs_dT, _, _, _ = np.linalg.lstsq(M_dT, measurement_dT, rcond=None)
        alpha_dT, beta_dT, offset_dT = coeffs_dT

        # Evaluate fit
        fit_dT = alpha_dT * spec + beta_dT * spec_dT+ offset_dT

        chi_dT = (measurement_dT - fit_dT) / noise_dT
        chi2 += np.sum(chi_dT ** 2)
    save = False
    if save:
        import random
        import string

        def generate_random_string(length):
            """Generates a random string of specified length using letters and digits."""
            characters = string.ascii_letters + string.digits
            random_string = ''.join(random.choice(characters) for i in range(length))
            return random_string

        # Example usage:
        random_str = generate_random_string(5)
        allf = np.vstack([fit, spec, fit_dT, spec_dT])
        np.savetxt(f'temp_outputs/fit_spec_{random_str}.csv', allf)

    # Output scalar parameters
    scalars = np.array([
        float(gamma), float(ratio_bc), float(offset),
        float(alpha_dT), float(beta_dT), float(offset_dT)
    ])

    return -0.5 * chi2, scalars

def model_log_likelihood_C2v(params, data_wavelength, data_flux, data_flux_dT, noise_std, noise_std_dT, central_wav = 15272):
    lp = log_prior(params)
    if not np.isfinite(lp):
        if args.use_scalar_prior:
            return -np.inf, np.zeros(7)
        else:
            return -np.inf, np.zeros(6)

    try:
        if args.B_not_equal_C:
            T, A, B, C, frac_A, frac_B, frac_C, lorentz_width = params
        else:
            T, A, C, frac_A, frac_C, lorentz_width = params
            B = A
            frac_B = frac_A
        
        spec_txt, base = generate_pgopher_input(T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, axis='a')
        _, model_flux_b = convolve_pgopher_spectrum(spec_txt, central_wav)

        if args.fit_dT:
            spec_txt_dT, base_dT = generate_pgopher_input(T + 0.05, A, B, C, frac_A, frac_B, frac_C, lorentz_width, axis='a')
            _, model_flux_dT = convolve_pgopher_spectrum(spec_txt_dT, central_wav)
        else:
            model_flux_dT = np.zeros_like(data_flux)

        lnlike, scalars = compute_loglikelihood_C2v(
                model_flux_b, model_flux_dT,
                data_flux, data_flux_dT,
                noise_std, noise_std_dT)
        
        # print( np.hstack([[-2*lnlike], scalars, [scalars[3]/0.31/scalars[0], scalars[4]/0.05/0.31], params]) )
        if args.fit_spec:
            base = osp.basename(spec_txt)  # e.g. spec_T20.005_A0.0026984_...txt
            if base.startswith("spec_") and base.endswith(".txt"):
                param_str = base[len("spec_"):-len(".txt")]  # strip prefix/suffix
                temp_pgo_file = os.path.join(TEMP_DIR, f'temp_{param_str}.pgo')

                # Clean up
                os.remove(spec_txt)
                os.remove(temp_pgo_file)

        if args.fit_dT:
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
        if args.use_scalar_prior:
            return -np.inf, np.zeros(7)
        else:
            return -np.inf, np.zeros(6)

# Clear TEMP_DIR on start
for file in Path(TEMP_DIR).iterdir():
    if file.is_file():
        file.unlink()

if args.B_not_equal_C:
    ndim = 8
    if args.symmetry_group == 'Cs':
        p0_center = [20, 0.02, 0.004, 0.003, 0.999, 0.999, 0.999, 0.1] # T, A, B, C, frac_A, frac_B, frac_C, lorentz
        step_scales = [15, 0.005, 0.0015, 0.0015, 0.001, 0.001, 0.001, 0.05]
        # step_scales = [0.1, 0.000005, 0.0000015, 0.0000015, 0.000001, 0.000001, 0.000001]

    if args.symmetry_group == 'C2v':
        p0_center = [20, 0.003, 0.004, 0.02, 0.999, 0.999, 0.999, 0.1] # T, A, B, C, frac_A, frac_B, frac_C, lorentz
        step_scales = [15, 0.0015, 0.0015, 0.005, 0.001, 0.001, 0.001, 0.05]
        # step_scales = [0.1, 0.000005, 0.0000015, 0.0000015, 0.000001, 0.000001, 0.000001]
else:
    ndim = 6
    if args.symmetry_group == 'Cs':
        p0_center = [20, 0.02, 0.003, 0.99, 0.95, 0.1]  # T, A, BC, frac_AB, frac_C
        step_scales = [15, 0.005, 0.0015, 0.001, 0.001, 0.05]
    if args.symmetry_group == 'C2v': ## pay attention to how the nomenclature changes!
        p0_center = [20, 0.003, 0.02, 0.99, 0.99, 0.1]  # T, AB, C, frac_AB, frac_C
        step_scales = [15, 0.0015, 0.005, 0.001, 0.001, 0.05]
    

nsteps = args.nsteps
ncpu_to_use = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else max(1, os.cpu_count())
nwalkers = ncpu_to_use
print(f"Using {ncpu_to_use} CPUs")

fudge = float(args.fudge) #how much to inflate errors that we may not believe in

DIB_15272 = h5py.File(osp.expanduser('~/DIB/new_errs/res_dib_15272.h5'), "r")
data_wavelength = DIB_15272['wav'][:]
data_flux = DIB_15272['mean'][:][:,0]
data_flux_dT = DIB_15272['mean'][:][:,1]

noise_std = fudge*np.sqrt(DIB_15272['var'][:][:,0])
noise_std_dT = fudge*np.sqrt(DIB_15272['var'][:][:,1])

if args.use_direct:
    data_flux = DIB_15272['mean'][:][:,0]
    data_flux_dT = DIB_15272['mean'][:][:,1]
    if args.old_errs:
        errs0 = h5py.File(osp.expanduser('~/DIB/jackknife_dib.h5'), "r")
        measurements = pd.read_csv(osp.expanduser('~/DIB/pca_version.txt'), sep='\s+', names=['wavelength', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'])
        data_wavelength = measurements['wavelength']
        data_flux = errs0['mean'][0,:,0]
        data_flux_dT = errs0['mean'][0,:,1]
        if args.cov:
            noise_std = errs0['cov'][:, :, 0]
            noise_std_dT = errs0['cov'][:, :, 1]
        else:
            noise_std = fudge * np.sqrt(errs0['var'][:, 0])
            noise_std_dT = np.sqrt(errs0['var'][:, 1])
    if args.plate_errs:
        errs0 = h5py.File(osp.expanduser('~/DIB/new_errs/jackknife_plates_dib_15272.h5'), "r")
        data_flux = errs0['mean'][:,0]
        data_flux_dT = errs0['mean'][:,1]
        if args.cov:
            noise_std = errs0['cov'][:, :, 0]
            noise_std_dT = errs0['cov'][:, :, 1]
        else:
            noise_std = fudge * np.sqrt(errs0['var'][:, 0])
            noise_std_dT = np.sqrt(errs0['var'][:, 1])

else:
    errs0 = h5py.File(osp.expanduser('~/DIB/jackknife_dib.h5'), "r")
    measurements = pd.read_csv(osp.expanduser('~/DIB/pca_version.txt'), sep='\s+', names=['wavelength', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'])
    data_flux = measurements['PC1_1'].values
    data_flux_dT = measurements['PC2_2'].values
    if args.cov:
        noise_std = errs0['cov'][:, :, 0]
        noise_std_dT = errs0['cov'][:, :, 1]
    else:
        noise_std = fudge * np.sqrt(errs0['var'][:, 0])
        noise_std_dT = np.sqrt(errs0['var'][:, 1])
        print(noise_std.shape)
        print(data_flux.shape)
        print(noise_std_dT.shape)
        print(data_flux_dT.shape)

backend_file = osp.expanduser(f"~/../../scratch/gpfs/cj1223/DIB/bc_run_{TEMP_SUFFIX}.h5")
if osp.exists(backend_file):
    os.remove(backend_file)  # Ensure clean start

backend = emcee.backends.HDFBackend(backend_file)

with get_context("fork").Pool(processes=ncpu_to_use) as pool:
    errtext = 'Using default, large errors'
    if args.old_errs:
        errtext = 'Using old, tight, errors'
    if args.plate_errs:
        errtext = 'Using (newer) errors jackknifed over approximately a plate size'

    print(
    f"Running MCMC with the following settings for {args.title} run:\n"
    f"- B and C treated as {'different' if args.B_not_equal_C else 'equal'}\n"
    f"- Fudge factor on noise: {args.fudge}\n"
    f"- Using {'direct' if args.use_direct else 'PCA'} spectrum data\n"
    f"- {'Flat priors' if args.flat_prior else 'Priors chosen by Andrew and I'}\n"
    f"- {'Using diagonal of covariance only' if not args.cov else 'Using full covariance'}\n"
    f"- Fitting main spectrum: {args.fit_spec}\n"
    f"- Fitting temperature derivative: {args.fit_dT}\n"
    f"- Using {args.symmetry_group} symmetry\n"
    f"- {'Doing non-linear scalar fits' if args.nonlinear_fit else 'Doing linear scalar fits'}\n"
    f"- Using {args.tau_prior} inverse cm as a prior slope on the exponential prior on Lorentzian broadening\n"
    f"- Using {args.alpha_prior} gaussian prior width for the vibrational stretch (alpha) in the rotational constants\n"
    f"- {errtext}\n"
    f"- Truncating the spectral fitting by {args.extra_truncation} extra wavelength elements\n"
    f"- {'Fitting b/c ratio with a prior' if args.use_scalar_prior else 'Doing joint, exact, b/c fits'}\n" ) 

    if args.symmetry_group == 'Cs':
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            model_log_likelihood_Cs,
            args=(data_wavelength, data_flux, data_flux_dT, noise_std, noise_std_dT),
            pool=pool,
            backend=backend
        )

    if args.symmetry_group == 'C2v':
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            model_log_likelihood_C2v,
            args=(data_wavelength, data_flux, data_flux_dT, noise_std, noise_std_dT),
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