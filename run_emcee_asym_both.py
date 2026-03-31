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

    parser.add_argument("-balance_errs", "--balance_errs",
                        type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=False,
                        help="If True, balance the errors on the DIB/dT profiles such that the relative errors on the two are the same")

    parser.add_argument("-emphasize_dT_center", "--emphasize_dT_center",
                        type=float,
                        nargs=2,
                        default=None,
                        metavar=('WIDTH', 'FACTOR'),
                        help="Lower errors at the center of the dT profile. Pass WIDTH FACTOR (e.g. --emphasize_dT_center 7 3). Default: disabled.")

    parser.add_argument("-err_factor", "--err_factor",
                        type=float,
                        default=1.0,
                        help="Divide both noise_std and noise_std_dT by this factor (default: 1.0)")

    def float_or_false(x):
        if x.lower() in ['False', 'false', '0', 'no']:
            return False
        try:
            return float(x)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a float or one of: False, false, 0, no")

    parser.add_argument(
        "-fit_peakcenter_offset", "--fit_peakcenter_offset",
        type=float_or_false,
        default=False,
        help="If any float is given, parsed as float; otherwise False. If float, that is the cm^-1 Gaussian prior width of the center of the peak"
    )

    parser.add_argument("-dib", "--dib",
                        type=str,
                        default='15272',
                        choices=['15272', '15672'],
                        help="Which DIB to fit (default: 15272)")

    return parser.parse_args()

args = parse_args()

assert not (args.old_errs and args.plate_errs), "Cannot both want to use old errors and new plate jackknifed errors. Set either --old_errs OR --plate_errs to True."

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

TEMP_SUFFIX = f"DIB{args.dib}_Symmetry{val_to_str(args.symmetry_group)}_BC{val_to_str(args.B_not_equal_C)}_F{val_to_str(args.fudge)}_D{val_to_str(args.use_direct)}_" + \
              f"Flat{val_to_str(args.flat_prior)}_Spec{val_to_str(args.fit_spec)}_dT{val_to_str(args.fit_dT)}_cov{val_to_str(args.cov)}_nonlin{val_to_str(args.nonlinear_fit)}" + \
              f'_tauSlope{val_to_str(args.tau_prior)}_alphaSlope{val_to_str(args.alpha_prior)}_err{errType}_trunc{args.extra_truncation}_balanceErr{args.balance_errs}_dTcenter{args.emphasize_dT_center}_centeroffset{val_to_str(args.fit_peakcenter_offset)}_{args.title}'

# DIB-specific constants (set after parsing args)
if args.dib == '15272':
    CENTRAL_INVCM_BASE = 1e8 / 15272.27178113337  # ~6544.44
    LSF_FILE = osp.expanduser('~/DIB/LSFs/lsf_15272.h5')
    DATA_FILE = osp.expanduser('~/DIB/new_errs/res_dib_15272.h5')
    PLATE_ERR_FILE = osp.expanduser('~/DIB/new_errs/jackknife_plates_dib_15272.h5')
else:  # 15672
    CENTRAL_INVCM_BASE = 6381.0
    LSF_FILE = osp.expanduser('~/DIB/LSFs/lsf_15672.h5')
    DATA_FILE = osp.expanduser('~/DIB/new_errs/res_dib_15672.h5')
    PLATE_ERR_FILE = osp.expanduser('~/DIB/new_errs/jackknife_plates_dib_15672.h5')

if args.symmetry_group == 'C2v':
    if args.dib == '15272':
        PGO_TEMPLATE = osp.expanduser("~/DIB/pgo_files/asym_top_15272_C2v.pgo")
    else:
        PGO_TEMPLATE = osp.expanduser("~/DIB/pgo_files/asym_top_15672_C2v_w_gauss.pgo")
else:
    PGO_TEMPLATE = osp.expanduser("~/DIB/pgo_files/asym_top_15272_Cs.pgo")  # fallback for Cs (not used directly)

TEMP_DIR = osp.expanduser(f"~/../../scratch/gpfs/cj1223/DIB/pgo_temppy_{TEMP_SUFFIX}")

os.makedirs(TEMP_DIR, exist_ok=True)
shutil.rmtree(TEMP_DIR, ignore_errors=False, onerror=None)
os.makedirs(TEMP_DIR, exist_ok=True)

def filename_base(T, A_base, B_base, C_base, frac_A, frac_B, frac_C, lorentz_width=0.01, axis='b', centeroffset=0.00):
    return f"T{T:.4f}_A{A_base:.7f}_B{B_base:.7f}_C{C_base:.7f}_FA{frac_A:.5f}_FB{frac_B:.5f}_FC{frac_C:.5f}_ax{axis}_lifetime{lorentz_width:.3f}_offset{centeroffset:.2f}"

def generate_pgopher_input_Cs(T, A_base, B_base, C_base, frac_A, frac_B, frac_C, lorentz_width=0.01, axis="a", center_invcm_offset=0.0):
    A_g, B_g, C_g = A_base, B_base, C_base
    A_e, B_e, C_e = A_base * frac_A, B_base * frac_B, C_base * frac_C
    center = CENTRAL_INVCM_BASE + center_invcm_offset
    base = filename_base(T, A_base, B_base, C_base, frac_A, frac_B, frac_C, axis=axis, centeroffset=center_invcm_offset)
    pgo_file = os.path.join(TEMP_DIR, f"temp_{base}.pgo")
    spec_txt = os.path.join(TEMP_DIR, f"spec_{base}.txt")
    if args.dib == '15272':
        if axis == 'a':
            PGO_TEMPLATE_CS = osp.expanduser("~/DIB/pgo_files/asym_top_15272_Cs_a.pgo")
        if axis == 'b':
            PGO_TEMPLATE_CS = osp.expanduser("~/DIB/pgo_files/asym_top_15272_Cs_b.pgo")
    else:  # 15672
        if axis == 'a':
            PGO_TEMPLATE_CS = osp.expanduser("~/DIB/pgo_files/asym_top_15672_Cs_a_w_gauss.pgo")
        if axis == 'b':
            PGO_TEMPLATE_CS = osp.expanduser("~/DIB/pgo_files/asym_top_15672_Cs_b_w_gauss.pgo")
    awk_script = f'''
    awk -v temp="{T}" \\
        -v A_ground="{A_g}" -v B_ground="{B_g}" -v C_ground="{C_g}" \\
        -v A_excited="{A_e}" -v B_excited="{B_e}" -v C_excited="{C_e}" \\
        -v axis="{axis}" -v lorentz_width="{lorentz_width}" -v center="{center}" '
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
    inside_excited && /<Parameter Name="Origin" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" center "\\"") }}
    inside_excited && /<Parameter Name="A" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" A_excited "\\"") }}
    inside_excited && /<Parameter Name="B" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" B_excited "\\"") }}
    inside_excited && /<Parameter Name="C" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" C_excited "\\"") }}
    /<Parameter Name="Lorentzian" Value="/ {{
        sub(/Value="[0-9.eE+-]+"/, "Value=\\"" lorentz_width "\\"")
    }}
    {{ print }}
    ' {PGO_TEMPLATE_CS} > {pgo_file}
    '''

    subprocess.run(awk_script, shell=True, check=True, executable="/bin/bash")
    subprocess.run([osp.expanduser("~/DIB/./pgo"), "--plot", pgo_file, spec_txt], check=True, stdout=subprocess.DEVNULL)
    return spec_txt, base

def generate_pgopher_input_C2v(T, A_base, B_base, C_base, frac_A, frac_B, frac_C,
                               lorentz_width=0.01, axis="a", center_invcm_offset=0.0):
    A_g, B_g, C_g = A_base, B_base, C_base
    A_e, B_e, C_e = A_base * frac_A, B_base * frac_B, C_base * frac_C
    center = CENTRAL_INVCM_BASE + center_invcm_offset
    base = filename_base(T, A_base, B_base, C_base, frac_A, frac_B, frac_C, axis=axis, centeroffset=center_invcm_offset)
    pgo_file = os.path.join(TEMP_DIR, f"temp_{base}.pgo")
    spec_txt = os.path.join(TEMP_DIR, f"spec_{base}.txt")

    if args.dib == '15272':
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
    else:  # 15672
        awk_script = f'''
    awk -v temp="{T}" \\
        -v A_ground="{A_g}" -v B_ground="{B_g}" -v C_ground="{C_g}" \\
        -v A_excited="{A_e}" -v B_excited="{B_e}" -v C_excited="{C_e}" \\
        -v axis="{axis}" -v lorentz_width="{lorentz_width}" -v center="{center}" '
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
    inside_excited && /<Parameter Name="Origin" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" center "\\"") }}
    inside_excited && /<Parameter Name="A" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" A_excited "\\"") }}
    inside_excited && /<Parameter Name="B" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" B_excited "\\"") }}
    inside_excited && /<Parameter Name="C" Value="/ {{ sub(/Value="[0-9.eE+-]+"/, "Value=\\"" C_excited "\\"") }}
    /<Parameter Name="Lorentzian" Value="/ {{
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

def get_pgopher_spectrum(spectrum_file, center_invcm_offset=0.0, dlam=0.01, window=8):
    """
    Load a PGOPHER spectrum and return it on the data wavelength grid.
    For 15672 C2v the LSF is already baked into the PGO template, so the spectrum
    is interpolated directly. For all other cases (15272, or Cs), the analytical
    LSF is convolved before interpolation.
    """
    inv_cm, flux = np.loadtxt(spectrum_file).T
    wav_pgo = 1e8 / inv_cm

    if args.use_direct and not args.old_errs:
        with h5py.File(LSF_FILE, 'r') as f:
            wav_load = f['wav'][:]
    else:
        measurements = pd.read_csv(osp.expanduser('~/DIB/pca_version.txt'), sep='\s+', names=['wavelength', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'])
        wav_load = measurements['wavelength'].values

    do_convolve = not (args.dib == '15672' and args.symmetry_group == 'C2v')

    if do_convolve:
        center_invcm = CENTRAL_INVCM_BASE + center_invcm_offset
        wavc = 1e8 / center_invcm
        wav_reg = np.arange(wavc - window, wavc + window, dlam)

        flux_interp = interp1d(wav_pgo, flux, bounds_error=False, fill_value=0.0)
        flux_reg = flux_interp(wav_reg)

        sig1 = 0.3
        sig2 = 1.85 * sig1
        sig3 = 9.5 * sig1
        f1 = 0.895
        f2 = 0.1
        c0 = 1.3e-3

        rel_grid = wav_reg - wavc
        p1 = 1/np.sqrt(2*np.pi*sig1**2)*np.exp(-rel_grid**2 / (2 * sig1**2))
        p2 = 1/np.sqrt(2*np.pi*sig2**2)*np.exp(-rel_grid**2 / (2 * sig2**2))
        p3 = 1/np.sqrt(2*np.pi*sig3**2)*np.exp(-rel_grid**2 / (2 * sig3**2))

        lsf_kernel = f1 * p1 + f2 * p2 + (1 - f1 - f2) * p3 + c0
        lsf_kernel /= np.sum(lsf_kernel)

        convolved_flux = convolve(flux_reg, lsf_kernel, mode='same')
        out_interp = interp1d(wav_reg, convolved_flux, bounds_error=False, fill_value=0.0)
    else:
        out_interp = interp1d(wav_pgo, flux, bounds_error=False, fill_value=0.0)

    flux_on_grid = out_interp(wav_load)
    return wav_load, flux_on_grid

def _unpack_Cs_params(params):
    """Unpack Cs params, returning (T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, center_offset) or None on length mismatch."""
    use_center_offset = isinstance(args.fit_peakcenter_offset, float)
    if args.B_not_equal_C:
        expected_len = 9 if use_center_offset else 8
        if len(params) != expected_len:
            return None
        if use_center_offset:
            T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, center_offset = params
        else:
            T, A, B, C, frac_A, frac_B, frac_C, lorentz_width = params
            center_offset = 0.0
    else:
        expected_len = 7 if use_center_offset else 6
        if len(params) != expected_len:
            return None
        if use_center_offset:
            T, A, C, frac_A, frac_C, lorentz_width, center_offset = params
        else:
            T, A, C, frac_A, frac_C, lorentz_width = params
            center_offset = 0.0
        B = C
        frac_B = frac_C
    return T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, center_offset

def log_prior_Cs_15272(params):
    unpacked = _unpack_Cs_params(params)
    if unpacked is None:
        return -np.inf
    T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, center_offset = unpacked
    use_center_offset = isinstance(args.fit_peakcenter_offset, float)

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

    if T <= 3 or T > 100: return -np.inf
    mu = np.log(25)
    sigma = 0.4
    temp_logprior = -np.log(T * sigma * np.sqrt(2 * np.pi)) - ((np.log(T) - mu) ** 2) / (2 * sigma ** 2)

    if C < 0.0005 or C > 0.3: return -np.inf
    if B < 0.0005 or B > 0.3: return -np.inf
    if not (0.0 <= lorentz_width <= 1.0): return -np.inf
    lorentz_width_prior = - (lorentz_width/args.tau_prior)  # exponential

    if B > A: return np.inf  # enforcing hierarchy
    if args.B_not_equal_C:
        if C >= B: return np.inf  # enforcing hierarchy

    if args.B_not_equal_C:
        C0 = (1/A+1/B)**(-1)
        CB_logprior = - ( (C-C0)/( np.sqrt(2) * 1 * C0 ) )**2
    else:
        CB_logprior = 0.0

    if A < 0.001 or A > 0.3: return -np.inf
    A_logprior = - ((0.013 - C)**2/(2*0.02**2))

    alpha_sig = args.alpha_prior
    if frac_A > 1: return -np.inf
    frac_a_logprior = - (frac_A - 1) ** 2 / (2*alpha_sig**2)
    if frac_B > 1: return -np.inf
    frac_b_logprior = - (frac_B - 1) ** 2 / (2*alpha_sig**2)
    if frac_C > 1: return -np.inf
    frac_c_logprior = -100 * (frac_C - 1) ** 2

    lp = temp_logprior + CB_logprior + A_logprior + frac_a_logprior + frac_b_logprior + frac_c_logprior + lorentz_width_prior
    if use_center_offset:
        if abs(center_offset) > 1: return -np.inf
        lp += -center_offset**2/(2*args.fit_peakcenter_offset**2)
    return lp

def log_prior_Cs_15672(params):
    unpacked = _unpack_Cs_params(params)
    if unpacked is None:
        return -np.inf
    T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, center_offset = unpacked
    use_center_offset = isinstance(args.fit_peakcenter_offset, float)

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

    if T <= 3 or T > 100: return -np.inf
    mu = np.log(15)
    sigma = 0.4
    temp_logprior = -np.log(T * sigma * np.sqrt(2 * np.pi)) - ((np.log(T) - mu) ** 2) / (2 * sigma ** 2)

    if not (0.0 <= lorentz_width <= 1.0): return -np.inf
    lorentz_width_prior = - (lorentz_width - 0.3)**2/(2*args.tau_prior**2)  # gaussian

    # Cs hierarchy: A > B >= C
    if B > A: return -np.inf
    if args.B_not_equal_C:
        if C >= B: return -np.inf

    if A < 0.0005 or A > 0.3: return -np.inf
    A_logprior = -(A - 0.08)**2/(2*0.02**2)

    if B < 0.0005 or B > 0.3: return -np.inf
    B_logprior = -(C - 0.006)**2/(2*0.003**2)

    if C < 0.0005 or C > 0.3: return -np.inf
    C_logprior = -(C - 0.004)**2/(2*0.003**2)

    alpha_sig = args.alpha_prior
    alphaA_fac = 1
    if frac_A > 1: return -np.inf
    frac_a_logprior = - (frac_A - 0.96) ** 2 / (2*(alphaA_fac*alpha_sig)**2)
    if frac_B > 1: return -np.inf
    frac_b_logprior = - (frac_B - 0.99) ** 2 / (2*alpha_sig**2)
    if frac_C > 1: return -np.inf
    frac_c_logprior = - (frac_C - 0.99) ** 2 / (2*alpha_sig**2)

    lp = temp_logprior + A_logprior + B_logprior + C_logprior + frac_a_logprior + frac_b_logprior + frac_c_logprior + lorentz_width_prior
    if use_center_offset:
        if abs(center_offset) > 1: return -np.inf
        lp += -(center_offset - 0.1)**2/(2*args.fit_peakcenter_offset**2)
    return lp

def log_prior_C2v_15272(params):
    use_center_offset = isinstance(args.fit_peakcenter_offset, float)
    if args.B_not_equal_C:
        expected_len = 9 if use_center_offset else 8
        if len(params) != expected_len:
            return -np.inf
        if use_center_offset:
            T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, center_offset = params
        else:
            T, A, B, C, frac_A, frac_B, frac_C, lorentz_width = params
    else:
        expected_len = 7 if use_center_offset else 6
        if len(params) != expected_len:
            return -np.inf
        if use_center_offset:
            T, A, C, frac_A, frac_C, lorentz_width, center_offset = params
        else:
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

        if use_center_offset:
            if abs(center_offset) > 1: return -np.inf
            log_center_offset_prior = -center_offset**2/(2*args.fit_peakcenter_offset**2)
            return temp_logprior + C_logprior + A_logprior + frac_a_logprior + frac_b_logprior + frac_c_logprior + lorentz_width_prior + log_center_offset_prior
        else:
            return temp_logprior + C_logprior + A_logprior + frac_a_logprior + frac_b_logprior + frac_c_logprior + lorentz_width_prior

def log_prior_C2v_15672(params):
    use_center_offset = isinstance(args.fit_peakcenter_offset, float)
    if args.B_not_equal_C:
        expected_len = 9 if use_center_offset else 8
        if len(params) != expected_len:
            return -np.inf
        if use_center_offset:
            T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, center_offset = params
        else:
            T, A, B, C, frac_A, frac_B, frac_C, lorentz_width = params
    else:
        expected_len = 7 if use_center_offset else 6
        if len(params) != expected_len:
            return -np.inf
        if use_center_offset:
            T, A, C, frac_A, frac_C, lorentz_width, center_offset = params
        else:
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
        mu = np.log(15)
        sigma = 0.4
        temp_logprior = -np.log(T * sigma * np.sqrt(2 * np.pi)) - ((np.log(T) - mu) ** 2) / (2 * sigma ** 2)

        if not (0.0 <= lorentz_width <= 1.0): return -np.inf

        lorentz_width_prior = - (lorentz_width-0.3)**2/(2*args.tau_prior**2) #gaussian

        if C>=B:
            return -np.inf # if enforcing hierarchy
        if args.B_not_equal_C:
            if B>=A:
                return -np.inf # if enforcing hierarchy

        if A < 0.0005 or A > 0.3: return -np.inf
        A_logprior = -(A - 0.08)**2/(2*0.02**2)

        if B < 0.0005 or B > 0.3: return -np.inf
        B_logprior = -(C - 0.006)**2/(2*0.003**2)

        if C < 0.0005 or C > 0.3: return -np.inf
        C_logprior = -(C - 0.004)**2/(2*0.003**2)

        alpha_sig = args.alpha_prior
        alphaA_fac = 1

        if frac_A > 1: return -np.inf
        frac_a_logprior = - (frac_A - 0.96) ** 2 / (2*(alphaA_fac*alpha_sig)**2)

        if frac_B > 1: return -np.inf
        frac_b_logprior = - (frac_B - 0.99) ** 2 / (2*alpha_sig**2)

        if frac_C > 1: return -np.inf
        frac_c_logprior = - (frac_C - 0.99) ** 2 / (2*alpha_sig**2)

        if use_center_offset:
            if abs(center_offset) > 1: return -np.inf
            log_center_offset_prior = -(center_offset-0.1)**2/(2*args.fit_peakcenter_offset**2)
            return temp_logprior + A_logprior + B_logprior + C_logprior + frac_a_logprior + frac_b_logprior + frac_c_logprior + lorentz_width_prior + log_center_offset_prior
        else:
            return temp_logprior + A_logprior + B_logprior + C_logprior + frac_a_logprior + frac_b_logprior + frac_c_logprior + lorentz_width_prior

if args.symmetry_group == 'Cs':
    if args.dib == '15272':
        log_prior = log_prior_Cs_15272
    else:
        log_prior = log_prior_Cs_15672
if args.symmetry_group == 'C2v':
    if args.dib == '15272':
        log_prior = log_prior_C2v_15272
    else:
        log_prior = log_prior_C2v_15672

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
            chi_spec = (measurement - fit) / noise
            chi2 += np.sum(chi_spec**2)

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
                ratio_direct = 0.1  # fallback

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

            # Optional: ratio constraint on the shape of the dT contributions
            ratio_dT = b_frac_dT / (c_frac_dT + 1e-10)
            ratio_tol = 0.01 * ratio_direct
            ratio_deviation = ((ratio_dT - ratio_direct) / ratio_tol) ** 2

            chi2 += np.sum(chi_dT**2)
            chi2 += ratio_deviation

        scalars = np.array([
        float(b_frac), float(c_frac), float(offset),
        float(base_frac_dT), float(b_frac_dT), float(c_frac_dT), float(offset_dT)])
    else:
        ## New as of August 4th, doing joint fit for the ratio between b- and c-type transitions
        ## I think that this has to be non-linear, so switching to scipy.optimize
        if args.fit_spec and not args.fit_dT:
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

            Xw_dT = X_dT * W_dT[:, None]
            yw_dT = y_dT * W_dT
            alpha_dT, beta_dT, offset_dT = np.linalg.lstsq(Xw_dT, yw_dT, rcond=None)[0]

            # Evaluate fits
            fit = gamma * (spec_b + ratio_bc * spec_c) + offset_spec
            fit_dT = alpha_dT * base_spec + beta_dT * delta_spec + offset_dT
            chi2 += np.sum(((measurement - fit) / noise) ** 2)
            chi2 += np.sum(((measurement_dT - fit_dT) / noise_dT) ** 2)

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

            # Output scalar parameters
            scalars = np.array([
                float(gamma), float(ratio_bc), float(offset_spec),
                float(alpha_dT), float(beta_dT), float(offset_dT)
            ])

    return -0.5 * chi2, scalars

def model_log_likelihood_Cs(params, data_wavelength, data_flux, data_flux_dT, noise_std, noise_std_dT):
    lp = log_prior(params)
    if not np.isfinite(lp):
        if args.use_scalar_prior:
            return -np.inf, np.zeros(7)
        else:
            return -np.inf, np.zeros(6)

    try:
        use_center_offset = isinstance(args.fit_peakcenter_offset, float)
        if args.B_not_equal_C:
            if use_center_offset:
                T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, center_offset = params
            else:
                T, A, B, C, frac_A, frac_B, frac_C, lorentz_width = params
                center_offset = 0.0
        else:
            if use_center_offset:
                T, A, C, frac_A, frac_C, lorentz_width, center_offset = params
            else:
                T, A, C, frac_A, frac_C, lorentz_width = params
                center_offset = 0.0
            B = C
            frac_B = frac_C

        spec_txt_b, base_b = generate_pgopher_input(T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, axis='a', center_invcm_offset=center_offset)
        _, model_flux_b = get_pgopher_spectrum(spec_txt_b, center_invcm_offset=center_offset)

        spec_txt_c, base_c = generate_pgopher_input(T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, axis='b', center_invcm_offset=center_offset)
        _, model_flux_c = get_pgopher_spectrum(spec_txt_c, center_invcm_offset=center_offset)

        if args.fit_dT:
            spec_txt_dT_b, base_dT_b = generate_pgopher_input(T + 0.05, A, B, C, frac_A, frac_B, frac_C, lorentz_width, axis='a', center_invcm_offset=center_offset)
            _, model_flux_dT_b = get_pgopher_spectrum(spec_txt_dT_b, center_invcm_offset=center_offset)

            spec_txt_dT_c, base_dT_c = generate_pgopher_input(T + 0.05, A, B, C, frac_A, frac_B, frac_C, lorentz_width, axis='b', center_invcm_offset=center_offset)
            _, model_flux_dT_c = get_pgopher_spectrum(spec_txt_dT_c, center_invcm_offset=center_offset)
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
    model_flux, model_flux_dT, data_flux, data_flux_dT, noise_std, noise_std_dT):
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
        fit_dT = alpha_dT * spec + beta_dT * spec_dT + offset_dT

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

def model_log_likelihood_C2v(params, data_wavelength, data_flux, data_flux_dT, noise_std, noise_std_dT):
    lp = log_prior(params)
    if not np.isfinite(lp):
        if args.use_scalar_prior:
            return -np.inf, np.zeros(7)
        else:
            return -np.inf, np.zeros(6)

    try:
        use_center_offset = isinstance(args.fit_peakcenter_offset, float)
        if args.B_not_equal_C:
            if use_center_offset:
                T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, center_offset = params
            else:
                T, A, B, C, frac_A, frac_B, frac_C, lorentz_width = params
                center_offset = 0.0
        else:
            if use_center_offset:
                T, A, C, frac_A, frac_C, lorentz_width, center_offset = params
            else:
                T, A, C, frac_A, frac_C, lorentz_width = params
                center_offset = 0.0
            B = A
            frac_B = frac_A

        spec_txt, base = generate_pgopher_input(T, A, B, C, frac_A, frac_B, frac_C, lorentz_width, axis='a', center_invcm_offset=center_offset)

        _, model_flux_b = get_pgopher_spectrum(spec_txt, center_invcm_offset=center_offset)

        if args.fit_dT:
            spec_txt_dT, base_dT = generate_pgopher_input(T + 0.05, A, B, C, frac_A, frac_B, frac_C, lorentz_width, axis='a', center_invcm_offset=center_offset)
            _, model_flux_dT = get_pgopher_spectrum(spec_txt_dT, center_invcm_offset=center_offset)
        else:
            model_flux_dT = np.zeros_like(data_flux)

        lnlike, scalars = compute_loglikelihood_C2v(
                model_flux_b, model_flux_dT,
                data_flux, data_flux_dT,
                noise_std, noise_std_dT)

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

use_center_offset = isinstance(args.fit_peakcenter_offset, float)

if args.B_not_equal_C:
    ndim = 8 + (1 if use_center_offset else 0)
    if args.symmetry_group == 'Cs':
        if args.dib == '15272':
            p0_center = [20, 0.02, 0.004, 0.003, 0.999, 0.999, 0.999, 0.1]
            step_scales = [15, 0.005, 0.0015, 0.0015, 0.001, 0.001, 0.001, 0.05]
        else:
            p0_center = [7, 0.02, 0.004, 0.003, 0.999, 0.999, 0.999, 0.1]
            step_scales = [15, 0.005, 0.0015, 0.0015, 0.001, 0.001, 0.001, 0.05]
    if args.symmetry_group == 'C2v':
        if args.dib == '15272':
            p0_center = [20, 0.003, 0.004, 0.02, 0.999, 0.999, 0.999, 0.1]
            step_scales = [15, 0.0015, 0.0015, 0.005, 0.001, 0.001, 0.001, 0.05]
        else:
            p0_center = [7, 0.08, 0.006, 0.004, 0.96, 0.99, 0.99, 0.3]
            step_scales = [5, 0.01, 0.0015, 0.001, 0.002, 0.001, 0.001, 0.05]
else:
    ndim = 6 + (1 if use_center_offset else 0)
    if args.symmetry_group == 'Cs':
        if args.dib == '15272':
            p0_center = [20, 0.02, 0.003, 0.99, 0.95, 0.1]
            step_scales = [15, 0.005, 0.0015, 0.001, 0.001, 0.05]
        else:
            p0_center = [20, 0.02, 0.003, 0.99, 0.95, 0.1]
            step_scales = [15, 0.005, 0.0015, 0.001, 0.001, 0.05]
    if args.symmetry_group == 'C2v':
        if args.dib == '15272':
            p0_center = [20, 0.003, 0.02, 0.99, 0.99, 0.1]
            step_scales = [15, 0.0015, 0.005, 0.001, 0.001, 0.05]
        else:
            p0_center = [7, 0.08, 0.04, 0.96, 0.99, 0.3]
            step_scales = [5, 0.005, 0.0015, 0.002, 0.001, 0.05]

if use_center_offset:
    p0_center = p0_center + [0.0]
    step_scales = step_scales + [0.05]

nsteps = args.nsteps
ncpu_to_use = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else max(1, os.cpu_count())
nwalkers = ncpu_to_use
print(f"Using {ncpu_to_use} CPUs")

fudge = float(args.fudge)  # how much to inflate errors that we may not believe in

DIB_file = h5py.File(DATA_FILE, "r")
data_wavelength = DIB_file['wav'][:]
data_flux = DIB_file['mean'][:][:,0]
data_flux_dT = DIB_file['mean'][:][:,1]

noise_std = fudge*np.sqrt(DIB_file['var'][:][:,0])
noise_std_dT = fudge*np.sqrt(DIB_file['var'][:][:,1])

if args.use_direct:
    data_flux = DIB_file['mean'][:][:,0]
    data_flux_dT = DIB_file['mean'][:][:,1]
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
        errs0 = h5py.File(PLATE_ERR_FILE, "r")
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

if args.emphasize_dT_center is not None:
    width, emphasize_fac = int(args.emphasize_dT_center[0]), args.emphasize_dT_center[1]
    center_idx = len(noise_std_dT)//2
    noise_std_dT[center_idx-width:center_idx+width] /= emphasize_fac

if args.balance_errs:
    peak_range = np.abs( np.nanmax(data_flux) - np.nanmin(data_flux) )
    peak_noise_med = np.nanmedian(noise_std)
    peak_rel_err = peak_noise_med/peak_range

    dT_range = np.abs( np.nanmax(data_flux_dT) - np.nanmin(data_flux_dT) )
    dT_noise_med = np.nanmedian(noise_std_dT)
    dT_rel_err = dT_noise_med/dT_range

    noise_std *= dT_rel_err/peak_rel_err

if args.err_factor != 1.0:
    noise_std /= args.err_factor
    noise_std_dT /= args.err_factor

backend_file = osp.expanduser(f"~/../../scratch/gpfs/cj1223/DIB/{args.dib}_run_{TEMP_SUFFIX}.h5")
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
    f"Running MCMC for DIB {args.dib} with the following settings for {args.title} run:\n"
    f"- B and C treated as {'different' if args.B_not_equal_C else 'equal'}\n"
    f"- Fudge factor on noise: {args.fudge}\n"
    f"- Using {'direct' if args.use_direct else 'PCA'} spectrum data\n"
    f"- {'Flat priors' if args.flat_prior else 'Priors chosen by Andrew and I'}\n"
    f"- {'Using diagonal of covariance only' if not args.cov else 'Using full covariance'}\n"
    f"- Fitting main spectrum: {args.fit_spec}\n"
    f"- Fitting temperature derivative: {args.fit_dT}\n"
    f"- Using {args.symmetry_group} symmetry\n"
    f"- Using {args.tau_prior} inverse cm as a prior on the Lorentzian broadening\n"
    f"- Using {args.alpha_prior} gaussian prior width for the vibrational stretch (alpha) in the rotational constants\n"
    f"- {errtext}\n"
    f"- Truncating the spectral fitting by {args.extra_truncation} extra wavelength elements\n"
    f"- {'Fitting a/b ratio with a prior if Cs' if args.use_scalar_prior else 'Doing joint, exact, a/b fits if Cs'}\n"
    f"- {'Doing non-linear scalar fits' if args.nonlinear_fit else 'Doing linear scalar fits'}\n"
    f"- {f'Reducing dT center errors by factor {args.emphasize_dT_center[1]} over width {int(args.emphasize_dT_center[0])}' if args.emphasize_dT_center is not None else 'Not emphasizing dT center'}\n"
    f"- {'Balancing errors between DIB profile and delta-T profile' if args.balance_errs else 'Keeping original errors'}\n"
    f"- {f'Letting peak center offset vary with a Gaussian prior with width {args.fit_peakcenter_offset} inv cm' if isinstance(args.fit_peakcenter_offset, float) else 'Peak center offset fixed (not fitted)'}\n")

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
