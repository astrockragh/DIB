from multiprocessing import Pool, cpu_count, get_context
import multiprocessing as mp
mp.set_start_method("fork", force=True)
import subprocess, h5py, time, os, emcee, shutil
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d

# Constants and paths
PGO_TEMPLATE = "asym_top_15272_Cs.pgo"
TEMP_DIR = os.path.expanduser("~/../../scratch/gpfs/cj1223/pgo_temp_test_plot")
os.makedirs(TEMP_DIR, exist_ok=True)
shutil.rmtree(TEMP_DIR, ignore_errors=False, onerror=None)
os.makedirs(TEMP_DIR, exist_ok=True)

def filename_base(T, AB_base, C_base, frac_AB, frac_C):
    return f"T{T:.1f}_AB{AB_base:.6f}_C{C_base:.6f}_FAB{frac_AB:.4f}_FC{frac_C:.4f}"

def generate_pgopher_input(T, AB_base, C_base, frac_AB, frac_C):
    A_g = B_g = AB_base
    C_g = C_base
    A_e = B_e = AB_base * frac_AB
    C_e = C_base * frac_C

    base = filename_base(T, AB_base, C_base, frac_AB, frac_C)
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
    /<\\/AsymmetricManifold>/ {{ inside_ground = 0; inside_excited = 0 }}
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

    # Run PGOPHER
    start = time.time()
    subprocess.run(["./pgo", "--plot", pgo_file, spec_txt], check=True, stdout=subprocess.DEVNULL)
    print(f'pgopher run time {time.time()-start:.2f}')
    return spec_txt, base

def convolve_pgopher_spectrum(
    spectrum_file,
    center_wav,
    lsf_key='LCO+APO'
):
    """
    Convolve a PGOPHER output spectrum directly on its own wavelength grid, then
    interpolate the result onto the LSF wavelength grid.

    Parameters:
        spectrum_file (str): Path to PGOPHER output file.
        lsf_file (str): Path to LSF .h5 file.
        lsf_key (str): 'LCO', 'APO', or 'LCO+APO'.
        center_on (str or float): Center of the LSF in Å. Use 'auto' for mean of PGOPHER grid.
        normalize_lsf (bool): Normalize LSF before convolution.

    Returns:
        (wav_lsf, y_interp): Tuple of LSF wavelength grid and convolved+interpolated spectrum.
    """
    # Load PGOPHER spectrum
    inv_cm, flux = np.loadtxt(spectrum_file).T
    wav_pgo = 1e8 / inv_cm  # cm⁻¹ → Å
    
    lsf_file=f'LSFs/lsf_{center_wav}.h5'
    
    # Load LSF and its wavelength grid
    with h5py.File(lsf_file, 'r') as f:
        wav_lsf = f['wav'][:]
        lsf = f['LCO'][:] if lsf_key == 'LCO' else f['APO'][:]
        if lsf_key == 'LCO+APO':
            lsf = f['LCO'][:] + f['APO'][:]
    # Resample LSF onto PGOPHER wavelength grid
    lsf_interp = interp1d(wav_lsf, lsf, bounds_error=False, fill_value=0.0)
    lsf_on_pgo_grid = lsf_interp(wav_pgo)
    
    # print(np.max(lsf))
    # print(np.max(wav_pgo), np.max(wav_lsf))
    # print(np.max(lsf_on_pgo_grid))
    
    lsf_on_pgo_grid /= np.sum(lsf_on_pgo_grid)

    # Convolve on native grid
    convolved_flux = fftconvolve(flux, lsf_on_pgo_grid, mode='same')

    # Interpolate final result onto wav_lsf grid
    final_interp = interp1d(wav_pgo, convolved_flux, bounds_error=False, fill_value=0.0)
    flux_on_lsf_grid = final_interp(wav_lsf)

    return wav_lsf, flux_on_lsf_grid


def compute_likelihood(model_flux, data_flux, model_flux_dT, data_flux_dT, noise_std, noise_std_dT, c = 20):
    scalar = np.dot(data_flux[c:-c], model_flux[c:-c])/np.dot(model_flux[c:-c], model_flux[c:-c])
    chi2 = np.sum((data_flux[c:-c] - scalar*model_flux[c:-c])**2 / noise_std[c:-c]**2)

    scalar = np.dot(data_flux_dT[c:-c], model_flux_dT[c:-c])/np.dot(model_flux_dT[c:-c], model_flux_dT[c:-c])
    chi2_dT = np.sum((data_flux[c:-c] - scalar*model_flux[c:-c])**2 / noise_std[c:-c]**2)
    
    return -0.5 * (chi2+chi2_dT)

def log_prior(params):
    T, AB, C, frac_AB, frac_C = params

    # Log-normal prior on T
    if T <= 3 or T > 100:
        return -np.inf
    mu = np.log(30)
    sigma = 0.5
    temp_logprior = -np.log(T * sigma * np.sqrt(2 * np.pi)) - ((np.log(T) - mu)**2) / (2 * sigma**2)

    # AB flat prior: [0.001, 0.01]
    if AB < 0.001 or AB > 0.01:
        return -np.inf
    ab_logprior = 0.0

    # C flat prior: [0.004, 0.03]
    if C < 0.004 or C > 0.03:
        return -np.inf
    c_logprior = 0.0

    # Fraction AB prior: min at 1, falls off fast above 1
    if frac_AB > 1:
        return -np.inf
    frac_ab_logprior = -10 * (frac_AB - 1)**2  # steeper than Gaussian

    # Fraction C prior: min at 1, also steep falloff
    if frac_C > 1:
        return -np.inf
    frac_c_logprior = -10 * (frac_C - 1)**2

    return temp_logprior + ab_logprior + c_logprior + frac_ab_logprior + frac_c_logprior

def model_log_likelihood(params, data_wavelength, data_flux, data_flux_dT, noise_std, noise_std_dT, central_wav = 15272):

    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
        
    try:
        T, AB_base, C_base, frac_AB, frac_C = params
        spec_txt, base = generate_pgopher_input(T, AB_base, C_base, frac_AB, frac_C)
        wav_lsf, model_flux = convolve_pgopher_spectrum(spec_txt, central_wav)
        print(spec_txt)
        spec_txt_dT, base_dT = generate_pgopher_input(T+0.1, AB_base, C_base, frac_AB, frac_C)
        _, model_flux_dT = convolve_pgopher_spectrum(spec_txt_dT, central_wav)

        lnlike = compute_likelihood(model_flux, data_flux, model_flux_dT, data_flux_dT, noise_std, noise_std_dT)

        # Clean up
        # os.remove(spec_txt)
        # os.remove('pgo_temp/temp_'+base+'.pgo')

        return lnlike
    except Exception as e:
        print(f"Error for params {params}: {e}")
        return -np.inf

ndim = 5
nwalkers = 16

# Replace with your actual data

fudge1 = 5
fudge2 = 1
nsteps = 40
ncpu_to_use = 64

DIB_15272 = h5py.File('new_errs/res_dib_15272.h5', "r")
data_wavelength = DIB_15272['wav'][:]
data_flux = DIB_15272['mean'][:][:,0]
data_flux_dT = DIB_15272['mean'][:][:,1]

noise_std = fudge1*np.sqrt(DIB_15272['var'][:][:,0])
noise_std_dT = fudge2*np.sqrt(DIB_15272['var'][:][:,1])

start = time.time()
import multiprocessing as mp
mp.set_start_method("fork", force=True)
with get_context("fork").Pool(processes = ncpu_to_use) as pool:
    # Starting guess for each parameter
    p0_center = [20, 0.0023, 0.012, 1/1.002, 1/1.002]

    # Define relative scales per parameter
    step_scales = [10, 0.001, 0.002, 0.001, 0.001]  # T, AB, C, frac_AB, frac_C

    
    # Build walker initial positions
    p0 = np.array([
        p0_center + np.array(step_scales)/np.sqrt(nwalkers) * np.random.randn(len(p0_center))
        for _ in range(nwalkers)
    ])
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, model_log_likelihood,
        args=(data_wavelength, data_flux, data_flux_dT, noise_std, noise_std_dT, 15272),
        pool = pool
    )
    
    sampler.run_mcmc(p0, nsteps, progress=True)
stop = time.time()