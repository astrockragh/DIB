from multiprocessing import Pool, cpu_count, get_context
import multiprocessing as mp
mp.set_start_method("fork", force=True)
import subprocess, h5py, time, os, emcee, shutil
import numpy as np
import pandas as pd

# Constants and paths
PGO_TEMPLATE = "asym_top_15272_C1.pgo"

JULIA_SCRIPT = "ingest_pgo_asymtop_emcee.jl"
TEMP_DIR = os.path.expanduser("~/../../scratch/gpfs/cj1223/pgo_temp_test")
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
    start = time.time()
    subprocess.run(awk_script, shell=True, check=True, executable="/bin/bash")
    s2 = time.time()
    print(np.round(s2-start, 5))
    # Run PGOPHER
    subprocess.run(["./pgo", pgo_file, "-o", spec_txt], check=True, stdout=subprocess.DEVNULL)
    print(np.round(time.time()-s2, 5))
    return spec_txt, base

def run_julia_convolution(spec_txt, base):
    output_h5 = os.path.join(TEMP_DIR, f"convolved_{base}")
    subprocess.run([
        "julia", JULIA_SCRIPT, spec_txt, output_h5
    ], check=True)
    return output_h5+'.h5'

def read_h5_spectrum(h5_file):
    with h5py.File(h5_file, "r") as f:
        # spectrum = f['spectra'][:, 0, 0, 0, 0, 0]
        spectrum = f['spectra'][:]
        wavelength = f['wavelengths'][:]
    return wavelength, spectrum

def compute_likelihood(model_flux, data_flux, noise_std):
    chi2 = np.sum((data_flux - model_flux)**2 / noise_std**2)
    return -0.5 * chi2

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

def model_log_likelihood(params, data_wavelength, data_flux, noise_std):

    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
        
    try:
        T, AB_base, C_base, frac_AB, frac_C = params
        spec_txt, base = generate_pgopher_input(T, AB_base, C_base, frac_AB, frac_C)
        print(spec_txt)
        h5_file = run_julia_convolution(spec_txt, base)
        model_wavelength, model_flux = read_h5_spectrum(h5_file)

        lnlike = compute_likelihood(model_flux, data_flux, noise_std)

        # Clean up
        # os.remove(spec_txt)
        # os.remove('pgo_temp/temp'+spec_txt[13:-4]+'.pgo')
        # os.remove(h5_file)
        
        return lnlike + lnlike
    except Exception as e:
        print(f"Error for params {params}: {e}")
        return -np.inf

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

DIB_15272 = h5py.File('jackknife_dib.h5', "r")
measurements = pd.read_csv('pca_version.txt', delim_whitespace = True, names = ['wavelength', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'])
data_wavelength = measurements['wavelength']
data_flux = DIB_15272['mean'][:][0,:,0]
data_flux_dT = DIB_15272['mean'][:][0,:,1]
fudge1 = 5
fudge2 = 1
noise_std = fudge1*np.sqrt(DIB_15272['var'][:][:,0])
noise_std_dT = fudge2*np.sqrt(DIB_15272['var'][:][:,1])

ndim = 5
nwalkers = 64
nsteps = 40
ncpu_to_use = 64
with get_context("fork").Pool(processes = ncpu_to_use) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model_log_likelihood,
    args=(data_wavelength, data_flux, noise_std), pool=pool)
    startm = time.time()
    p0_center = [20, 0.0023, 0.012, 0.98, 0.98]

    # Define relative scales per parameter
    step_scales = [10, 0.001, 0.002, 0.001, 0.001]  # T, AB, C, frac_AB, frac_C
    
    # Build walker initial positions
    p0 = np.array([
        p0_center + np.array(step_scales)/np.sqrt(nwalkers) * np.random.randn(len(p0_center))
        for _ in range(nwalkers)
    ])
    
    sampler.run_mcmc(p0, nsteps, progress=True)
    end = time.time()
    multi_time = end - startm
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))