import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as con
from krithika import plotstyles
from glob import glob
import seaborn as sns
from chatak import load, utils
import corner
import os

#import multiprocessing
#multiprocessing.set_start_method('fork')

sns.set_style('ticks')
nthreads = 32

# ---------------------------------------------------------
#                  Retrieval setup
#    M1) Model 1: H2/He-rich atmosphere with CO2 and CH4
#.                (we use Mass fractions)
# ---------------------------------------------------------
pout = os.getcwd() + '/Analysis/Retrieval_M1'

# ----------------------------------------
##        Planetary parameters
# ----------------------------------------
rprs, rprs_err = 0.01460, 0.00052 
rst, rst_err = 1.258, 0.019
mp, mp_err = 3.55, 0.95
surf_grav = ( con.G * mp * u.M_earth / ( ( rprs * rst * u.R_sun ) )**2 ).to(u.cm / u.s**2).value

# ----------------------------------------
##        Observational data
# ----------------------------------------

# Loading the data
instruments = ['F322W2', 'F444W', 'F150W2-F322', 'F150W2-F444']
wav, dep, dep_err = {}, {}, {}
wav_band, res_func = {}, {}
for ins in instruments:
    fname = glob(os.getcwd() + f'/Data/M1/*{ins.lower()}_R250.txt')[0]
    # Spectrum data
    wav1, dep1, dep_err1 = np.loadtxt(fname, usecols=(0,2,3), unpack=True)
    wav[ins], dep[ins], dep_err[ins] = wav1, dep1, dep_err1

    # Instrumental response function data
    ins1 = ins.split('-')[0]
    wav_band[ins], res_func[ins] = np.loadtxt(os.getcwd() + f'/Data/M1/JWST_NIRCam.{ins1}.dat', usecols=(0,1), unpack=True)

# -------------------------------------------------
#    Method 2: Setting up the priors
# -------------------------------------------------

par_line = ['line-logCO2', 'line-logCH4']
dist_line = ['uniform', 'uniform']
hypers_line = [[-10., -2.], [-10., -2.]]

pars_pl = ['isotemp', 'rprs', 'mp', 'refP', 'rst']
dist_pl = ['uniform', 'fixed', 'fixed', 'loguniform', 'fixed']
hypers_pl = [[1000., 2000.], rprs, mp, [1e-4, 1e-2], rst]

par_ins, dist_ins, hypers_ins = [], [], []
for ins in instruments:
    par_ins = par_ins + ['offset_' + ins, 'sigma_w_' + ins]
    if ins != 'F444W':
        dist_ins = dist_ins + ['uniform', 'fixed']
        hypers_ins = hypers_ins + [[-100, 100], 0.]
    else:
        dist_ins = dist_ins + ['fixed', 'fixed']
        hypers_ins = hypers_ins + [0., 0.]

par_tot = par_line + pars_pl + par_ins
dist_tot = dist_line + dist_pl + dist_ins
hypers_tot = hypers_line + hypers_pl + hypers_ins

priors = utils.generate_priors(par_tot, dist_tot, hypers_tot)

## And the model
data = load(wavelength=wav, depth=dep, depth_err=dep_err, wav_band=wav_band, res_func=res_func,\
            priors=priors, pout=pout, mode='transmission', pressure_range=[-8, 3],\
            filling_species={'H2': 37, 'He': 12}, filling_line=[], rayleigh_species=['H2', 'He'],\
            resolution='c-k')
res = data.fit(sampler='dynamic_dynesty', nthreads=nthreads, dynesty_save_states=True, checkpoint_every=10*60)#, dynesty_resume=True, resume=True)

# ----------------------------------------
##        Plotting
# ----------------------------------------
for i in instruments:
    samples, model, _, _, wave_model = res.model.evaluate(i, return_err=True, return_samples=True, resolution=1000)

    ## Plot the model
    fig, axs = plt.subplots(figsize=(15, 5))
    axs.errorbar(wav[i], dep[i], yerr=dep_err[i], fmt='.', markersize=1., elinewidth=0.7, color='dodgerblue')
    axs.plot(wave_model, model, color='navy', zorder=10)

    for j in range(50):
        axs.plot(wave_model, samples[j, :], color='orangered', alpha=0.1, zorder=5)

    axs.set_xlabel('Wavelength [$\mu$m]')
    axs.set_ylabel('Transit Depth [ppm]')

    #axs.set_ylim([100, 700])

    plt.savefig(pout + f'/Full_model_{i}.png')

### Corner plot
post1 = res.posteriors['posterior_samples']

free_params = [p for p, d in zip(par_tot, dist_tot) if d != 'fixed']
data = np.vstack([ post1[fp] for fp in free_params ]).T

fig = corner.corner(data, labels=free_params, show_titles=True)
plt.savefig(pout + '/corner.png')#"""