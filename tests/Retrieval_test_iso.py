import numpy as np
import matplotlib.pyplot as plt
from chatak import load
from chatak import utils
import os

import multiprocessing
multiprocessing.set_start_method('fork')

# Loading the data
instruments = ['F322W2', 'F444W']
wav, dep, dep_err = {}, {}, {}
wav_band, res_func = {}, {}
for ins in instruments:
    # Spectrum data
    wav1, dep1, dep_err1 = np.loadtxt(os.getcwd() + f'/tests/Data/transit_spectrum_isothermal_{ins}_R50.txt', usecols=(0,2,3), unpack=True)
    wav[ins], dep[ins], dep_err[ins] = wav1, dep1, dep_err1

    # Instrumental response function data
    wav_band[ins], res_func[ins] = np.loadtxt(os.getcwd() + f'/tests/Data/JWST_NIRCam.{ins}.dat', usecols=(0,1), unpack=True)


# Priors
par = ['line-logCO2', 'line-logCH4', 'rayleigh-H2', 'rayleigh-He', 'isotemp', 'offset_F322W2', 'offset_F444W', 'rprs', 'rst', 'surfgrav', 'refP', 'sigma_w_F322W2', 'sigma_w_F444W']
dist = ['uniform', 'uniform', 'fixed', 'fixed', 'uniform', 'fixed', 'uniform', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed']
hypers = [[-10, -1], [-10, -1], 75, 25, [500, 2000], 0., [-100, 100], 0.01460199, 1.258, 866.14, 0.01, 0., 0.]

priors = utils.generate_priors(par, dist, hypers)

# Output directory
pout = os.getcwd() + '/tests/Analysis/Analysis_isothermal_R50_resolution/'

res = {'F322W2': 300, 'F444W': 200}

# Fitting
data = load(wavelength=wav, depth=dep, depth_err=dep_err, wav_band=wav_band, res_func=res_func,\
            priors=priors, pout=pout, mode='transmission', resolution=res)
res = data.fit(sampler='dynamic_dynesty', nthreads=14, dynesty_save_states=True, checkpoint_every=10*60)#, dynesty_resume=True)