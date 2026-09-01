import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chatak import load
from chatak import utils
import os

sns.set_style('ticks')

rst = 0.769

# Priors
par = ['line-logH2O', 'line-logCO', 'line-logCO2', 'line-logCH4', 'rayleigh-H2', 'rayleigh-He', 'isotemp', 'rprs', 'mp', 'refP', 'rst']
dist = ['fixed', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed']
hypers = [-1, -1, -1, -1, 75, 25, 743, 0.02663, 6.1, 0.001, rst]


priors = utils.generate_priors(par, dist, hypers)

# Output directory
pout = os.getcwd() + '/tests/Analysis/Analysis_isothermal_forward_HD_207496/'

# Fitting
data = load(priors=priors, pout=pout, cia=['H2--H2', 'H2--He'], mode='forward-transmission', pressure_range=[-6, 10])
wave, model = data.generate_forward_models()

plt.figure(figsize=(16/1.5, 9/1.5))
plt.plot(wave, model, color='navy', label='Forward Model', zorder=10)
plt.xscale('log')
plt.xlim(0.5, 5)
plt.xlabel('Wavelength (micron)')
plt.ylabel('Transit Depth (ppm)')

sns.despine()

plt.show()

# Save the spectrum data
#np.savetxt(pout + 'spectrum_toi5789b.txt', np.column_stack((models.model_spec['FORWARD']['wavelength'], models.model_spec['FORWARD']['spectrum'])), header='Wavelength (micron)  Transit Depth (ppm)')