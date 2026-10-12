import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as con
from krithika import plotstyles
import seaborn as sns
from chatak import load, utils
import os

sns.set_style('ticks')

# ---------------------------------------------------------
#                  Forward modeling
#    M1) Model 1: H2/He-rich atmosphere with CO2 and CH4
#.                (we use Mass fractions)
# ---------------------------------------------------------
pout = os.getcwd() + '/Analysis/Forward_M1'

# ----------------------------------------
##        Planetary parameters
# ----------------------------------------
rprs, rprs_err = 0.01460, 0.00052 
rst, rst_err = 1.258, 0.019
mp, mp_err = 3.55, 0.95
surf_grav = ( con.G * mp * u.M_earth / ( ( rprs * rst * u.R_sun ) )**2 ).to(u.cm / u.s**2).value

# -------------------------------------------------
#           Setting up the priors
# -------------------------------------------------

par_line = ['line-logCO2', 'line-logCH4']
dist_line = ['fixed', 'fixed']
hypers_line = [np.log10(1000e-6), np.log10(100e-6)]

pars_pl = ['isotemp', 'rprs', 'mp', 'refP', 'rst']
dist_pl = ['fixed' for _ in pars_pl]
hypers_pl = [1375, rprs, mp, 1e-3, rst]

par_tot = par_line + pars_pl
dist_tot = dist_line + dist_pl
hypers_tot = hypers_line + hypers_pl

priors = utils.generate_priors(par_tot, dist_tot, hypers_tot)

## And the model
data = load(priors=priors, pout=pout, mode='forward-transmission', pressure_range=[-8, 3],\
            filling_species={'H2': 37, 'He': 12}, filling_line=[], rayleigh_species=['H2', 'He'], cia=[],
            wavelength_range=[0.1, 5.5], resolution='c-k')
wave, model = data.generate_forward_models(save=False)


# -------------------------------------------------
#    Plotting the spectrum
# -------------------------------------------------
plt.figure(figsize=(15, 5))
plt.plot(wave, model, color='navy', label='Forward Model', zorder=10)
plt.xlabel('Wavelength [micron]')
plt.ylabel('Transit Depth [ppm]')
plt.title('Transmission Spectrum')
plt.legend()
plt.grid(True)
plt.show()