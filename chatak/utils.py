import numpy as np
from scipy.stats import gamma, norm, beta, truncnorm, lognorm
from scipy.special import logsumexp, expit
try:
    from petitRADTRANS.physics import (
        cubic_spline_profile,
        dtdp_temperature_profile,
        flux2irradiance,
        linear_spline_profile,
        madhu_seager_2009,
        planck_function_cm,
        power_law_temperature_profile,
        temperature_profile_function_guillot_global,
        temperature_profile_function_isothermal,
        temperature_profile_function_ret_model
    )
except:
    pass


def to_builtin(value):
    # Convert numpy scalars/arrays (recursively) to plain python objects, so that repr() can be read back with ast.literal_eval.
    if isinstance(value, dict):
        return {to_builtin(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_builtin(v) for v in value)
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value

def generate_priors(params, dists, hyperps):
    priors = {}
    for param, dist, hyperp in zip(params, dists, hyperps):
        priors[param] = {}
        priors[param]['distribution'], priors[param][
            'hyperparameters'] = dist, hyperp
    return priors

def get_quantiles(dist, alpha=0.68, method='median'):
    """
    get_quantiles function
    DESCRIPTION
        This function returns, in the default case, the parameter median and the error%
        credibility around it. This assumes you give a non-ordered
        distribution of parameters.
    OUTPUTS
        Median of the parameter,upper credibility bound, lower credibility bound
    """
    ordered_dist = dist[np.argsort(dist)]
    param = 0.0
    # Define the number of samples from posterior
    nsamples = len(dist)
    nsamples_at_each_side = int(nsamples * (alpha / 2.) + 1)
    if (method == 'median'):
        med_idx = 0
        if (nsamples % 2 == 0.0):  # Number of points is even
            med_idx_up = int(nsamples / 2.) + 1
            med_idx_down = med_idx_up - 1
            param = (ordered_dist[med_idx_up] + ordered_dist[med_idx_down]) / 2.
            return param,ordered_dist[med_idx_up+nsamples_at_each_side],\
                   ordered_dist[med_idx_down-nsamples_at_each_side]
        else:
            med_idx = int(nsamples / 2.)
            param = ordered_dist[med_idx]
            return param,ordered_dist[med_idx+nsamples_at_each_side],\
                   ordered_dist[med_idx-nsamples_at_each_side]

def writepp(fout, posteriors, priors):
    
    fout.write('# {0:18} \t \t {1:12} \t \t {2:12} \t \t {3:12}\n'.format(
        'Parameter Name', 'Median', 'Upper 68 CI', 'Lower 68 CI'))
    
    for pname in posteriors['posterior_samples'].keys():
        if pname != 'unnamed' and pname != 'loglike':
            val, valup, valdown = get_quantiles(
                posteriors['posterior_samples'][pname])
            usigma = valup - val
            dsigma = val - valdown
            fout.write(
                '{0:18} \t \t {1:.10f} \t \t {2:.10f} \t \t {3:.10f}\n'.format(
                    pname, val, usigma, dsigma))

    # Derived quantities (e.g., abundances, mean molar mass from composition priors), if any.
    if 'derived_samples' in posteriors.keys():
        fout.write('# Derived quantities\n')
        for pname in posteriors['derived_samples'].keys():
            val, valup, valdown = get_quantiles(
                posteriors['derived_samples'][pname])
            usigma = valup - val
            dsigma = val - valdown
            fout.write(
                '{0:18} \t \t {1:.10f} \t \t {2:.10f} \t \t {3:.10f}\n'.format(
                    pname, val, usigma, dsigma))

    fout.close()

# Prior transforms for nested samplers:
def transform_uniform(x, hyperparameters):
    a, b = hyperparameters
    return a + (b-a)*x

def transform_loguniform(x, hyperparameters):
    a, b = hyperparameters
    la = np.log(a)
    lb = np.log(b)
    return np.exp(la + x * (lb - la))

def transform_normal(x, hyperparameters):
    mu, sigma = hyperparameters
    return norm.ppf(x, loc=mu, scale=sigma)

def transform_logituniform(x, hyperparameters):
    # Uniform in logit(y) = ln[y/(1-y)] between a and b; returns y in (0, 1).
    a, b = hyperparameters
    return expit(a + x * (b - a))

def transform_beta(x, hyperparameters):
    a, b = hyperparameters
    return beta.ppf(x, a, b)

def transform_exponential(x, hyperparameters):
    a = hyperparameters
    return gamma.ppf(x, a)

def transform_truncated_normal(x, hyperparameters):
    mu, sigma, a, b = hyperparameters
    ar, br = (a - mu) / sigma, (b - mu) / sigma
    return truncnorm.ppf(x, ar, br, loc=mu, scale=sigma)

def transform_modifiedjeffreys(x, hyperparameters):
    turn, hi = hyperparameters
    return turn * (np.exp((x + 1e-10) * np.log(hi / turn + 1)) - 1)


# --------------------- Temperature profile functions
def TP_isothermal(pressures, T_iso, **kwargs):
    return np.ones_like(pressures) * T_iso