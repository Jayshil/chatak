import numpy as np
from scipy.stats import gamma, norm, beta, truncnorm, lognorm

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