import numpy as np
from scipy.stats import gamma, norm, beta, truncnorm, lognorm
from scipy.special import logsumexp

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

def transform_clr(x, hyperparameters):
    ## Centred-log-ratio (CLR) prior on the abundances of n species (Benneke & Seager 2012, ApJ 753, 100).
    ## x are the unit-cube values of the first n-1 species (shape (..., n-1)); the n-th species is dependent,
    ## because the CLR variables xi_i = ln(X_i / g(X)) (g: geometric mean) must add up to zero.
    ## hyperparameters is log10 of the minimum abundance allowed for any species (e.g., -12).
    ## Returns the log10 abundances of all n species (shape (..., n)), which add up to 1. Points outside the
    ## allowed simplex (any abundance below the minimum) are NOT removed here; the caller has to reject them.
    x = np.asarray(x, dtype=float)
    n = x.shape[-1] + 1
    ln_xmin = hyperparameters * np.log(10.)
    ## Bounds on xi_i, corresponding to X_i = X_min (with the rest equally abundant) and to X_i ~ 1 (with the rest at X_min).
    ## Same as in POSEIDON (MacDonald & Madhusudhan 2017).
    xi_low = ((n - 1.) / n) * (ln_xmin + np.log(n - 1.))
    xi_up = -((n - 1.) / n) * ln_xmin
    xi = xi_low + (xi_up - xi_low) * x
    xi = np.concatenate([xi, -np.sum(xi, axis=-1, keepdims=True)], axis=-1)
    return (xi - logsumexp(xi, axis=-1, keepdims=True)) / np.log(10.)

def clr_prior_volume(nfree, hyperparameters, nsamples=1000000, seed=42):
    ## Fraction of the unit cube (of the nfree = n-1 free CLR parameters) which maps inside the allowed simplex.
    ## Samples outside of it are rejected via the likelihood, so the evidence returned by the sampler has to be divided
    ## by this fraction to get the evidence under the (normalised) CLR prior. Returns the fraction and its Monte Carlo error.
    rng = np.random.default_rng(seed)
    ok = np.zeros(nsamples, dtype=bool)
    for i in range(0, nsamples, 100000):
        log_ab = transform_clr(rng.uniform(size=(min(100000, nsamples - i), nfree)), hyperparameters)
        ok[i:i + log_ab.shape[0]] = np.all(log_ab >= hyperparameters, axis=-1)
    frac = np.mean(ok)
    return frac, np.sqrt(frac * (1. - frac) / nsamples)