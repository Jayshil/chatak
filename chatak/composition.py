import numpy as np
from scipy.stats import beta
from scipy.special import logsumexp

from utils import transform_uniform, transform_loguniform, transform_beta, transform_truncated_normal, \
                   transform_logituniform

try:
    from petitRADTRANS.chemistry.utils import compute_mean_molar_masses_from_volume_mixing_ratios
    from petitRADTRANS.chemistry.utils import volume_mixing_ratios2mass_fractions
    from petitRADTRANS.chemistry.utils import get_species_molar_mass
except ImportError:
    pass  # chatak.py already warns if petitRADTRANS is missing.

## Composition priors for free-chemistry retrievals (see composition_priors_spec.md).
##
## All priors are defined on volume mixing ratios (VMRs) of the bulk components, which sum to 1:
##   - the species listed in the priors with a 'clr' or 'dirichlet' distribution (e.g., line-CO, line-logH2O), and
##   - the background, i.e., the filling species of the load class, with fixed relative VMR weights.
##
## Schemes:
##   - 'clr'               : centred log-ratio prior on (listed species + background)
##   - 'dirichlet'         : symmetric Dirichlet(alpha) prior on (listed species + background)
##   - 'fraction_clr'      : prior on the non-background (metal) fraction fmetal, CLR on the listed species
##   - 'fraction_dirichlet': prior on the non-background (metal) fraction fmetal, Dirichlet on the listed species
##
## With n components on the simplex, only n-1 values are sampled; the last component is computed implicitly.
## The implicit component is the background in 'clr'/'dirichlet' (or the last listed species if there is
## no background), and the last listed species in the fraction schemes.

SCHEMES = ('clr', 'dirichlet', 'fraction_clr', 'fraction_dirichlet')
COMPOSITION_DISTRIBUTIONS = ('clr', 'dirichlet')
FRACTION_PRIOR_NAME = 'fmetal'

## Distributions allowed for the metal fraction, and their transforms from the unit cube.
F_TRANSFORMS = {
    'uniform': transform_uniform,
    'loguniform': transform_loguniform,
    'jeffreys': transform_loguniform,
    'truncatednormal': transform_truncated_normal,
    'beta': transform_beta,
    'logituniform': transform_logituniform,
}

# ------------------------------------------------------------------
#                 Mapping to the simplex
# ------------------------------------------------------------------

def clr_bounds(n, X_min):
    """Exact bounds (xi_min, xi_max) of the CLR coordinates for n components with floor X_min."""
    xi_max = ( (n - 1) / n ) * ( np.log( 1 - (n - 1) * X_min ) - np.log(X_min) )
    xi_min = ( (n - 1) / n ) * np.log( (n - 1) * X_min / (1 - X_min) )
    return xi_min, xi_max

def clr_to_simplex(xi, X_min):
    """Map n-1 CLR coordinates (last axis) to n VMRs. Returns (X, valid) where valid is False if any X < X_min."""
    xi = np.asarray(xi, dtype=float)
    ## The last coordinate is minus the sum of the others
    xi_full = np.concatenate([xi, -np.sum(xi, axis=-1, keepdims=True)], axis=-1)
    ## Working in log space to avoid overflows
    log_X = xi_full - logsumexp(xi_full, axis=-1, keepdims=True)
    valid = np.all(log_X >= np.log(X_min), axis=-1)
    return np.exp(log_X), valid

def dirichlet_from_cube(u, alpha, X_min):
    """Map n-1 unit-cube values (last axis) to an n-component symmetric Dirichlet(alpha) sample (stick-breaking)."""
    u = np.asarray(u, dtype=float)
    n = u.shape[-1] + 1
    v = beta.ppf(u, alpha, alpha * (n - 1 - np.arange(n - 1)))
    ones = np.ones(u.shape[:-1] + (1,))
    rem = np.concatenate([ones, np.cumprod(1 - v, axis=-1)], axis=-1)
    X_simplex = np.concatenate([v, ones], axis=-1) * rem
    valid = np.all(X_simplex >= X_min, axis=-1)
    return X_simplex, valid

# ------------------------------------------------------------------
#                 Configuration
# ------------------------------------------------------------------

def _validate_fraction_prior(f_prior):
    ## The metal fraction must lie in [0, 1], so only distributions with that support are allowed.
    dist = f_prior['distribution'].lower()
    hyp = f_prior['hyperparameters']
    if dist == 'fixed':
        ok = 0 < hyp < 1
    elif dist == 'uniform':
        a, b = hyp
        ok = 0 <= a < b <= 1
    elif dist in ['loguniform', 'jeffreys']:
        a, b = hyp
        ok = 0 < a < b <= 1
    elif dist == 'truncatednormal':
        _, sigma, a, b = hyp
        ok = (sigma > 0) and (0 <= a < b <= 1)
    elif dist == 'beta':
        a, b = hyp
        ok = (a > 0) and (b > 0)
    elif dist == 'logituniform':
        a, b = hyp
        ok = a < b
    else:
        raise ValueError(f"Distribution '{f_prior['distribution']}' is not allowed for the metal fraction. "
                         f"Use one of: fixed, uniform, loguniform, truncatednormal, beta, logituniform.")
    if not ok:
        raise ValueError(f"Hyperparameters {hyp} of the metal fraction prior ({dist}) do not restrict it to [0, 1].")

def build_config(scheme, species, background=None, X_min=1e-10, alpha=None, f_prior=None, prior_names=None,
                 f_name=FRACTION_PRIOR_NAME):
    """Validate and build the composition configuration.

    scheme: one of SCHEMES
    species: list of the listed (non-background) bulk species, in priors order
    background: dict of background species and their relative VMR weights (normalised here)
    X_min: VMR floor
    alpha: Dirichlet concentration (dirichlet schemes only)
    f_prior: prior dict ({'distribution', 'hyperparameters'}) on the metal fraction (fraction schemes only)
    prior_names: dict mapping each species to its prior name (default: the species name itself)
    f_name: prior name of the metal fraction
    """
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown composition scheme '{scheme}'; must be one of {SCHEMES}.")

    species = list(species)
    if len(species) == 0:
        raise ValueError('At least one species is needed for composition priors.')
    if len(set(species)) != len(species):
        raise ValueError(f'Duplicate species in composition priors: {species}.')

    background = {} if background is None else dict(background)
    overlap = set(species) & set(background.keys())
    if overlap:
        raise ValueError(f'Species {sorted(overlap)} cannot be both a sampled bulk species and a background (filling) species.')
    if any(q <= 0 for q in background.values()):
        raise ValueError('Background (filling) species weights must be positive.')
    total = float(np.sum(list(background.values()))) if background else 0.
    background = {b: float(q) / total for b, q in background.items()}

    fraction = scheme.startswith('fraction')
    if fraction:
        if len(background) == 0:
            raise ValueError('Metal fraction priors require background (filling) species.')
        if f_prior is None:
            raise ValueError('Fraction schemes require a prior on the metal fraction.')
        _validate_fraction_prior(f_prior)

    ## Components on the simplex; 'bg' is the background (as a whole).
    if fraction or len(background) == 0:
        components = list(species)
    else:
        components = list(species) + ['bg']
    n = len(components)
    if len(species) + (len(background) > 0) < 2:
        raise ValueError('Composition priors need at least two bulk components (listed species + background).')

    if not (0 < X_min < 1. / max(n, 2)):
        raise ValueError(f'X_min = {X_min} must be in (0, 1/n) with n = {n} components.')
    if 'dirichlet' in scheme:
        if alpha is None or alpha <= 0:
            raise ValueError('Dirichlet priors need alpha > 0.')

    if prior_names is None:
        prior_names = {sp: sp for sp in species}

    sampled = components[:-1]
    implicit = components[-1]
    prefix = 'xi-' if 'clr' in scheme else 'u-'

    cfg = {
        'scheme': scheme,
        'species': species,
        'background': background,
        'X_min': float(X_min),
        'alpha': None if alpha is None else float(alpha),
        'f_prior': f_prior,
        'f_name': f_name if fraction else None,
        'n': n,
        'sampled': sampled,
        'implicit': implicit,
        'prior_names': dict(prior_names),
        'sampled_priors': [prior_names[sp] for sp in sampled],
        'implicit_prior': None if implicit == 'bg' else prior_names[implicit],
        'output_names': {prior_names[sp]: prefix + sp for sp in sampled},
        'xi_bounds': clr_bounds(n, X_min) if ('clr' in scheme and n > 1) else None,
    }
    return cfg

def composition_from_priors(priors, filling_species=None, filling_line=None, petit_vmr=False):
    """Detect composition priors in a chatak priors dict, validate them, and return the configuration (or None)."""
    comp_priors = [p for p, v in priors.items()
                   if p[0:4].lower() == 'line' and str(v['distribution']).lower() in COMPOSITION_DISTRIBUTIONS]
    f_names = [p for p in priors.keys() if p.lower() == FRACTION_PRIOR_NAME]
    f_name = f_names[0] if len(f_names) > 0 else None

    if len(comp_priors) == 0:
        if f_name is not None:
            raise ValueError(f"A prior on '{f_name}' requires 'clr' or 'dirichlet' priors on the line species.")
        return None

    dists = set(priors[p]['distribution'].lower() for p in comp_priors)
    if len(dists) > 1:
        raise ValueError('All composition priors must use the same distribution (either all clr or all dirichlet).')
    dist = dists.pop()

    if not petit_vmr:
        raise ValueError('clr/dirichlet priors are defined on volume mixing ratios; set petit_vmr=True.')

    other_lines = [p for p in priors.keys() if p[0:4].lower() == 'line' and p not in comp_priors]
    if len(other_lines) > 0:
        raise ValueError(f'All line species must have {dist} priors when composition priors are used; got {other_lines}.')

    species, prior_names, X_mins, alphas = [], {}, [], []
    for p in comp_priors:
        if len(p.split('_')) > 1:
            raise ValueError(f'Composition prior {p} cannot be instrument-dependent.')
        name = p.split('-', 1)[1]
        is_log = name[0:3] == 'log'
        sp = name[3:] if is_log else name

        hyp = priors[p]['hyperparameters']
        if dist == 'clr':
            if np.ndim(hyp) != 0:
                raise ValueError(f'Hyperparameter of the clr prior on {p} must be the minimum abundance (a single number).')
            x_min = hyp
        else:
            if np.ndim(hyp) != 1 or len(hyp) != 2:
                raise ValueError(f'Hyperparameters of the dirichlet prior on {p} must be [alpha, minimum abundance].')
            alphas.append(float(hyp[0]))
            x_min = hyp[1]
        ## For line-log<species>, the minimum abundance is given in log10.
        X_mins.append(10 ** float(x_min) if is_log else float(x_min))

        species.append(sp)
        prior_names[sp] = p

    if not np.allclose(X_mins, X_mins[0], rtol=1e-8, atol=0.):
        raise ValueError(f'All composition priors must have the same minimum abundance; got {X_mins}.')
    if len(alphas) > 0 and not np.allclose(alphas, alphas[0], rtol=1e-8, atol=0.):
        raise ValueError(f'All dirichlet priors must have the same alpha; got {alphas}.')

    filling_species = {} if filling_species is None else filling_species
    filling_line = [] if filling_line is None else filling_line
    if any(len(fs.split('_')) > 1 for fs in filling_species.keys()):
        raise ValueError('Filling species cannot be instrument-dependent when composition priors are used.')
    missing = [fl for fl in filling_line if fl not in filling_species.keys()]
    if len(missing) > 0:
        raise ValueError(f'filling_line species {missing} must also be in filling_species.')

    f_prior = priors[f_name] if f_name is not None else None
    scheme = ('fraction_' + dist) if f_name is not None else dist

    return build_config(scheme, species, background=filling_species, X_min=X_mins[0],
                        alpha=alphas[0] if len(alphas) > 0 else None, f_prior=f_prior,
                        prior_names=prior_names, f_name=f_name)

# ------------------------------------------------------------------
#                 From sampled values to VMRs
# ------------------------------------------------------------------

def ndim(cfg):
    """Number of unit-cube dimensions used by the composition (including the metal fraction)."""
    return len(cfg['sampled']) + (1 if cfg['scheme'].startswith('fraction') else 0)

def params_to_vmr(theta, cfg):
    """Map sampled values to VMRs.

    theta: array (..., ndim); [fmetal, s_1, ..., s_k] for fraction schemes, otherwise [s_1, ..., s_k],
           where s are the CLR coordinates (xi) or the Dirichlet unit-cube values (u) of cfg['sampled'].
    Returns (X, valid): X is a dict of VMRs of all species (background expanded into its species), valid is a bool (array).
    """
    theta = np.asarray(theta, dtype=float)
    fraction = cfg['scheme'].startswith('fraction')
    if fraction:
        f, s = theta[..., 0], theta[..., 1:]
    else:
        s = theta

    if 'clr' in cfg['scheme']:
        X_simplex, valid = clr_to_simplex(s, cfg['X_min'])
    else:
        X_simplex, valid = dirichlet_from_cube(s, cfg['alpha'], cfg['X_min'])
        #valid = np.all(X_simplex >= cfg['X_min'], axis=-1)#np.ones(X_simplex.shape[:-1], dtype=bool)

    if fraction:
        X_bg = 1. - f
        X_species = f[..., None] * X_simplex
    elif len(cfg['background']) > 0:
        X_bg = X_simplex[..., -1]
        X_species = X_simplex[..., :-1]
    else:
        X_bg = None
        X_species = X_simplex

    X = {}
    for i, sp in enumerate(cfg['species']):
        X[sp] = X_species[..., i]
    for b, q in cfg['background'].items():
        X[b] = q * X_bg
    return X, valid

def theta_from_parameters(parameter_values, cfg):
    """Collect the sampled composition values from a dict of parameter values (keyed by prior names)."""
    cols = [parameter_values[p] for p in cfg['sampled_priors']]
    if cfg['scheme'].startswith('fraction'):
        cols = [parameter_values[cfg['f_name']]] + cols
    cols = np.broadcast_arrays(*[np.asarray(c, dtype=float) for c in cols])
    return np.stack(cols, axis=-1)

def _f_from_cube(u0, f_prior):
    dist = f_prior['distribution'].lower()
    if dist == 'fixed':
        return np.full(np.shape(u0), float(f_prior['hyperparameters']))
    return F_TRANSFORMS[dist](u0, f_prior['hyperparameters'])

def cube_to_params(u, cfg):
    """Map unit-cube values (..., ndim) to the sampled values (fmetal, xi or u)."""
    u = np.asarray(u, dtype=float)
    fraction = cfg['scheme'].startswith('fraction')
    s = u[..., 1:] if fraction else u
    if 'clr' in cfg['scheme']:
        xi_min, xi_max = cfg['xi_bounds']
        s = xi_min + s * (xi_max - xi_min)
    if fraction:
        f = _f_from_cube(u[..., 0], cfg['f_prior'])
        return np.concatenate([f[..., None], s], axis=-1)
    return s

def composition_from_cube(u, cfg):
    """Map unit-cube values (..., ndim) to (X, valid) for the scheme in cfg."""
    return params_to_vmr(cube_to_params(u, cfg), cfg)

def _from_cube(u, cfg, scheme):
    if cfg['scheme'] != scheme:
        raise ValueError(f"Configuration is for scheme '{cfg['scheme']}', not '{scheme}'.")
    return composition_from_cube(u, cfg)

## One function per scheme: (u_slice, config) -> (X_dict, valid)
def clr(u, cfg):
    return _from_cube(u, cfg, 'clr')

def dirichlet(u, cfg):
    return _from_cube(u, cfg, 'dirichlet')

def fraction_clr(u, cfg):
    return _from_cube(u, cfg, 'fraction_clr')

def fraction_dirichlet(u, cfg):
    return _from_cube(u, cfg, 'fraction_dirichlet')

def acceptance_fraction(cfg, nsamples=200000, seed=0):
    """Fraction of the CLR box that maps to valid compositions (1 for Dirichlet schemes).
    The evidence of a CLR run carries a factor equal to this fraction."""
    if len(cfg['sampled']) == 0:
        return 1.
    rng = np.random.default_rng(seed)

    if 'clr' in cfg['scheme']:
        xi_min, xi_max = cfg['xi_bounds']
        xi = xi_min + rng.random((nsamples, len(cfg['sampled']))) * (xi_max - xi_min)
        _, valid = clr_to_simplex(xi, cfg['X_min'])
    elif 'dirichlet' in cfg['scheme']:
        s = rng.random((nsamples, len(cfg['sampled'])))
        _, valid = dirichlet_from_cube(s, cfg['alpha'], cfg['X_min'])
    return float(np.mean(valid))

# ------------------------------------------------------------------
#                 Mass fractions and derived quantities
# ------------------------------------------------------------------

def background_molar_mass(background):
    """Molar mass of the background mixture, sum_b q_b m_b (q_b normalised here)."""
    total = np.sum(list(background.values()))
    return float(np.sum([q * get_species_molar_mass(b) for b, q in background.items()]) / total)

def vmr_to_mass_fractions(X, X_min):
    """Clip VMRs at X_min and convert them to mass fractions. Returns (X_clipped, mean molar mass, mass fractions)."""
    X_clip = {sp: np.maximum(np.asarray(x, dtype=float), X_min) for sp, x in X.items()}
    mu = compute_mean_molar_masses_from_volume_mixing_ratios(X_clip)
    w = volume_mixing_ratios2mass_fractions(X_clip, mu)
    return X_clip, mu, w

def derived_quantities(X, X_min, background=()):
    """Derived quantities for (arrays of) VMRs: X, log10 X, mass fractions w, mean molar mass mu,
    metal fraction fmetal (VMR of everything except the background) and Z (mass fraction of everything except H2 and He)."""
    X_clip, mu, w = vmr_to_mass_fractions(X, X_min)
    derived = {}
    for sp in X.keys():
        derived['X-' + sp] = np.asarray(X[sp], dtype=float)
    for sp in X.keys():
        derived['log10X-' + sp] = np.log10(X_clip[sp])
    for sp in X.keys():
        derived['w-' + sp] = w[sp]
    derived['mu'] = mu
    if len(background) > 0:
        derived['fmetal'] = 1. - np.sum([np.asarray(X[b], dtype=float) for b in background], axis=0)
    else:
        derived['fmetal'] = np.ones(np.shape(mu))
    derived['Z'] = np.sum([w[sp] for sp in X.keys() if sp not in ['H2', 'He']], axis=0) * np.ones(np.shape(mu))
    return derived
