import numpy as np
import ast

try:
    from petitRADTRANS.spectral_model import SpectralModel
except ImportError:
    print('Warning: petitRADTRANS not found. Forward models from petitRADTRANS cannot be used.')

try:
    from ultranest import ReactiveNestedSampler
except:
    print('Warning: UltraNest not found. UltraNest sampler cannot be used.')

import dynesty
from dynesty.utils import resample_equal
import astropy.units as u
import astropy.constants as con
from tqdm import tqdm
import pickle
import os
import multiprocessing
from .utils import *

log2pi = np.log(2. * np.pi)  # ln(2*pi)

class load(object):
    def __init__(self, wavelength=None, depth=None, depth_err=None, wav_band=None, res_func=None, priors=None, mode=None,\
                 pressure_range=[-6, 2], pressure_points=100, cia=[], resolution='c-k', code='petit', pout=None, pin=None, verbose=False):
        # Normal runs save wavelength-space inputs and per-instrument modes.
        # Forward runs only need priors and output location.
        self.wavelength = wavelength
        self.depth = depth
        self.depth_err = depth_err
        self.wav_band = wav_band
        self.res_func = res_func
        self.priors = priors
        self.pout = pout
        self.pin = pin
        self.mode = mode
        self.instruments = []
        self.resolution = resolution

        ## Pressure range and resolution (in termns of number of points) to create pressure grid to generate models.
        ## Pressure range is in log10(P) where P is in bar. The pressure grid will be created using np.logspace(pressure_range[0], pressure_range[1], pressure_points).
        self.pressure_range = pressure_range
        self.pressure_points = pressure_points
        
        # CIA (collision-induced absorption) opacities to include in the modeling. 
        # This should be a list of strings corresponding to the CIA species available in the opacity files (e.g., 'H2--H2', 'H2--He').
        # I think this is only used for petitRADTRANS, but we can keep it as a general input for now.
        self.cia = cia

        # The code to generate forward models; currently only petitRADTRANS is supported.
        # Use petit for petitRADTRANS and poseidon for POSEIDON (POSEIDON support is planned but not yet implemented).
        self.code = code

        self._spectrum_file = 'spectrum_data.txt'
        self._response_file = 'response_data.txt'
        self._priors_file = 'priors.txt'
        self.verbose = verbose

        if self.pin is not None:
            # Rerun mode: restore everything from the saved files.
            provided = [wavelength, depth, depth_err, wav_band, res_func, priors, pout, mode]
            if any(value is not None for value in provided):
                raise ValueError('When pin is provided, do not pass wavelength, depth, depth_err, wav_band, res_func, priors, pout, or mode.')
            if resolution != 'c-k' and resolution is not None:
                raise ValueError('When pin is provided, do not pass resolution explicitly; it will be restored from disk.')
            self.read()
            if self.pout is None:
                self.pout = self.pin
            if isinstance(self.mode, str):
                self.mode = self._normalize_mode(self.mode, self.instruments)
            if isinstance(self.resolution, (str, int, np.integer)):
                self.resolution = self._normalize_resolution(self.resolution, self.instruments)
            self.datadict_preparation()
            return

        if self.mode == 'forward-transmission' or self.mode == 'forward-emission':
            # Forward-model runs only need priors and an output directory.
            if any(value is not None for value in [wavelength, depth, depth_err, wav_band, res_func]):
                raise ValueError('Forward mode does not use wavelength, depth, depth_err, wav_band, or res_func.')
            if self.priors is None or self.pout is None:
                raise ValueError('Forward mode requires priors and pout.')
            self._validate_priors()
            if not os.path.exists(self.pout):
                os.makedirs(self.pout)
            self.instruments = ['FORWARD']
            self.mode = self._normalize_mode(self.mode, self.instruments)
            if self.resolution is not None and isinstance(self.resolution, (str, int, np.integer)):
                self.resolution = self._normalize_resolution(self.resolution, self.instruments)
            self.save()
            self.datadict_preparation()

            ## Wavelength ranges for forward runs are not defined by data, so we set them to None. The forward model code will need to handle this appropriately (e.g., by using the full wavelength range of the opacity files or a user-specified range).
            self.data_dict['FORWARD']['wav_min'] = 0.1
            self.data_dict['FORWARD']['wav_max'] = 30.

            # Initialize the models
            self.init_models()
            return

        if self.wavelength is None or self.depth is None or self.depth_err is None:
            raise ValueError('For a new run, wavelength, depth, and depth_err are required.')
        if self.priors is None or self.pout is None or self.mode is None:
            raise ValueError('For a new run, provide wavelength, depth, depth_err, priors, pout, and mode.')

        if self.wav_band is None and self.res_func is None:
            self.wav_band, self.res_func = {}, {}
        elif (self.wav_band is None) != (self.res_func is None):
            raise ValueError('wav_band and res_func must both be provided or both be None.')

        self.mode = self._normalize_mode(self.mode, self.wavelength.keys())
        self.resolution = self._normalize_resolution(self.resolution, self.wavelength.keys())
        if any(value is None for value in self.resolution.values()):
            raise ValueError('resolution must be provided for all instruments in normal runs.')
        self._validate_inputs()
        self.instruments = sorted(self.wavelength.keys())

        if not os.path.exists(self.pout):
            os.makedirs(self.pout)

        self.save()
        self.datadict_preparation()

        ## Calculating the wavelength ranges for each instrument and saving them in the data dictionary.
        for ins in self.instruments:
            self.data_dict[ins]['wav_min'] = np.min(self.wavelength[ins]) - ( 1.2 * np.median( np.diff( self.wavelength[ins] ) ) )
            self.data_dict[ins]['wav_max'] = np.max(self.wavelength[ins]) + ( 1.2 * np.median( np.diff( self.wavelength[ins] ) ) )

        # Initialize the models
        self.init_models()

    def _normalize_mode(self, mode, instrument_keys):
        # Allow either a single legacy mode string or a per-instrument mode dict.
        if isinstance(mode, str):
            if mode not in ['transmission', 'emission', 'forward-transmission', 'forward-emission']:
                raise ValueError("mode must be a dict of transmission/emission values or the string 'forward-transmission', 'forward-emission'.")
            return {ins: mode for ins in instrument_keys}
        if isinstance(mode, dict):
            if set(mode.keys()) != set(instrument_keys):
                raise ValueError('mode must have the same instrument keys as wavelength, depth, and depth_err.')
            normalized = {}
            for ins, value in mode.items():
                if value not in ['transmission', 'emission', 'forward-transmission', 'forward-emission']:
                    raise ValueError(f"Invalid mode for instrument {ins}: {value}")
                normalized[ins] = value
            return normalized
        raise TypeError("mode must be either a dict of instrument modes or the string 'forward'.")

    def _normalize_resolution(self, resolution, instrument_keys):
        # Allow a single resolution value/string for all instruments or a per-instrument dict.
        if resolution is None:
            return {ins: None for ins in instrument_keys}
        if isinstance(resolution, (str, int, np.integer)):
            return {ins: resolution for ins in instrument_keys}
        if isinstance(resolution, dict):
            if set(resolution.keys()) != set(instrument_keys):
                raise ValueError('resolution must have the same instrument keys as wavelength, depth, and depth_err.')
            normalized = {}
            for ins, value in resolution.items():
                if not isinstance(value, (str, int, np.integer)):
                    raise TypeError(f'resolution for instrument {ins} must be a string or integer.')
                normalized[ins] = value
            return normalized
        raise TypeError("resolution must be either a dict, a string, an integer, or None.")

    def _validate_priors(self):
        if not isinstance(self.priors, dict):
            raise TypeError('priors must be a dict with parameter names as keys.')
        for param, value in self.priors.items():
            if not isinstance(value, dict):
                raise TypeError(f'Prior entry for {param} must be a dict.')
            if set(value.keys()) != {'distribution', 'hyperparameters'}:
                raise ValueError(f'Prior entry for {param} must contain distribution and hyperparameters.')

    def _validate_inputs(self):
        # Validate normal-run spectral inputs and the per-instrument mode mapping.
        for name, value in [('wavelength', self.wavelength), ('depth', self.depth), ('depth_err', self.depth_err)]:
            if not isinstance(value, dict):
                raise TypeError(f'{name} must be a dict with instrument names as keys.')

        spec_keys = set(self.wavelength.keys())
        if set(self.depth.keys()) != spec_keys or set(self.depth_err.keys()) != spec_keys:
            raise ValueError('wavelength, depth, and depth_err must have identical instrument keys.')

        if not isinstance(self.mode, dict):
            raise TypeError('mode must be a dict with instrument names as keys for normal runs.')
        if set(self.mode.keys()) != spec_keys:
            raise ValueError('mode must have identical instrument keys as wavelength, depth, and depth_err.')

        if not isinstance(self.resolution, dict):
            raise TypeError('resolution must be a dict with instrument names as keys for normal runs.')
        if set(self.resolution.keys()) != spec_keys:
            raise ValueError('resolution must have identical instrument keys as wavelength, depth, and depth_err.')

        for ins in spec_keys:
            w = np.asarray(self.wavelength[ins], dtype=float)
            d = np.asarray(self.depth[ins], dtype=float)
            e = np.asarray(self.depth_err[ins], dtype=float)
            if not (w.ndim == d.ndim == e.ndim == 1):
                raise ValueError(f'Spectral arrays for instrument {ins} must be 1D.')
            if not (len(w) == len(d) == len(e)):
                raise ValueError(f'wavelength/depth/depth_err length mismatch for instrument {ins}.')
            if self.mode[ins] not in ['transmission', 'emission']:
                raise ValueError(f"mode for instrument {ins} must be either 'transmission' or 'emission'.")
            if self.resolution[ins] is not None and not isinstance(self.resolution[ins], (str, int, np.integer)):
                raise TypeError(f'resolution for instrument {ins} must be a string or integer.')

        if not isinstance(self.wav_band, dict) or not isinstance(self.res_func, dict):
            raise TypeError('wav_band and res_func must be dict objects when provided.')

        if len(self.wav_band) != 0 or len(self.res_func) != 0:
            if set(self.wav_band.keys()) != set(self.res_func.keys()):
                raise ValueError('wav_band and res_func must have identical instrument keys.')

            # Every spectrum instrument must have a response-function entry.
            missing_response = spec_keys - set(self.wav_band.keys())
            if missing_response:
                missing_list = ', '.join(sorted(missing_response))
                raise ValueError(f'Missing response data for spectrum instruments: {missing_list}.')

            for ins in self.wav_band:
                wb = np.asarray(self.wav_band[ins], dtype=float)
                rf = np.asarray(self.res_func[ins], dtype=float)
                if not (wb.ndim == rf.ndim == 1):
                    raise ValueError(f'Response arrays for instrument {ins} must be 1D.')
                if len(wb) != len(rf):
                    raise ValueError(f'wav_band/res_func length mismatch for instrument {ins}.')
        else:
            raise ValueError('wav_band and res_func are required for all instruments with spectrum data.')

        self._validate_priors()

    def _write_multicolumn_txt(self, filename, rows, header):
        # Write ASCII tables with a single commented header line.
        with open(filename, 'w') as f:
            f.write(header + '\n')
            for row in rows:
                f.write(row + '\n')

    def _read_multicolumn_txt(self, filename, expected_cols):
        # Read whitespace-delimited tables, skipping blank lines and comments.
        rows = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                cols = line.split()
                if len(cols) != expected_cols:
                    raise ValueError(
                        f'File {filename} has malformed line with {len(cols)} columns; expected {expected_cols}.'
                    )
                rows.append(cols)
        return rows

    def save(self):
        # Save priors for every run; normal runs also save spectra and responses.
        if self.pout is None:
            raise ValueError('pout must be provided to save data.')
        if not os.path.exists(self.pout):
            os.makedirs(self.pout)

        priors_path = os.path.join(self.pout, self._priors_file)
        param_width = max(len('parameter'), max((len(p) for p in self.priors.keys()), default=0))
        dist_width = max(len('distribution'), max((len(v['distribution']) for v in self.priors.values()), default=0))

        with open(priors_path, 'w') as f:
            f.write(f'# {"parameter":<{param_width}} {"distribution":<{dist_width}} hyperparameters\n')
            for param, value in self.priors.items():
                distribution = value['distribution']
                hyperparameters = value['hyperparameters']
                f.write(f'{param:<{param_width}} {distribution:<{dist_width}} {repr(hyperparameters)}\n')

        if 'FORWARD' in self.mode.keys():
            return

        spectrum_path = os.path.join(self.pout, self._spectrum_file)
        response_path = os.path.join(self.pout, self._response_file)

        spectrum_rows = []
        for ins in self.wavelength:
            wav = np.asarray(self.wavelength[ins], dtype=float)
            dep = np.asarray(self.depth[ins], dtype=float)
            dep_err = np.asarray(self.depth_err[ins], dtype=float)
            mode = self.mode[ins]
            resolution = self.resolution[ins]
            # Keep the instrument label in the last column and the per-instrument mode in the fifth.
            for i in range(len(wav)):
                spectrum_rows.append(f'{wav[i]:.16e} {dep[i]:.16e} {dep_err[i]:.16e} {ins} {mode} {resolution}')

        self._write_multicolumn_txt(
            spectrum_path,
            spectrum_rows,
            '# wavelength_micron depth_ppm depth_err_ppm instrument mode resolution',
        )

        response_rows = []
        for ins in self.wav_band:
            wb = np.asarray(self.wav_band[ins], dtype=float)
            rf = np.asarray(self.res_func[ins], dtype=float)
            # Response files keep the instrument label in the last column.
            for i in range(len(wb)):
                response_rows.append(f'{wb[i]:.16e} {rf[i]:.16e} {ins}')

        self._write_multicolumn_txt(
            response_path,
            response_rows,
            '# wav_band_micron response_function instrument',
        )

    def read(self):
        # Restore saved priors for every run; spectrum and response files are optional in forward mode.
        if self.pin is None:
            raise ValueError('pin must be provided to read saved data.')
        if not os.path.isdir(self.pin):
            raise ValueError(f'pin directory does not exist: {self.pin}')

        priors_path = os.path.join(self.pin, self._priors_file)
        if not os.path.exists(priors_path):
            raise FileNotFoundError(f'Missing saved priors file: {priors_path}')

        priors = {}
        with open(priors_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(None, 2)
                if len(parts) != 3:
                    raise ValueError(f'Invalid priors line: {line}')
                param, distribution, hyperparameters_text = parts
                try:
                    hyperparameters = ast.literal_eval(hyperparameters_text)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(f'Could not parse hyperparameters for {param}: {hyperparameters_text}') from exc
                priors[param] = {'distribution': distribution, 'hyperparameters': hyperparameters}
        self.priors = priors

        spectrum_path = os.path.join(self.pin, self._spectrum_file)
        response_path = os.path.join(self.pin, self._response_file)

        if not os.path.exists(spectrum_path):
            # Forward runs only need priors, so no spectra or response files are required.
            self.wavelength = {}
            self.depth = {}
            self.depth_err = {}
            self.wav_band = {}
            self.res_func = {}
            self.mode = 'forward'
            self.instruments = []
            return

        if not os.path.exists(response_path):
            raise FileNotFoundError(f'Missing saved response data file: {response_path}')

        spec_rows = self._read_multicolumn_txt(spectrum_path, expected_cols=6)
        wavelength, depth, depth_err, mode, resolution = {}, {}, {}, {}, {}
        for row in spec_rows:
            w, d, e, ins, ins_mode, ins_resolution = row
            if ins not in wavelength:
                wavelength[ins], depth[ins], depth_err[ins] = [], [], []
                mode[ins] = ins_mode
                resolution[ins] = ins_resolution
            elif mode[ins] != ins_mode:
                raise ValueError(f'Inconsistent mode values found for instrument {ins}.')
            elif resolution[ins] != ins_resolution:
                raise ValueError(f'Inconsistent resolution values found for instrument {ins}.')
            wavelength[ins].append(float(w))
            depth[ins].append(float(d))
            depth_err[ins].append(float(e))

        self.wavelength = {k: np.asarray(v, dtype=float) for k, v in wavelength.items()}
        self.depth = {k: np.asarray(v, dtype=float) for k, v in depth.items()}
        self.depth_err = {k: np.asarray(v, dtype=float) for k, v in depth_err.items()}
        self.mode = mode
        self.resolution = {k: self._parse_resolution_value(v) for k, v in resolution.items()}
        self.instruments = sorted(self.wavelength.keys())

        resp_rows = self._read_multicolumn_txt(response_path, expected_cols=3)
        wav_band, res_func = {}, {}
        for row in resp_rows:
            wb, rf, ins = row
            if ins not in wav_band:
                wav_band[ins], res_func[ins] = [], []
            wav_band[ins].append(float(wb))
            res_func[ins].append(float(rf))

        self.wav_band = {k: np.asarray(v, dtype=float) for k, v in wav_band.items()}
        self.res_func = {k: np.asarray(v, dtype=float) for k, v in res_func.items()}

        self._validate_inputs()

    def _parse_resolution_value(self, value):
        # Try to restore integer resolutions, otherwise keep the original string label.
        if value is None or value == 'None':
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    def datadict_preparation(self):
        ## This function prepare a data dictionary which will save the fitting type for each instrument.
        ## This function will also save the list of line and cloud species which will be used in the modelling.
        self.data_dict = {}

        for i in range( len(self.instruments)):
            self.data_dict[ self.instruments[i] ] = {}
            
            ### Listing different names of the fitting and set them False by default.
            ### We will set them True depending on the priors.
            self.data_dict[ self.instruments[i] ]['petitIsoTrans'] = False

            ### Array to save the line species which will be used in the modelling.
            self.data_dict[ self.instruments[i] ]['line_species'] = []
            self.data_dict[ self.instruments[i] ]['logabundance'] = False

            ### Array to save the cloud species which will be used in the modelling.
            self.data_dict[ self.instruments[i] ]['cloud_species'] = []
            self.data_dict[ self.instruments[i] ]['logcloud'] = False

            ### Array to save the rayleigh scattering species which will be used in the modelling.
            self.data_dict[ self.instruments[i] ]['rayleigh_species'] = []

        # Let's see if the user wants to fit either for the planetary mass or the surface gravity.
        self.data_dict['mass_fit'] = False
        for pri in self.priors.keys():
            if pri.lower() == 'mp':
                self.data_dict['mass_fit'] = True
                if self.verbose:
                    print("Planet mass will be a free parameter in the fitting.")

        # Going through the instruments and checking the priors to set the fitting type, line and cloud species.
        for i in range( len(self.instruments) ):

            if type( self.resolution[self.instruments[i]] ) == int:
                
                resolution = '.R' + str( self.resolution[self.instruments[i]] )
                self.data_dict[ self.instruments[i] ]['opacity_mode'] = 'c-k'
            
            else:
                resolution = ''

                if self.resolution[self.instruments[i]] == 'c-k':
                    self.data_dict[ self.instruments[i] ]['opacity_mode'] = 'c-k'
                
                elif self.resolution[self.instruments[i]] == 'lbl':
                    self.data_dict[ self.instruments[i] ]['opacity_mode'] = 'lbl'

            
            for pri in self.priors.keys():
                
                if pri[0:7].lower() == 'isotemp':
                    if ( self.instruments[i] in pri.split('_') ) or ( len(pri.split('_')) == 1 ):
                        if self.code == 'petit':
                            if self.mode[self.instruments[i]] == 'transmission' or self.mode[self.instruments[i]] == 'forward-transmission':
                                self.data_dict[ self.instruments[i] ]['petitIsoTrans'] = True
                                if self.verbose:
                                    print(f"Setting petitRADTRANS isothermal transmission fitting for instrument {self.instruments[i]}.")
                        else:
                            raise NotImplementedError(f"Isothermal fitting is currently only implemented for petitRADTRANS. Unsupported code: {self.code}")
                        
                if pri[0:4].lower() == 'line':
                    ## This means that the prior is for a line species.
                    if ( self.instruments[i] in pri.split('_') ) or ( len(pri.split('_')) == 1 ):
                        if pri.split('-')[1][0:3] == 'log':
                            self.data_dict[ self.instruments[i] ]['line_species'].append( pri.split('-')[1][3:] + resolution )
                            self.data_dict[ self.instruments[i] ]['logabundance'] = True
                            if self.verbose:
                                print(f"Adding line species {pri.split('-')[1][3:]} to instrument {self.instruments[i]} with log-abundance.")
                        else:
                            self.data_dict[ self.instruments[i] ]['line_species'].append( pri.split('-')[1] + resolution )
                            if self.verbose:
                                print(f"Adding line species {pri.split('-')[1]} to instrument {self.instruments[i]}.")

                if pri[0:5].lower() == 'cloud':
                    ## This means that the prior is for a cloud species.
                    if ( self.instruments[i] in pri.split('_') ) or ( len(pri.split('_')) == 1 ):
                        if pri.split('-')[1][0:3] == 'log':
                            self.data_dict[ self.instruments[i] ]['cloud_species'].append( pri.split('-')[1][3:] )
                            self.data_dict[ self.instruments[i] ]['logcloud'] = True
                            if self.verbose:
                                print(f"Adding cloud species {pri.split('-')[1][3:]} to instrument {self.instruments[i]} with log-abundance.")
                        else:
                            self.data_dict[ self.instruments[i] ]['cloud_species'].append( pri.split('-')[1] )
                            if self.verbose:
                                print(f"Adding cloud species {pri.split('-')[1]} to instrument {self.instruments[i]}.")

                if pri[0:8].lower() == 'rayleigh':
                    ## This means that the prior is for a rayleigh scattering species.
                    if ( self.instruments[i] in pri.split('_') ) or ( len(pri.split('_')) == 1 ):
                        self.data_dict[ self.instruments[i] ]['rayleigh_species'].append( pri.split('-')[1] )
                        if self.verbose:
                            print(f"Adding Rayleigh scattering species {pri.split('-')[1]} to instrument {self.instruments[i]}.")

        # Warn (without breaking execution) if setups differ across instruments.
        if len(self.instruments) > 1:
            fit_values = [self.data_dict[ins]['petitIsoTrans'] for ins in self.instruments]
            if len(set(fit_values)) > 1:
                print('Warning: fitting type is different across instruments.')

            line_sets = [tuple(sorted(set(self.data_dict[ins]['line_species']))) for ins in self.instruments]
            if len(set(line_sets)) > 1:
                print('Warning: line species are different across instruments.')

            cloud_sets = [tuple(sorted(set(self.data_dict[ins]['cloud_species']))) for ins in self.instruments]
            if len(set(cloud_sets)) > 1:
                print('Warning: cloud species are different across instruments.')

            rayleigh_sets = [tuple(sorted(set(self.data_dict[ins]['rayleigh_species']))) for ins in self.instruments]
            if len(set(rayleigh_sets)) > 1:
                print('Warning: Rayleigh scattering species (and continuum opacities) are different across instruments.')

    def init_models(self):
        self.models = {}
        for i in range( len( self.instruments ) ):
            if self.code == 'petit':
                self.models[ self.instruments[i] ] = SpectralModel(
                                                                   pressures=np.logspace(self.pressure_range[0], self.pressure_range[1], self.pressure_points),
                                                                   line_species=self.data_dict[ self.instruments[i] ]['line_species'],
                                                                   rayleigh_species=self.data_dict[ self.instruments[i] ]['rayleigh_species'],
                                                                   gas_continuum_contributors=self.cia,
                                                                   wavelength_boundaries=[self.data_dict[ self.instruments[i] ]['wav_min'], self.data_dict[ self.instruments[i] ]['wav_max']],
                                                                   line_opacity_mode=self.data_dict[ self.instruments[i] ]['opacity_mode']
                                                                  )
            else:
                raise NotImplementedError(f"Currently only petitRADTRANS is supported for models. Unsupported code: {self.code}")
    
    def generate_forward_models(self, save=True, **kwargs):
        # This function will generate forward models using model class.
        ## ------ Loading the model class.
        mod = model(self, **kwargs)

        ## Creating the parameter dictionary from the priors.
        post = {}
        for i in self.priors.keys():
            post[i] = self.priors[i]['hyperparameters']

        ## ------- Calculating the forward model for the given parameters.
        mod.generate_forward_models(parameter_values=post, rebin=False)

        ## ------- Extracting the forward model
        forward_wavelength = mod.model_spec['FORWARD']['wavelength']
        forward_spectrum = mod.model_spec['FORWARD']['spectrum']

        ## ------- Saving the forward model if save is True.
        if save:
            forward_path = os.path.join(self.pout, 'forward_model.txt')
            with open(forward_path, 'w') as f:
                f.write('# wavelength_micron spectrum [ppm]\n')
                for w, s in zip(forward_wavelength, forward_spectrum):
                    f.write(f'{w:.16e} {s:.16e}\n')

        return forward_wavelength, forward_spectrum

    def fit(self, **kwargs):
        return fit(self, **kwargs)


class fit(object):
    def __init__(self, data, sampler='dynesty', n_live_points=500, nthreads=None, dynesty_save_states=False, dynesty_resume=False, **kwargs):
        # The following line will inherit the data and priors from the load instance and set up the fitting framework.
        self.data = data

        # Define output results object:
        self.results = None
        self.sampler = sampler
        self.nthreads = nthreads

        self.n_live_points = n_live_points
        
        # Inhert the output folder:
        self.pout = data.pout

        # Define sampler prefix for output files based on the chosen sampler.
        if self.sampler == 'dynesty':
            self.sampler_prefix = '_dynesty_NS_'
        elif self.sampler == 'dynamic_dynesty':
            self.sampler_prefix = '_dynesty_DNS_'
        else:
            self.sampler_prefix = self.sampler + '_'

        self.dynesty_save_states = dynesty_save_states
        self.dynesty_resume = dynesty_resume

        # Generate the posteriors dictionary which will save the current values of the parameters during the fitting. For fixed parameters, we will set the posterior value to the fixed value.
        self.posteriors = {}
        self.model_parameters = list(self.data.priors.keys())
        self.paramnames = []
        nfree = 0
        for pname in self.model_parameters:
            if self.data.priors[pname]['distribution'] == 'fixed':
                self.posteriors[pname] = self.data.priors[pname]['hyperparameters']
            else:
                self.posteriors[pname] = 0.
                self.paramnames.append(pname)
                nfree += 1

        self.transformed_priors = np.zeros(nfree)

        # For each of the variables in the prior that is not fixed, define an internal dictionary that will save the
        # corresponding transformation function to the prior corresponding to that variable. Idea is that with this one
        # simply does self.transform_prior[variable_name](value) and you get the transformed value to the 0,1 prior.
        # This avoids having to keep track of the prior distribution on each of the iterations. This is only useful for
        # nested samplers:
        self.transform_prior = {}
        self.set_prior_transform()

        self.model = model(self.data)

        # First, check if a run has already been performed with the user-defined sampler. If it hasn't, run it.
        # If it has (detected through its output filename), skip running again and jump straight to loading the
        # data:
        out = {}
        runSampler = False
        if self.pout is None:
            self.pout = os.getcwd() + '/'
        if (not os.path.exists(self.pout + self.sampler_prefix +
                               'posteriors.pkl')):
            runSampler = True

        # If runSampler is True, then run the sampler of choice:
        if runSampler:
            if 'dynesty' in self.sampler:
                if self.sampler == 'dynamic_dynesty':
                    DynestySampler = dynesty.DynamicNestedSampler
                elif self.sampler == 'dynesty':
                    DynestySampler = dynesty.NestedSampler

                # To run dynesty, we do it a little bit different depending if we are doing multithreading or not:
                if self.nthreads is None:
                    # As with the other samplers, first extract list of possible args (try-except for back-compatibility with prior dynesty versions):
                    try:
                        args = vars(DynestySampler)['__init__'].__code__.co_varnames
                    except:
                        args = vars(DynestySampler).keys()

                    d_args = {}
                    d_args['bound'] = 'multi'
                    d_args['sample'] = 'rwalk'
                    d_args['nlive'] = self.n_live_points
                    # Match them with kwargs (kwargs take preference):
                    for arg in args:
                        if arg in kwargs:
                            d_args[arg] = kwargs[arg]

                    # Define the sampler:
                    ### NOTE: In case we are resuming a dynesty run, we need to restore from checkpoint file:
                    if not self.dynesty_resume:
                        sampler = DynestySampler(self.loglike,
                                                self.prior_transform_r,
                                                nfree, **d_args)
                    else:
                        sampler = DynestySampler.restore(self.pout + self.sampler_prefix + '_checkpoint.pkl')

                    # Now do the same for the actual sampler:
                    try:
                        args = sampler.run_nested.__func__.__code__.co_varnames
                    except:
                        args = vars(sampler).keys()

                    ds_args = {}
                    # Load ones from kwargs:
                    for arg in args:
                        if arg in kwargs:
                            ds_args[arg] = kwargs[arg]

                    ### NOTE: If the user wants to save and resume dynesty runs, they need to provide one additional keyword to chatak.fit: checkpoint_every 
                    ### (sampler state will be saved every checkpoint_every seconds); we will automatically save the checkpoint file in the output directory location
                    ### See, #127 for more details.
                    if self.dynesty_save_states:
                        ds_args['checkpoint_file'] = self.pout + self.sampler_prefix + '_checkpoint.pkl'
                        ds_args['resume'] = self.dynesty_resume

                    # Now run:
                    sampler.run_nested(**ds_args)

                    # And extract results
                    results = sampler.results

                else:

                    # Before running the whole multithread magic, match kwargs with functional arguments (try-except 
                    # for back-compatibility with prior dynesty versions):
                    try: 
                        args = vars(DynestySampler)['__init__'].__code__.co_varnames
                    except:
                        args = vars(DynestySampler).keys()

                    d_args = {}
                    d_args['bound'] = 'multi'
                    d_args['sample'] = 'rwalk'
                    d_args['nlive'] = self.n_live_points
                    # Match them with kwargs:
                    for arg in args:
                        if arg in kwargs:
                            d_args[arg] = kwargs[arg]

                    # Now define a mock sampler to retrieve variable names:
                    mock_sampler = DynestySampler(self.loglike,
                                                  self.prior_transform_r,
                                                  nfree, **d_args)
                    # Extract args:
                    try:
                        args = mock_sampler.run_nested.__func__.__code__.co_varnames
                    except:
                        args = vars(mock_sampler).keys()

                    ds_args = {}
                    # Load ones from kwargs:
                    for arg in args:
                        if arg in kwargs:
                            ds_args[arg] = kwargs[arg]
                    
                    ### NOTE: If the user wants to save and resume dynesty runs, they need to provide one additional keyword to juliet.fit: checkpoint_every 
                    ### (sampler state will be saved every checkpoint_every seconds); we will automatically save the checkpoint file in the output directory location
                    ### See, #127 for more details.
                    if self.dynesty_save_states:
                        ds_args['checkpoint_file'] = self.pout + self.sampler_prefix + '_checkpoint.pkl'
                        ds_args['resume'] = self.dynesty_resume


                    with multiprocessing.Pool(self.nthreads) as pool:

                        ### NOTE: In case we are resuming a dynesty run, we need to restore from checkpoint file:
                        if not self.dynesty_resume:
                            sampler = DynestySampler(self.loglike,
                                                    self.prior_transform_r,
                                                    nfree,
                                                    pool = pool, 
                                                    queue_size=self.nthreads,
                                                    **d_args)
                        else:
                            sampler = DynestySampler.restore(self.pout + self.sampler_prefix + '_checkpoint.pkl', pool=pool)

                        sampler.run_nested(**ds_args)

                    results = sampler.results 

                # Extract dynesty outputs:
                out['dynesty_output'] = results

                # Get weighted posterior:
                weights = np.exp(results['logwt'] - results['logz'][-1])
                posterior_samples = resample_equal(results.samples, weights)

            elif 'ultranest' in self.sampler:

                # Match kwargs to possible ReactiveNestedSampler keywords. First, extract possible arguments of ReactiveNestedSampler:
                args = ReactiveNestedSampler.__init__.__code__.co_varnames
                rns_args = {}
                # First, define some standard ones:
                rns_args['transform'] = self.prior_transform_r
                rns_args['log_dir'] = self.pout
                rns_args['resume'] = True
                # Now extract arguments from kwargs; they take presedence over the standard ones above:
                for arg in args:
                    if arg in kwargs:
                        rns_args[arg] = kwargs[arg]
                # ...and load the sampler:
                sampler = ReactiveNestedSampler(self.paramnames, self.loglike,
                                                **rns_args)

                if 'slicesampler' in self.sampler:
                    import ultranest.stepsampler

                    # Match kwarfs to possible args in RegionSliceSampler:
                    args = ultranest.stepsampler.SliceSampler.__init__.__code__.co_varnames
                    rss_args = {}
                    # First, define standard ones:
                    rss_args['nsteps'] = 400
                    rss_args['adaptive_nsteps'] = 'move-distance'
                    # Extract kwargs, add them in:
                    for arg in args:
                        if arg in kwargs:
                            rss_args[arg] = kwargs[arg]

                    # Apply stepsampler:
                    sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(
                        **rss_args)

                # Now do the same for ReactiveNestedSampler.run --- load any kwargs the user has given as input:
                args = ReactiveNestedSampler.run.__code__.co_varnames
                rns_run_args = {}
                # Define some standard ones:
                rns_run_args['frac_remain'] = 0.1
                rns_run_args['min_num_live_points'] = self.n_live_points
                rns_run_args['max_num_improvement_loops'] = 1
                # Load the ones from the kwargs:
                for arg in args:
                    if arg in kwargs:
                        rns_run_args[arg] = kwargs[arg]
                # Run the sampler:
                results = sampler.run(**rns_run_args)
                sampler.print_results()
                sampler.plot()

                # Save ultranest outputs:
                out['ultranest_output'] = results
                # Get weighted posterior:
                posterior_samples = results['samples']
                # Get lnZ:
                out['lnZ'] = results['logz']
                out['lnZerr'] = results['logzerr']
            

            # Save posterior samples as outputted by Multinest/Dynesty:
            out['posterior_samples'] = {}
            out['posterior_samples']['unnamed'] = posterior_samples

            # Save log-likelihood of each of the samples:
            out['posterior_samples']['loglike'] = np.zeros(posterior_samples.shape[0])
            for i in tqdm(range(posterior_samples.shape[0])):
                out['posterior_samples']['loglike'][i] = self.loglike(posterior_samples[i, :])

            pcounter = 0
            for pname in self.model_parameters:
                if self.data.priors[pname]['distribution'] != 'fixed':
                    self.posteriors[pname] = np.median(posterior_samples[:, pcounter])
                    out['posterior_samples'][pname] = posterior_samples[:, pcounter]
                    pcounter += 1

            # Get lnZ:
            out['lnZ'] = results.logz[-1]
            out['lnZerr'] = results.logzerr[-1]

            pickle.dump(out, open(self.pout + self.sampler_prefix + 'posteriors.pkl', 'wb'))

        else:
            # If the sampler was already ran, then user really wants to extract outputs from previous fit:
            print('Detected ' + self.sampler + ' sampler output files --- extracting from ' + self.pout + self.sampler_prefix + 'posteriors.pkl')
            out = pickle.load( open( self.pout + self.sampler_prefix + 'posteriors.pkl', 'rb'))

        # Either fit done or extracted. If doesn't exist, create the posteriors.dat file:
        if self.pout is not None:
            if not os.path.exists(self.pout + 'posteriors.dat'):
                outpp = open(self.pout + 'posteriors.dat', 'w')
                writepp(outpp, out, self.data.priors)

        # Save all results (posteriors) to the self.results object:
        self.posteriors = out
        self.model.set_posterior_samples(out['posterior_samples'])

    def loglike(self, cube, ndim=None, nparams=None):
        # Evaluate the joint log-likelihood. For this, first extract all inputs:
        pcounter = 0
        for pname in self.model_parameters:
            if self.data.priors[pname]['distribution'].lower() != 'fixed':
                self.posteriors[pname] = cube[pcounter]
                pcounter += 1
        # Initialize log-likelihood:
        log_likelihood = 0.0

        # Evaluate the model first:
        if self.data.wavelength is not None:
            self.model.generate(self.posteriors, True)
            if self.model.modelOK:
                log_likelihood += self.model.get_loglikelihood()
            else:
                return -1e101
        
        # This is an extra check if the log-likelihood is non-nan
        # (I found that fitting kelp inhomogeneous light curve often produces Nan likelihoods, 
        # so this if statement will prevent that)
        if np.isnan(log_likelihood):
            log_likelihood = -1e101

        # Return total log-likelihood:
        return log_likelihood

    def set_prior_transform(self):
        for pname in self.model_parameters:
            dist_name = self.data.priors[pname]['distribution'].lower()
            if dist_name != 'fixed':
                if dist_name == 'uniform':
                    self.transform_prior[pname] = transform_uniform
                if dist_name == 'normal':
                    self.transform_prior[pname] = transform_normal
                if dist_name == 'truncatednormal':
                    self.transform_prior[pname] = transform_truncated_normal
                if self.data.priors[pname]['distribution'] == 'jeffreys' or self.data.priors[pname]['distribution'] == 'loguniform':
                    self.transform_prior[pname] = transform_loguniform
                if dist_name == 'beta':
                    self.transform_prior[pname] = transform_beta
                if dist_name == 'exponential':
                    self.transform_prior[pname] = transform_exponential
                if dist_name == 'modjeffreys':
                    self.transform_prior[pname] = transform_modifiedjeffreys

    # Prior transform for nested samplers (this one spits the transformed priors from the unit cube):
    def prior_transform_r(self, cube):
        pcounter = 0
        transformed_priors = np.copy(self.transformed_priors)
        for pname in self.model_parameters:
            if self.data.priors[pname]['distribution'].lower() != 'fixed':
                transformed_priors[pcounter] = self.transform_prior[pname](cube[pcounter], \
                                               self.data.priors[pname]['hyperparameters'])
                pcounter += 1
        return transformed_priors

class model(object):
    def __init__(self, data):
        # The following line will inherit the data and priors from the load instance and set up the forward modeling framework.
        self.data = data
        
        ## Redefining several variables for easier access.
        self.instruments = self.data.instruments
        self.data_dict = self.data.data_dict
        self.priors = self.data.priors
        self.models = self.data.models

        self.modelOK = True

        ## Instrumental dependence of various parameters
        self.line_inames = {}
        self.cloud_inames = {}
        self.rayleigh_inames = {}
        self.tp_inames = {}

        # Define a variable that will save the posterior samples:
        self.posteriors = None

        self.model_spec = {}

        for ins in self.instruments:
            ## Loading the 
            self.model_spec[ins] = {}

            if self.data.mode[ins][0:7] != 'forward':
                self.model_spec[ins]['wavelength'] = np.zeros_like( self.data.depth[ins] )
                self.model_spec[ins]['spectrum'] = np.zeros_like( self.data.depth[ins] )
                self.model_spec[ins]['variances'] = np.zeros_like( self.data.depth_err[ins] )
            else:
                self.model_spec[ins]['wavelength'] = np.zeros( 1000 )  # Placeholder wavelength array for forward runs; actual wavelengths will depend on the forward model setup.
                self.model_spec[ins]['spectrum'] = np.zeros( 1000 )  # Placeholder spectrum array for forward runs; actual values will be computed by the forward model.
                self.model_spec[ins]['variances'] = np.zeros( 1000 )  # Placeholder variances array for forward runs; actual values will be computed by the forward model.

            for pri in self.priors.keys():
                if pri[0:4].lower() == 'line':
                    vec = pri.split('_')
                    if len(vec) > 1:
                        ## This means that the prior is instrument-dependent.
                        if ins in vec:
                            self.line_inames[ins] = '_' + '_'.join(vec[1:])
                    else:
                        ## This means that the prior is global (not instrument-dependent).
                        self.line_inames[ins] = ''
                
                if pri[0:5].lower() == 'cloud':
                    vec = pri.split('_')
                    if len(vec) > 1:
                        ## This means that the prior is instrument-dependent.
                        if ins in vec:
                            self.cloud_inames[ins] = '_' + '_'.join(vec[1:])
                    else:
                        ## This means that the prior is global (not instrument-dependent).
                        self.cloud_inames[ins] = ''

                if pri[0:8].lower() == 'rayleigh':
                    vec = pri.split('_')
                    if len(vec) > 1:
                        ## This means that the prior is instrument-dependent.
                        if ins in vec:
                            self.rayleigh_inames[ins] = '_' + '_'.join(vec[1:])
                    else:
                        ## This means that the prior is global (not instrument-dependent).
                        self.rayleigh_inames[ins] = ''

                if pri[0:7].lower() == 'isotemp':
                    vec = pri.split('_')
                    if len(vec) > 1:
                        ## This means that the prior is instrument-dependent.
                        if ins in vec:
                            self.tp_inames[ins] = '_' + '_'.join(vec[1:])
                    else:
                        ## This means that the prior is global (not instrument-dependent).
                        self.tp_inames[ins] = ''
        
        # Set the model-type to M(t):
        self.evaluate = self.evaluate_model
        self.generate = self.generate_forward_models

    def set_posterior_samples(self, posterior_samples):
        self.posteriors = posterior_samples
        self.median_posterior_samples = {}
        
        for parameter in self.posteriors.keys():
            if parameter != 'unnamed':
                self.median_posterior_samples[parameter] = np.median(self.posteriors[parameter])

        for parameter in self.priors:
            if self.priors[parameter]['distribution'] == 'fixed':
                self.median_posterior_samples[parameter] = self.priors[parameter]['hyperparameters']
        
        try:
            self.generate(self.median_posterior_samples, True)
        except:
            print(
                'Warning: model evaluated at the posterior median did not compute properly.'
            )


    def evaluate_model(self, instrument = None, parameter_values = None, all_samples = False, nsamples = 1000, return_samples = False, 
                       resolution = None, return_err = False, alpha = 0.68):
        # This function will generate the model spectrum for the given instrument and parameter values, and return the model spectrum. 
        # If all_samples is True, then it will use all samples to generate the model spectrum (parameter_values should be a dictionary with parameter names as keys and their corresponding samples as values). 
        # If return_samples is True, then it will also return models created for each samples
        # If resolution is not None, then it will rebin the model spectrum to the given resolution. 
        # If return_err is True, then it will also return the error on the model spectrum (which can be calculated using a simple Monte Carlo approach by generating multiple spectra with parameters). 
        # If alpha is provided, then it will return the credible interval corresponding to that alpha value.
        
        ## ------------------ 
        ##.  First if statement: if parameter_values are given then we will generate the model spectrum for those parameter values.
        ##.        else statement: if parameter_values are not given, the model spectrum is generated for the posteriors
        if parameter_values is not None:
            ## Generate the model spectrum for the given parameter values.
            #### ---- parameter_values can be np.arrays (i.e., samples) or single values. 
            parameters = list(self.priors.keys())
            input_parameters = list(parameter_values.keys())
            if type(parameter_values[input_parameters[0]]) is np.ndarray:
                ## This means that the parameter values are samples. We will generate the model spectrum for each sample.
                ### The user can either use all_samples (i.e., all_samples=True), or we can provide nsamples
                nsampled = len(parameter_values[input_parameters[0]])
                if all_samples:
                    nsamples = nsampled
                    idx_samples = np.arange(nsamples)
                else:
                    idx_samples = np.random.choice(np.arange(nsampled),
                                                   np.min([nsamples, nsampled]),
                                                   replace=False)
                    idx_samples = idx_samples[np.argsort(idx_samples)]

                # Create dictionary that saves the current parameter_values to evaluate:
                current_parameter_values = dict.fromkeys(parameters)
                ### Adding the parameters which were fixed in the analysis.
                for parameter in parameters:
                    if self.priors[parameter]['distribution'] == 'fixed':
                        current_parameter_values[parameter] = self.priors[parameter]['hyperparameters']

                if resolution is None:
                    ## If resolution is None, i.e., we will use the resolution of the data to generate models
                    output_model_samples = np.zeros([
                            nsamples,
                            len( self.data.depth[instrument] )
                        ])
                else:
                    ## If resolution is provided, then we generate models at those resolutions
                    ### Currently, this means that the default resolution at which the model is created.
                    ### We don't know the length of wavelength array in that case, 
                    ### so, let's run generate_forward_models function to generate models at one set of values 
                    ### and then get the length of array from that.
                    for parameter in input_parameters:
                        # Populate the current parameter_values
                        current_parameter_values[parameter] = parameter_values[parameter][idx_samples[0]]
                    self.generate_forward_models(current_parameter_values, rebin=False)
                    output_model_samples = np.zeros([
                            nsamples,
                            len( self.model_spec[instrument]['spectrum'] )
                        ])
                    
                    ## Okay if the model resolution is not the same as the data resolution, then we also need to provide wavelengths
                    output_model_wavelengths = np.zeros( len( self.model_spec[instrument]['spectrum'] ) )
                
                # Now iterate through all samples:
                for i in tqdm(range(len(idx_samples))):
                    # Get parameters for the i-th sample:
                    for parameter in input_parameters:
                        # Populate the current parameter_values
                        current_parameter_values[parameter] = parameter_values[parameter][idx_samples[i]]
                    
                    if resolution is None:
                        self.generate_forward_models(current_parameter_values, rebin=True)
                    else:
                        self.generate_forward_models(current_parameter_values, rebin=False)
                        output_model_wavelengths = self.model_spec[instrument]['wavelength']
                    
                    output_model_samples[i,:] = self.model_spec[instrument]['spectrum']
                
                # If return_error is on, return upper and lower sigma (alpha x 100% CI) of the model(s):
                if return_err:
                    m_output_model, u_output_model, l_output_model = np.zeros(output_model_samples.shape[1]),\
                                                                     np.zeros(output_model_samples.shape[1]),\
                                                                     np.zeros(output_model_samples.shape[1])
                    
                    ## Generating quantiles
                    for i in range(output_model_samples.shape[1]):
                            m_output_model[i], u_output_model[i], l_output_model[i] = get_quantiles(output_model_samples[:, i], alpha=alpha)
                else:
                    output_model = np.nanmedian(output_model_samples, axis=0)
                    
            else:
                ## This means that the parameter values are single values. We will generate the model spectrum for those values.
                if resolution is None:
                    self.generate_forward_models(parameter_values, rebin=True)
                else:
                    self.generate_forward_models(parameter_values, rebin=False)
                    output_model_wavelengths = self.model_spec[instrument]['wavelength']

                output_model = self.model_spec[instrument]['spectrum']

        else:
            ## We will use the posteriors to generate the model spectrum.
            x = self.evaluate_model(instrument=instrument, parameter_values=self.posteriors, all_samples=all_samples, nsamples=nsamples, return_samples=return_samples, resolution=resolution, return_err=return_err, alpha=alpha)
            if return_samples:
                if return_err:
                    if resolution is None:
                        output_model_samples, m_output_model, u_output_model, l_output_model = x
                    else:
                        output_model_samples, m_output_model, u_output_model, l_output_model, output_model_wavelengths = x
                else:
                    if resolution is None:
                        output_model_samples, output_model = x
                    else:
                        output_model_samples, output_model, output_model_wavelengths = x
            else:
                if return_err:
                    if resolution is None:
                        m_output_model, u_output_model, l_output_model = x
                    else:
                        m_output_model, u_output_model, l_output_model, output_model_wavelengths = x
                else:
                    if resolution is None:
                        output_model = x
                    else:
                        output_model, output_model_wavelengths = x
        
        if return_samples:
            if return_err:
                if resolution is None:
                    return output_model_samples, m_output_model, u_output_model, l_output_model
                else:
                    return output_model_samples, m_output_model, u_output_model, l_output_model, output_model_wavelengths
            else:
                if resolution is None:
                    return output_model_samples, output_model
                else:
                    return output_model_samples, output_model, output_model_wavelengths
        else:
            if return_err:
                if resolution is None:
                    return m_output_model, u_output_model, l_output_model
                else:
                    return m_output_model, u_output_model, l_output_model, output_model_wavelengths
            else:
                if resolution is None:
                    return output_model
                else:
                    return output_model, output_model_wavelengths

    def generate_forward_models(self, parameter_values, rebin):
        ## Okay now we want to generate the forward model for each instrument using the parameter values.
        ## The parameter values will be a dictionary with parameter names as keys and their corresponding values as values.
        ## We will loop through the instruments and generate the model spectrum for each instrument using the SpectralModel class from petitRADTRANS (or the appropriate class for the chosen forward model code).
        
        for ins in self.instruments:
            if self.data.code == 'petit':
                # Alright, we will generate the forward model using petitRADTRANS for this instrument.
                # We also don't know if the input parameters are in log space or not.

                ## List of line species for this instrument.
                self.models[ins].model_parameters['imposed_mass_fractions'] = {}
                for ls in self.data_dict[ins]['line_species']:

                    ## If the resolution is int, then the name of the species would be something like 'CO2.R200'. 
                    ## Let's leave resolution out of the name of the species.
                    if type( self.data.resolution[ins] ) == int:
                        ls1 = ls.split('.')[0]

                    if self.data_dict[ins]['logabundance']:
                        abundance = 10 ** parameter_values[ 'line-log' + ls1 + self.line_inames[ins] ]
                    else:
                        abundance = parameter_values[ 'line-' + ls1 + self.line_inames[ins] ]
                    
                    self.models[ins].model_parameters['imposed_mass_fractions'][ls] = abundance
                
                ## List of filling species
                self.models[ins].model_parameters['filling_species'] = {}
                for rs in self.data_dict[ins]['rayleigh_species']:
                    self.models[ins].model_parameters['filling_species'][rs] = parameter_values[ 'rayleigh-' + rs + self.rayleigh_inames[ins] ]

                ## Setting up other planetary parameters
                ### Calculating and extracting the stellar radius in cm
                rst = parameter_values['rst']
                rst_cm = ( rst * u.R_sun ).to(u.cm).value

                rp = ( parameter_values['rprs'] * rst * u.R_sun ).to(u.cm).value
                self.models[ins].model_parameters['planet_radius'] = rp

                ### Either mass or surface gravity must be provided.
                if self.data_dict['mass_fit']:
                    mp = parameter_values['mp'] * u.M_earth
                    #self.models[ins].model_parameters['planet_mass'] = mp.to(u.g).value
                    # Converting the mass to surface gravity using the radius:
                    g = ( con.G * mp / ( rp * u.cm )**2 ).to(u.cm / u.s**2).value
                    self.models[ins].model_parameters['reference_gravity'] = g
                else:
                    self.models[ins].model_parameters['reference_gravity'] = parameter_values['surfgrav']

                ## Reference pressure
                self.models[ins].model_parameters['reference_pressure'] = parameter_values['refP']

                ## Setting the parameters for temperature profile
                if self.data_dict[ins]['petitIsoTrans']:
                    self.models[ins].model_parameters['temperature_profile_mode'] = 'isothermal'
                    self.models[ins].model_parameters['temperature'] = parameter_values['isotemp' + self.tp_inames[ins] ]

                    if rebin:
                        self.models[ins].model_parameters['rebinned_wavelengths'] = ( self.data.wavelength[ins] * u.micron ).to(u.cm).value

                    forward_wav_model, forward_spec_model = self.models[ins].calculate_spectrum(mode='transmission', rebin=rebin)

                    forward_wav_model = ( forward_wav_model[0,:] * u.cm ).to(u.micron).value
                    forward_spec_model = ( ( forward_spec_model[0,:] / rst_cm )**2 ) * 1e6

                # We don't need offset and sigma_w for forward models
                if 'FORWARD' in self.data.mode.keys():
                    parameter_values['offset_' + ins ] = 0.0
                    parameter_values['sigma_w_' + ins ] = 0.0

                self.model_spec[ins]['wavelength'] = forward_wav_model
                self.model_spec[ins]['spectrum'] = forward_spec_model + parameter_values['offset_' + ins ]

                if 'FORWARD' not in self.data.mode.keys():
                    self.model_spec[ins]['variances'] = self.data.depth_err[ins] ** 2 + parameter_values['sigma_w_' + ins ] ** 2

            else:
                raise NotImplementedError(f"Currently only petitRADTRANS is supported for models. Unsupported code: {self.data.code}")
            

    def gaussian_log_likelihood(self, residuals, variances):
        taus = 1. / variances
        return -0.5 * (len(residuals) * log2pi + np.sum(-np.log(taus.astype(float)) + taus * (residuals**2)))
    

    def get_loglikelihood(self):
        # This function will calculate the log-likelihood.
        log_like = 0.0
        for ins in self.instruments:
            resids = self.data.depth[ins] - self.model_spec[ins]['spectrum']
            log_like += self.gaussian_log_likelihood(resids, self.model_spec[ins]['variances'])
        return log_like