from ..core.metadata import get_extents, get_spacing
from .prior import Uniform

from lmfit import Minimizer, Parameters, Parameter

import numpy as np

from scipy.ndimage.filters import gaussian_filter


OPTIMIZER_METHODS = ['leastsq', 'least_squares', 'nelder', 'lbfgsb', 'powell',
                     'cg', 'newton', 'cobyla',  'bfgsb', 'tnc', 'trust-ncg',
                     'trust-exact', 'trust-krylov', 'trust-constr', 'dogleg',
                     'slsqp', 'differential_evolution', 'brute', 'basinhopping',
                     'ampgo', 'emcee']

class Optimizer(object):
    """Does fitting and inference on data using holopy models using lmfit
    """
    def __init__(self, model, method='leastsq'):
        """
        Args:
            model : Scattering model defined in holopy.inference
            method (str) : method of minimization provided by lmfit
                'leastsq' or 'least_squares': Levenberg-Marquardt 
                'nelder': Nelder-Mead
                'lbfgsb': L-BFGS-B
                'powell': Powell  
                'cg': Conjugate Gradient
                'newton': Newton-CG
                'cobyla': COBYLA  
                'bfgsb': BFGS
                'tnc': Truncated Newton
                'trust-ncg': Newton CG trust-region
                'trust-exact': Exact trust-region (SciPy >= 1.0)
                'trust-krylov': Newton GLTR trust-region (SciPy >= 1.0)
                'trust-constr': Constrained trust-region (SciPy >= 1.1)
                'dogleg': Dogleg
                'slsqp': Sequential Linear Squares Programming
                'differential_evolution': Differential Evolution
                'brute': Brute force method
                'basinhopping': Basinhopping
                'ampgo': Adaptive Memory Programming for Global Optimization
                'emcee': Maximum likelihood via Monte-Carlo Markov Chain
        """
        self.model = model
        self.method = method

    def fit(self, data, initial_guess):
        params = self._setup_params_from(initial_guess, data)
        cost_kwargs = {'data': data, 'noise': estimate_noise_from(data)}
        minimizer = self._setup_minimizer(params, cost_kwargs=cost_kwargs)
        fit_result = minimizer.minimize(params=params, method=self.method)
        return fit_result

    def _setup_params_from(self, initial_guess, data):
        params = Parameters()
        x, y, z = self._make_position_params(data, initial_guess)
        n = self._make_index_param(data, initial_guess)
        r = self._make_radius_param(data, initial_guess)
        params.add_many(x, y, z, n, r)
        if 'alpha' in self.model.parameters:
            alpha_val = self._alpha_guess(initial_guess)
            params.add(name = 'alpha', value=alpha_val, min=0.05, max=1.0)
        elif 'lens_angle' in self.model.parameters:
            angle_val = self._lens_guess(initial_guess)
            params.add(name = 'lens_angle', value=angle_val, min=0.05, max=1.1)
        else:
            raise KeyError
        return params

    def _make_position_params(self, data, guess):
        image_x_values = data.x.values
        image_min_x = image_x_values.min()
        image_max_x = image_x_values.max()

        image_y_values = data.y.values
        image_min_y = image_y_values.min()
        image_max_y = image_y_values.max()

        if ('x' in guess) and ('y' in guess):
            x_guess = guess['x'] 
            y_guess = guess['y']
        elif ('center.0' in guess) and ('center.1' in guess):
            x_guess = guess['center.0'] 
            y_guess = guess['center.1']
        else:
            pixel_spacing = get_spacing(data)
            image_lower_left = np.array([image_min_x, image_min_y])
            x_guess, y_guess = center_find(data) * pixel_spacing + image_lower_left

        extents = get_extents(data)
        zextent = 5 * max(extents['x'], extents['y']) # FIXME: 5 is a magic number.
        z_guess = guess['z'] if 'z' in guess else guess['center.2']

        x = Parameter(name='x', value=x_guess, min=image_min_x, max=image_max_x)
        y = Parameter(name='y', value=y_guess, min=image_min_y, max=image_max_y)
        z = Parameter(name='z', value=z_guess, min=-zextent, max=zextent)
        return x, y, z

    def _make_index_param(self, data, guess, bounds=None):
        """ Make index parameter
        # TODO: Use bounds paramter
        """
        n = guess['n']
        n_min, n_max = bounds if not (bounds is None) else data.medium_index*1.001, 2.5
        return Parameter(name = 'n', value=n, min=n_min, max=n_max)

    def _make_radius_param(self, data, guess, bounds=None):
        """ Make radius parameter
        """
        r = guess['r']
        r_min, r_max = bounds if not (bounds is None) else r / 10, r * 10
        return Parameter(name = 'r', value=r, min=r_min, max=r_max)

    def _lens_guess(self, guess):
        return guess['lens_angle'] if 'lens_angle' in guess else np.arcsin(1.2/2)

    def _alpha_guess(self, guess):
        return guess['alpha'] if 'alpha' in guess else 0.8

    def _setup_minimizer(self, params, cost_kwargs=None):
        cost_function = self._setup_cost_function()
        return Minimizer(cost_function, params, nan_policy='omit', fcn_kws=cost_kwargs)

    def _setup_cost_function(self):
        return self._calc_residuals

    def _calc_residuals(self, params, *, data=None, noise=None):
        if noise is None:
            noise = estimate_noise_from(data)
        priors = _holopy_priors_from(params)
        return self.model._residuals(priors, data, noise)

def estimate_noise_from(data):
    data = data.values.squeeze()
    smoothed_data = gaussian_filter(data, sigma=1)
    noise = np.std(data - smoothed_data)
    return noise

def _holopy_priors_from(params):
    priors = {p.name: Uniform(lower_bound=p.min, upper_bound=p.max, guess=p.value)
              for p in params.values()}
    priors['center.0'] = priors.pop('x')
    priors['center.1'] = priors.pop('y')
    priors['center.2'] = priors.pop('z')
    return priors
