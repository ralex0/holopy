"""
.. moduleauthor:: Ron Alexander <ralexander@g.harvard.edu>
"""
import time

import numpy as np

from holopy.core.holopy_object import HoloPyObject
from holopy.inference.model import AlphaModel
from holopy.inference.result import FitResult, InferenceResult, UncertainValue
from holopy.scattering import calc_holo
from holopy.scattering.scatterer import Sphere
from holopy.scattering.errors import InvalidScatterer

from nevergrad import instrumentation as instru
from nevergrad.optimization import optimizerlib

class NevergradStrategy(HoloPyObject):
    def __init__(self, optimizer='CMA', budget=100):
        self.optimizer = optimizer
        self.budget = budget

    def optimize(self, model, data):
        time_start = time.time()
        cost = self.cost_function(model, data)
        dim = self._get_model_diminsion(model)
        optimizer = self._setup_optimizer(dim)
        reccomendation = optimizer.optimize(cost, executor=None, batch_mode=True)
        parameters = {key: value for key, value in zip(model.parameters.keys(), reccomendation)}
        perrors = self._estimate_error_from_fit()
        intervals = [UncertainValue(value, perrors, name=key)
                     for key, value in parameters.items()]

        d_time = time.time() - time_start
        best_scatterer, best_scaling = self._scatterer_from_optimizer_params(reccomendation)
        result = FitResult(data=data, model=model, strategy=optimizer, 
                           intervals=intervals, time=d_time, kwargs={})
        return result

    def cost_function(self, model, data):
        if hasattr(model, 'theory'):
            def cost(*x):
                params, sixth = self._scatterer_from_optimizer_params(x)
                optics, scatterer = model._optics_scatterer(params, data)
                return self._cost_function(data, scatterer, alpha=sixth)
                #return np.min((-1 * AlphaModel(scatterer=scatterer, alpha=sixth, **optics).lnlike(params, data), 1e60))

            x = instru.variables.Gaussian(mean=model.scatterer.center[0].guess, std=model.scatterer.center[0].interval/10)
            y = instru.variables.Gaussian(mean=model.scatterer.center[1].guess, std=model.scatterer.center[1].interval/10)
            z = instru.variables.Gaussian(mean=model.scatterer.center[2].guess, std=model.scatterer.center[2].interval/10)
            n = instru.variables.Gaussian(mean=model.scatterer.n.guess, std=model.scatterer.n.interval/10)
            r = instru.variables.Gaussian(mean=model.scatterer.r.guess, std=model.scatterer.r.interval/10)
            alpha = instru.variables.Gaussian(mean=model.alpha.guess, std=.3)

            icost = instru.InstrumentedFunction(cost, x, y, z, n, r, alpha)
            return icost
        else:
            return self._make_cost_function(model, data)

    def _cost_function(self, data, scatterer, **kwargs):
        return self.calc_err_sq(data, scatterer, **kwargs)

    def calc_err_sq(self, data, scatterer, **kwargs):
        residual = self.calc_residual(data, scatterer, **kwargs)
        return np.sum(residual ** 2)
    
    def calc_residual(self, data, scatterer, **kwargs):
        dt = data.values.squeeze()
        scaling = kwargs['alpha']
        fit = calc_holo(data, scatterer, scaling=scaling).values.squeeze()
        return fit - dt

    def _make_cost_function(self, model, data):
        return lambda x: -model.lnlike(x, data)

    def _get_model_diminsion(self, model):
        return len(model.parameters)

    def _setup_optimizer(self, dimension):
        return optimizerlib.registry[self.optimizer](dimension=dimension, 
                                                     budget=self.budget)

    def _estimate_error_from_fit(self):
        return 0

    def _scatterer_from_optimizer_params(self, params):
        try: 
            x, y, z, index, radius, sixth = params
        except:
            return 1.0, 1.0
        scat_params = {'center.0': x,
                       'center.1': y,
                       'center.2': z,
                       'r': np.min([np.max([1e-16, radius]), 5]),
                       'n': np.min([np.max([1.00, index]), 2.5])}
        return scat_params, np.min([np.max([0, sixth]), 1])
            
