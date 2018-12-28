# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.
"""
.. moduleauthor:: Ron Alexander <ralexander@g.harvard.edu>
"""
import time

from concurrent import futures

import numpy as np

from holopy.core.holopy_object import HoloPyObject
from holopy.inference.model import AlphaModel, PerfectLensModel
from holopy.inference.result import FitResult, UncertainValue
from holopy.scattering import calc_holo
from holopy.scattering.scatterer import Sphere

from nevergrad import instrumentation as instru
from nevergrad.optimization import optimizerlib

class GradientFreeStrategy(HoloPyObject):
    def __init__(self, optimizer='OnePlusOne', budget=128, num_workers=8):
        self.optimizer = optimizer
        self.budget = budget
        self.num_workers = num_workers

    def optimize(self, model, data):
        time_start = time.time()
        cost = self.cost_function(model, data)
        dim = self._get_model_diminsion(model)
        optimizer = self._setup_optimizer(dim)
        reccomendation = self._minimize(cost, optimizer)
        if isinstance(model, AlphaModel):
            # This branch is if the model is Mie w/ alpha
            # TODO: Refactor like elif branch after model.parameters is OrderedDict
            param_names = ['center.0', 'center.1', 'center.2', 'n', 'r', 'alpha']
            parameters = {key: value for key, value in zip(param_names, reccomendation)}
        elif isinstance(model, PerfectLensModel):
            # This branch is if the model is Mie + lens
            # TODO: Refactor like elif branch after model.parameters is OrderedDict
            param_names = ['center.0', 'center.1', 'center.2', 'n', 'r', 'lens_angle']
            parameters = {key: value for key, value in zip(param_names, reccomendation)}
        else:
            # This branch is if the model is anything else... such as _SimpleModel
            parameters = {key: value for key, value in zip(model.parameters.keys(), reccomendation)}
        perrors = self._estimate_error_from_fit(optimizer)
        # TODO: Refactor this into the error estimate function
        if hasattr(cost, 'convert_to_arguments'):
            perrors, _ = cost.convert_to_arguments(perrors)
        intervals = self._make_intervals_from_parameters(parameters, perrors)
        d_time = time.time() - time_start
        best_scatterer, best_scaling = self._scatterer_from_optimizer_params(reccomendation)
        result = FitResult(data=data, model=model, strategy=optimizer, 
                           intervals=intervals, time=d_time, kwargs={})
        return result

    def cost_function(self, model, data):
        if isinstance(model, AlphaModel):
            def cost(*x):
                sph_params, alpha = self._scatterer_from_optimizer_params(x)
                optics, scatterer = model._optics_scatterer(sph_params, data)
                return self._cost_function(data, scatterer, alpha=alpha)
            x, y, z, n, r, alpha = self._make_instrumented_variables(model)
            return instru.InstrumentedFunction(cost, x, y, z, n, r, alpha)
        else:
            return lambda x: -model.lnlike(x, data)

    def _scatterer_from_optimizer_params(self, params):
        # TODO: Currently, parameter bounds are imposed by this function. Would 
        #       be better if nevergrad had bounded parameter types.
        try: 
            x, y, z, index, radius, sixth = params
        except:
            return 1.0, 1.0
        scat_params = {'center.0': x,
                       'center.1': y,
                       'center.2': z,
                       'r': np.min([np.max([1e-16, radius]), 5]),
                       'n': np.min([np.max([1.00, index]), 2.5])}
        return scat_params, np.min([np.max([0, sixth]), 1.1])

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

    def _make_instrumented_variables(self, model):
        # TODO: Shouldn't we be able to just use the dimension parameter of 
        #       Gaussian to made a 6D parameter?
        x = instru.variables.Gaussian(mean=model.scatterer.center[0].guess, 
                                      std=model.scatterer.center[0].interval/4)
        y = instru.variables.Gaussian(mean=model.scatterer.center[1].guess, 
                                      std=model.scatterer.center[1].interval/4)
        z = instru.variables.Gaussian(mean=model.scatterer.center[2].guess, 
                                      std=model.scatterer.center[2].interval/4)
        n = instru.variables.Gaussian(mean=model.scatterer.n.guess, 
                                      std=model.scatterer.n.interval/4)
        r = instru.variables.Gaussian(mean=model.scatterer.r.guess, 
                                      std=model.scatterer.r.interval/4)
        alpha = instru.variables.Gaussian(mean=model.alpha.guess, 
                                          std=model.alpha.interval/4)
        return x, y, z, n, r, alpha

    def _get_model_diminsion(self, model):
        return len(model.parameters)

    def _setup_optimizer(self, dimension):
        return optimizerlib.registry[self.optimizer](dimension=dimension, 
                                                     budget=self.budget,
                                                     num_workers=self.num_workers)

    def _minimize(self, cost, optimizer):
        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            reccomendation = optimizer.optimize(cost, executor=executor, batch_mode=True)
        if hasattr(cost, 'convert_to_arguments'):
            reccomendation, _ = cost.convert_to_arguments(reccomendation)
        return reccomendation

    def _estimate_error_from_fit(self, optimizer):
        # FIXME: Need to implement error estimation for each optimizer strategy
        if self.optimizer == "CMA":
            return optimizer.es.result.stds
        elif self.optimizer == "OnePlusOne":
            # FIXME: Is this the correct error estimate for OnePlusOne?
            return optimizer.sigma * np.ones(optimizer.dimension)
        return np.zeros(optimizer.dimension)

    def _make_intervals_from_parameters(self, parameters, par_errors):
        return [UncertainValue(value, error, name=key)
                for (key, value), error in zip(parameters.items(), par_errors)]


            
