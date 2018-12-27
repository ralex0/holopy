from copy import copy

import unittest

import numpy as np


from holopy.core.process import normalize
from holopy.core.tests.common import get_example_data
from holopy.inference.model import AlphaModel
from holopy.inference.prior import Prior, Uniform
from holopy.scattering import Sphere, Mie
from holopy.scattering.scatterer import _expand_parameters

import sys
sys.path.append('..')
from nevergradfit import NevergradStrategy

from nevergrad.optimization import optimizerlib

gold_alpha = .6497

gold_sphere = Sphere(1.582+1e-4j, 6.484e-7,
                     (5.534e-6, 5.792e-6, 1.415e-5))

class TestNevergradStrategy(unittest.TestCase):
    def test_nevergrad_OnePlusOne(self):
        cost = lambda x: _simple_cost_function(x, x_obs=1.0)
        optimizer = optimizerlib.registry['OnePlusOne'](dimension=1, budget=200)
        result = optimizer.optimize(cost, executor=None, batch_mode=True)
        result_ok = np.allclose(result, 1.0, rtol=.001)
        self.assertTrue(result_ok)

    def test_nevergrad_CMA(self):
        cost = lambda x: _simple_cost_function(x, x_obs=np.array([1.0, 1.0]))
        optimizer = optimizerlib.registry['CMA'](dimension=2, budget=200)
        result = optimizer.optimize(cost, executor=None, batch_mode=True)
        result_ok = np.allclose(result, [1.0, 1.0], rtol=.001)
        self.assertTrue(result_ok)

    def test_NevergradStrategy(self):
        data = 0.5
        model = _SimpleModel(x = Uniform(0, 1))
        strat = NevergradStrategy(optimizer='OnePlusOne', budget=200)
        result = strat.optimize(model, data)
        result_ok = np.allclose(result.parameters['x'], .5, rtol=.001)
        self.assertTrue(result_ok)

    def test_fit_mie_par_scatterer(self):
        holo = normalize(get_example_data('image0001'))
        center_guess = [
            Uniform(0, 1e-5, name='x', guess=.567e-5),
            Uniform(0, 1e-5, name='y', guess=.576e-5),
            Uniform(1e-5, 2e-5, name='z', guess=15e-6),
            ]
        scatterer = Sphere(
            n=Uniform(1, 2, name='n', guess=1.59),
            r=Uniform(1e-8, 1e-5, name='r', guess=8.5e-7),
            center=center_guess)
        alpha = Uniform(0.1, 1, name='alpha', guess=0.6)

        theory = Mie(compute_escat_radial=False)
        model = AlphaModel(scatterer, theory=theory, alpha=alpha)

        fitter = NevergradStrategy(optimizer='CMA', budget=200)
        result = fitter.optimize(model, holo)
        fitted = result.scatterer
        print(result.parameters)
        self.assertTrue(np.isclose(fitted.n, gold_sphere.n, rtol=1e-3))
        self.assertTrue(np.isclose(fitted.r, gold_sphere.r, rtol=1e-3))
        self.assertTrue(
            np.allclose(fitted.center, gold_sphere.center, rtol=1e-3))
        self.assertTrue(
            np.isclose(result.parameters['alpha'], gold_alpha, rtol=0.1))
        self.assertEqual(model, result.model)


def _simple_cost_function(x, x_obs=None):
    if x_obs is None:
        x_obs = np.ones(len(np.atleast_1d(x)))
    return np.sum((x-x_obs)**2)

class _SimpleModel:
    def __init__(self, x):
        self._parameters = []
        self._use_parameters({'x': x})

    def cost(self, x, x_obs=None):
        return _simple_cost_function(x, x_obs)

    def lnlike(self, pars, data):
        return -self.cost(pars, data)

    def _use_parameters(self, parameters, as_attr=True):
        if as_attr:
            for name, par in parameters.items():
                if par is not None:
                    setattr(self, name, par)
        parameters = dict(_expand_parameters(parameters.items()))
        for key, val in parameters.items():
            if isinstance(val, Prior):
                self._parameters.append(copy(val))
                self._parameters[-1].name = key

    @property
    def parameters(self):
        return {par.name:par for par in self._parameters}

if __name__ == '__main__':
    unittest.main()