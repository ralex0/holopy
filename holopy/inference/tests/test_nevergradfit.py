from copy import copy

import unittest

import numpy as np

from holopy.inference import NevergradStrategy
from holopy.inference.prior import Prior, Uniform
from holopy.scattering.scatterer import _expand_parameters

from nevergrad.optimization import optimizerlib

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