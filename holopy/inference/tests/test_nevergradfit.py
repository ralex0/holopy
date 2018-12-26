import unittest

import numpy as np

from nevergrad.optimization import optimizerlib

class TestNevergradStrategy(unittest.TestCase):
    def test_nevergrad_OnePlusOne(self):
        cost = lambda x: _simple_cost_function(x, x0=1.0)
        optimizer = optimizerlib.registry['OnePlusOne'](dimension=1, budget=200)
        result = optimizer.optimize(cost, executor=None, batch_mode=True)
        result_ok = np.allclose(result, 1.0, rtol=.001)
        self.assertTrue(result_ok)

    def test_nevergrad_CMA(self):
        cost = lambda x: _simple_cost_function(x, x0=np.array([1.0, 1.0]))
        optimizer = optimizerlib.registry['CMA'](dimension=2, budget=200)
        result = optimizer.optimize(cost, executor=None, batch_mode=True)
        result_ok = np.allclose(result, [1.0, 1.0], rtol=.001)
        self.assertTrue(result_ok)


def _simple_cost_function(x, x0=None):
    if x0 is None:
        x0 = np.ones(len(np.atleast_1d(x)))
    return np.sum((x-x0)**2)

if __name__ == '__main__':
    unittest.main()