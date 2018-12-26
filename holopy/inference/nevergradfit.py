"""
.. moduleauthor:: Ron Alexander <ralexander@g.harvard.edu>
"""
import time

import numpy as np

from holopy.core.holopy_object import HoloPyObject
from holopy.inference.result import InferenceResult, UncertainValue

from nevergrad.optimization import optimizerlib

class NevergradStrategy(HoloPyObject):
    def __init__(self, optimizer='CMA', budget=100):
        self.optimizer = optimizer
        self.budget = budget

    def optimize(self, model, data):
        time_start = time.time()
        cost = self._make_cost_function(model, data)
        dim = self._get_model_diminsion(model)
        optimizer = self._setup_optimizer(dim)
        reccomendation = optimizer.optimize(cost, executor=None, batch_mode=True)
        parameters = {key: value for key, value in zip(model.parameters.keys(), reccomendation)}

        perrors = self._estimate_error_from_fit()
        intervals = [UncertainValue(value, perrors, name=key)
                     for key, value in parameters.items()]

        d_time = time.time() - time_start
        result = InferenceResult(data=data, model=model, strategy=optimizer, intervals=intervals, time=d_time)
        return result

    def _make_cost_function(self, model, data):
        return lambda x: model.cost(x, data)

    def _get_model_diminsion(self, model):
        return len(model.parameters)

    def _setup_optimizer(self, dimension):
        return optimizerlib.registry[self.optimizer](dimension=dimension, 
                                                     budget=self.budget)

    def _estimate_error_from_fit(self):
        return 0
