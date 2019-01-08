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
from copy import copy
import unittest
import warnings

import numpy as np

from nose.plugins.attrib import attr

import holopy as hp
from holopy.core.io import get_example_data_path
from holopy.core.process import normalize, bg_correct
from holopy.core.tests.common import get_example_data
from holopy.inference import GradientFreeStrategy
from holopy.inference.model import AlphaModel
from holopy.inference.prior import Prior, Uniform
from holopy.scattering import Sphere, Mie
from holopy.scattering.scatterer import _expand_parameters

from nevergrad.optimization import optimizerlib

class TestGradientFreeStrategy(unittest.TestCase):
    def test_nevergrad_OnePlusOne(self):
        cost = lambda x: _simple_cost_function(x, x_obs=1.0)
        optimizer = optimizerlib.registry['OnePlusOne'](dimension=1, budget=256)
        result = optimizer.optimize(cost, executor=None, batch_mode=True)
        result_ok = np.allclose(result, 1.0, rtol=.001)
        self.assertTrue(result_ok)

    def test_nevergrad_CMA(self):
        cost = lambda x: _simple_cost_function(x, x_obs=np.array([1.0, 1.0]))
        optimizer = optimizerlib.registry['CMA'](dimension=2, budget=512)
        result = optimizer.optimize(cost, executor=None, batch_mode=True)
        result_ok = np.allclose(result, [1.0, 1.0], rtol=.001)
        self.assertTrue(result_ok)

    def test_GradientFreeStrategy(self):
        data = 0.5
        model = _SimpleModel(x = Uniform(0, 1))
        strat = GradientFreeStrategy()
        result = strat.optimize(model, data)
        result_ok = np.allclose(result.parameters['x'], .5, rtol=.001)
        self.assertTrue(result_ok)

    def test_fit_gold_particle(self):
        gold_alpha = .6497
        gold_sphere = Sphere(n=1.582+1e-4j, r=6.484e-7, 
                             center=(5.534e-6, 5.792e-6, 1.415e-5))

        data = _load_gold_example_data()
        scatterer_guess, alpha_guess = _get_gold_param_guesses()

        model = AlphaModel(scatterer_guess, theory=Mie(compute_escat_radial=False),
                           alpha=alpha_guess)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitter = GradientFreeStrategy(budget=256)
            result = fitter.optimize(model, data)
        
        fitted = result.scatterer

        n_isok = np.isclose(fitted.n, gold_sphere.n, atol=0.05)
        r_isok = np.isclose(fitted.r, gold_sphere.r, atol=0.1)
        center_isok = np.isclose(fitted.center, gold_sphere.center, rtol=0.1)
        alpha_isok = np.isclose(result.parameters['alpha'], gold_alpha, atol=0.3)
        self.assertTrue(all([n_isok, r_isok, *center_isok, alpha_isok]))
        self.assertEqual(model, result.model)

    @attr('slow')
    def test_fit_polystyrene_particle(self):
        data = _load_PS_example_data()
        scatterer_guess, alpha_guess = _get_PS_param_guesses()
        model = AlphaModel(scatterer=scatterer_guess, theory=Mie(), alpha=alpha_guess)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_strategy = GradientFreeStrategy(budget=256)
            result = fit_strategy.optimize(model, data)

        fitted = result.scatterer

        n_isok = np.isclose(fitted.n, 1.59, atol=1e-2)
        r_isok = np.isclose(fitted.r, 0.5, atol=0.1)
        center_isok = np.isclose(fitted.center, (24.17,21.84,16.42), rtol=0.1)
        alpha_isok = np.isclose(result.parameters['alpha'], 0.7, atol=0.3)
        isok = [n_isok, r_isok, *center_isok, alpha_isok]
        self.assertTrue(all(isok))
        self.assertEqual(model, result.model)

def _load_PS_example_data():
    imagepath = get_example_data_path('image01.jpg')
    raw_holo = hp.load_image(imagepath, spacing = 0.0851, medium_index=1.33, 
                                        illum_wavelen=0.660, 
                                        illum_polarization=(1,0))
    bgpath = get_example_data_path(['bg01.jpg', 'bg02.jpg', 'bg03.jpg'])
    bg = hp.core.io.load_average(bgpath, refimg = raw_holo)
    holo = bg_correct(raw_holo, bg)
    return normalize(holo)

def _load_gold_example_data():
    return normalize(get_example_data('image0001'))

def _get_PS_param_guesses():
    center_guess = [Uniform(0, 100, name='x', guess=24),
                    Uniform(0, 100, name='y', guess=22),
                    Uniform(0, 30, name='z', guess=15)]
    scatterer_guess = Sphere(n=Uniform(1, 2, name='n', guess=1.59),
                             r=Uniform(1e-8, 5, name='r', guess=.5),
                             center=center_guess)
    alpha_guess = Uniform(0.1, 1, name='alpha', guess=0.7)
    return scatterer_guess, alpha_guess

def _get_gold_param_guesses():
    center_guess = [Uniform(0, 1e-5, name='x', guess=5.67e-6),
                    Uniform(0, 1e-5, name='y', guess=5.76e-6),
                    Uniform(1e-5, 2e-5, name='z', guess=1.5e-5)]

    scatterer_guess = Sphere(n=Uniform(1, 2, name='n', guess=1.59),
                       r=Uniform(1e-8, 1e-5, name='r', guess=8.5e-7),
                       center=center_guess)

    alpha_guess = Uniform(0.1, 1, name='alpha', guess=0.6)
    return scatterer_guess, alpha_guess

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
