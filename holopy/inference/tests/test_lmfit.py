import unittest

import numpy as np

import holopy as hp
from holopy.core.process import normalize
from holopy.inference import Optimizer, AlphaModel, PerfectLensModel
from holopy.inference.prior import Uniform
from holopy.scattering import Sphere

GOLD_PARTICLE_INITIAL_GUESS = {'x': .567e-5, 'y': .576e-5, 'z': 15e-6,
                               'n': 1.59, 'r': 8.5e-7, 'alpha': 0.6}

GOLD_PARTICLE_TARGET_FIT = {'x': 5.534e-6, 'y': 5.792e-6, 'z': 1.415e-5,
                            'n': 1.582+1e-4j, 'r': 6.484e-7, 'alpha': .6497}

class TestOptimizer(unittest.TestCase):
    def test_fit_AlphaModel_leastsq(self):
        data = _load_gold_particle_example_data()
        guess = GOLD_PARTICLE_INITIAL_GUESS
        model = _default_AlphaModel()
        optimizer = Optimizer(model=model, method='leastsq')
        result = optimizer.fit(data, guess)
        sphere_ok, alpha_ok = _check_AlphaModel_fit_close(result.params, truth=GOLD_PARTICLE_TARGET_FIT)
        self.assertTrue(sphere_ok)
        self.assertTrue(alpha_ok)

    def test_fit_PerfectLensModel_leastsq(self):
        data = _load_gold_particle_example_data()
        guess = GOLD_PARTICLE_INITIAL_GUESS
        model = _default_PerfectLensModel()
        optimizer = Optimizer(model=model, method='leastsq')
        result = optimizer.fit(data, guess)
        sphere_ok = _check_PerfectLensModel_fit_close(result.params, truth=GOLD_PARTICLE_TARGET_FIT)
        self.assertTrue(sphere_ok)


def _load_gold_particle_example_data():
    return normalize(hp.load(hp.core.io.get_example_data_path('image0001.h5')))


def _default_AlphaModel():
    n, r, x, y, z, alpha = [Uniform(0, 1)]*6
    return AlphaModel(Sphere(n=n, r=n, center=(x,y,z)), alpha=alpha)


def _check_AlphaModel_fit_close(result, truth):
    keys = ['x', 'y', 'z', 'n', 'r']
    sph_ok = [np.allclose(result[k], truth[k], rtol=1e-3) for k in keys]
    alpha_ok = np.allclose(result['alpha'], truth['alpha'], rtol=0.1)
    return sph_ok, alpha_ok


def _default_PerfectLensModel():
    n, r, x, y, z, lens_angle = [Uniform(0, 1)]*6
    return PerfectLensModel(Sphere(n=n, r=n, center=(x,y,z)), lens_angle=lens_angle)


def _check_PerfectLensModel_fit_close(result, truth):
    keys = ['x', 'y', 'z', 'n', 'r']
    sph_ok = [np.allclose(result[k], truth[k], rtol=1e-3) for k in keys]
    return sph_ok


if __name__ == '__main__':
    unittest.main()
