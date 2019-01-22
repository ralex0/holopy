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
Calculates holograms of spheres using an analytical solution of
the Mie scattered field imaged by a perfect lens.
Uses superposition to calculate scattering from multiple spheres.

.. moduleauthor:: Brian D. Leahy <bleahy@seas.harvard.edu>
.. moduleauthor:: Ron Alexander <ralexander@g.harvard.edu>
"""

import numpy as np

from ..scatterer import Sphere, Scatterers
from .scatteringtheory import ScatteringTheory
from .mielensfunctions import MieLensCalculator


class MieLens(ScatteringTheory):
    def __init__(self, lens_angle=1.0, calculator_accuracy_kwargs={}):
        super(MieLens, self).__init__()
        self.lens_angle = lens_angle
        self.calculator_accuracy_kwargs = calculator_accuracy_kwargs

    def _can_handle(self, scatterer):
        return isinstance(scatterer, Sphere)

    def _raw_fields(self, positions, scatterer, medium_wavevec, medium_index,
                    illum_polarization):
        """
        Parameters
        ----------
        positions : (3, N) numpy.ndarray
            The (k * r, theta, phi) coordinates, relative to the sphere,
            of the points to calculate the fields. Note that the radial
            coordinate is rescaled by the wavevector.
        scatterer : ``scatterer.Sphere`` object
        medium_wavevec : float
        medium_index : float
        illum_polarization : 2-element tuple
            The (x, y) field polarizations.
        """
        index_ratio = scatterer.n / medium_index
        size_parameter = medium_wavevec * scatterer.r

        r, theta, phi = positions
        z = r * np.cos(theta)
        rho = r * np.sin(theta)

        pol_angle = np.arctan2(
            illum_polarization.values[1], illum_polarization.values[0])
        phi += pol_angle
        phi %= (2 * np.pi)

        # FIXME mielens assumes that the detector points are at a fixed z!
        # right now I'm picking one z:
        particle_kz = np.mean(z)
        if np.ptp(z) > 1e-13 * (1 + np.abs(particle_kz)):
            msg = ("mielens currently assumes the detector is a fixed " +
                   "z from the particle")
            raise ValueError(msg)

        field_calculator = MieLensCalculator(
            particle_kz=particle_kz, index_ratio=index_ratio,
            size_parameter=size_parameter, lens_angle=self.lens_angle,
            **self.calculator_accuracy_kwargs)
        fields_pll, fields_prp = field_calculator.calculate_scattered_field(
            rho, phi)  # parallel and perp to the polarization

        # Transfer from (parallel to, perpendicular to) polarziation
        # to (x, y)
        parallel = np.array([np.cos(pol_angle), np.sin(pol_angle)])
        perpendicular = np.array([-np.sin(pol_angle), np.cos(pol_angle)])
        field_xyz = np.zeros([3, fields_pll.size], dtype='complex')
        for i in range(2):
            field_xyz[i, :] += fields_pll * parallel[i]
            field_xyz[i, :] += fields_prp * perpendicular[i]

        # Then we need to do 2 separate modifications to the fields.
        # First, in a lens, the incident field is Gouy phase shifted
        # to be E0 * -1, whereas in holopy the field is considered as
        # imaged without a phase shift. So since in holopy the incident
        # field, imaged by the lens, is phase shifted by -pi, we need
        # to phase-shift the scattered fields by -pi as well = multiply
        # by -1.
        # Second, holopy phase-shifts the reference wave by e^{ikz}.
        # For numerical reasons, in the mielens calculation I consider
        # the incident wave fixed and the scattered wave shifted by
        # e^{ikz}. So we need to fix this by multiplying by e^{ikz}.
        # Combined, we multiply by -1 * e^{ikz}:
        field_xyz *= -1 * np.exp(1j * particle_kz)
        return field_xyz

