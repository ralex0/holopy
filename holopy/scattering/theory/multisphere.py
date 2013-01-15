# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
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
Defines Multisphere theory class, which calculates scattering for multiple
spheres using the (exact) superposition method implemented in
modified version of Daniel Mackowski's SCSMFO1B.FOR.  Uses full radial
dependence of spherical Hankel functions for the scattered field.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""
from __future__ import division

import numpy as np
from scipy.integrate import dblquad
from .mie_f import mieangfuncs
from .mie_f import scsmfo_min
from .mie_f import uts_scsmfo
from ..scatterer import Spheres
from ..errors import TheoryNotCompatibleError, UnrealizableScatterer
from .scatteringtheory import FortranTheory

class Multisphere(FortranTheory):
    """Calculate the scattered field of a collection of spheres through a
    numerical method that accounts for multiple scattering and near-field
    effects (see [Fung2011]_, [Mackowski1996]_).  This approach is much more
    accurate than Mie superposition, but it is also more computationally
    intensive.  The Multisphere code can handle any number of spheres.
    TODO: mention scfodim somewhere.

    Attributes
    ----------
    imshape : float or tuple (optional)
        Size of grid to calculate scattered fields or
        intensities. This is the shape of the image that calc_field or
        calc_intensity will return
    phi : array
        Specifies azimuthal scattering angles to calculate (incident
        direction is z)
    theta : array
        Specifies polar scattering angles to calculate
    optics : :class:`holopy.optics.Optics` object
        specifies optical train
    niter : integer (optional)
        maximum number of iterations to use in solving the interaction
        equations
    meth : integer (optional)
        method to use to solve interaction equations.  Set to 0 for
        biconjugate gradient; 1 for order-of-scattering
    eps : float (optional)
        relative error tolerance in solution for interaction equations
    qeps1 : float (optional)
        error tolerance used to determine at what order the
        single-sphere spherical harmonic expansion should be truncated
    qeps2 : float (optional)
        error tolerance used to determine at what order the cluster
        spherical harmonic expansion should be truncated

    Notes
    -----
    According to Mackowski's manual for SCSMFO1B.FOR [1]_ and later
    papers [2]_, the biconjugate gradient is generally the most
    efficient method for solving the interaction equations, especially
    for dense arrays of identical spheres.  Order-of-scattering may
    converge better for non-identical spheres.

    Multisphere does not check for overlaps becaue overlapping spheres can be
    useful for getting fits to converge.  The results to be sensible for small
    overlaps even though mathemtically speaking they are not xstrictly valid.

    References
    ----------
    .. [1] Daniel W. Mackowski, SCSMFO.FOR: Calculation of the Scattering
       Properties for a Cluster of Spheres,
       ftp://ftp.eng.auburn.edu/pub/dmckwski/scatcodes/scsmfo.ps

    .. [2] D.W. Mackowski, M.I. Mishchenko, A multiple sphere T-matrix
       Fortran code for use on parallel computer clusters, Journal of
       Quantitative Spectroscopy and Radiative Transfer, In Press,
       Corrected Proof, Available online 11 March 2011, ISSN 0022-4073,
       DOI: 10.1016/j.jqsrt.2011.02.019.

    """

    def __init__(self, niter=200, eps=1e-6, meth=1, qeps1=1e-5, qeps2=1e-8):
        self.niter = niter
        self.eps = eps
        self.meth = meth
        self.qeps1 = qeps1
        self.qeps2 = qeps2

        # call base class constructor
        super(Multisphere, self).__init__()

    def _scsmfo_setup(self, scatterer, optics):
        """
        Given multiple spheres, calculate amn coefficients for scattered
        field expansion in VSH using SCSMFO.

        Parameters
        ----------
        scatterer : :mod:`.scatterer` object
            scatterer or list of scatterers to compute field for
        optics: :mod:`.Optics` object 
            optical metadata

        Returns
        -------
        amn : arrays of field expansion coefficients
            
        """
        if not isinstance(scatterer, Spheres):
            raise TheoryNotCompatibleError(self, scatterer)
        # check for spheres being uniform
        for sph in scatterer.scatterers:
            if not np.isscalar(sph.n):
                raise TheoryNotCompatibleError(self, scatterer)

        # check that the parameters are in a range where the multisphere
        # expansion will work
        for s in scatterer.scatterers:
            if s.r * optics.wavevec > 1e3:
                raise UnrealizableScatterer(self, s, "radius too large, field "+
                                            "calculation would take forever")

        # switch to centroid weighted coordinate system tmatrix code expects
        # and nondimensionalize
        centers = (scatterer.centers - scatterer.centers.mean(0)) * optics.wavevec

        m = scatterer.n / optics.index

        if (centers > 1e4).any():
            raise UnrealizableScatterer(self, scatterer, "Particle separation "
                                        "too large, calculation would take forever")

        _, lmax, amn0, converged = scsmfo_min.amncalc(
            1, centers[:,0],  centers[:,1],
            # The fortran code uses oppositely directed z axis (they have laser
            # propagation as positive, we have it negative), so we multiply the
            # z coordinate by -1 to correct for that.
            -1.0 * centers[:,2],  m.real, m.imag,
            scatterer.r * optics.wavevec, self.niter, self.eps,
            self.qeps1, self.qeps2,  self.meth, (0,0))

        # converged == 1 if the SCSMFO iterative solver converged
        # f2py converts F77 LOGICAL to int
        if not converged:
            raise ConvergenceFailureMultisphere()

        # chop off unused parts of amn0, the fortran code currently has a hard
        # coded number of parameters so it will return too many coefficients.
        # We truncate here to reduce the length of stuff we have to compute with
        # later.
        limit = lmax**2 + 2*lmax
        amn = amn0[:, 0:limit, :]

        if np.isnan(amn).any():
            raise MultisphereExpansionNaN()

        return amn, lmax

    def _calc_field(self, scatterer, schema):
        """
        Calculate fields for single or multiple spheres

        Parameters
        ----------
        scatterer : :mod:`.scatterer` object
            scatterer or list of scatterers to compute field for
        selection : array of integers (optional)
            a mask with 1's in the locations of pixels where you
            want to calculate the field, defaults to all pixels

        Returns
        -------
        xfield, yfield, zfield : complex arrays with shape `imshape`
            x, y, z components of scattered fields

        """
        amn, lmax = self._scsmfo_setup(scatterer, schema.optics)
       
        positions = schema.positions_kr_theta_phi(origin = scatterer.centers.mean(0)).T

        fields = mieangfuncs.tmatrix_fields(positions, amn, lmax, 0, schema.optics.polarization)

        if np.isnan(fields[0][0]):
            raise MultisphereFieldNaN(self, scatterer, '')

        return self._finalize_fields(scatterer.z.mean(), fields, schema)

    def _calc_cext(self, scatterer, optics, amn = None, lmax = None):
        """
        Calculate extinction cross section via optical theorem.
        """
        # normalize the polarization
        pol = optics.polarization / np.sqrt((optics.polarization**2).sum())

        # calculate amn coefficients if need be
        if amn is None:
            amn, lmax = self._scsmfo_setup(scatterer, optics)

        # calculate forward scattering
        asm_fwd = _asm_far(0., 0., amn, lmax)
        ainc_sph = pol * np.array([1., -1.]) # assume theta, phi = 0 
        ascat_sph = np.dot(asm_fwd, ainc_sph) * np.array([1., -1.]) 
        # at theta, phi = 0, ascat_cart = ascat_sph
        cext = 4. * np.pi / optics.wavevec**2 * np.dot(pol, ascat_sph).real
        return cext

    def _calc_scat_matrix(self, scatterer, schema):
        amn, lmax = self._scsmfo_setup(scatterer, schema.optics)
        # TODO: consult Tom on schema (not just thetas, but phis too)
        # scat_matrs = []
        # return np.array(scat_matrs)
        
    def _calc_cscat(self, scatterer, optics, amn = None, lmax = None):
        """
        Calculate scattering cross section by quadrature over solid angle.
        """
        # normalize the polarization
        pol = optics.polarization / np.sqrt((optics.polarization**2).sum())

        # calculate amn coefficients if need be
        if amn is None:
            amn, lmax = self._scsmfo_setup(scatterer, optics)

        # define integrand: A^2 sin theta (vector scattering amplitude A)
        def ampsq(theta, phi):
            einc = mieangfuncs.incfield(*pol, phi = phi)
            asm = _asm_far(theta, phi, amn, lmax)
            ascat_sph = np.dot(asm, einc) # in par/perp basis
            ascatsq = (np.abs(ascat_sph)**2).sum()
            return ascatsq * np.sin(theta)

        integral = _integrate4pi(ampsq)

        cscat = integral / optics.wavevec**2
        return cscat

    def _calc_asym(self, optics, amn, lmax):
        """
        Calculate asymmetry parameter <cos theta> by quadrature over
        solid angle.
        """
        # normalize the polarization
        pol = optics.polarization / np.sqrt((optics.polarization**2).sum())

        # define integrand: A^2 sin theta cos theta 
        def costhetawt(theta, phi):
            einc = mieangfuncs.incfield(*pol, phi = phi)
            asm = _asm_far(theta, phi, amn, lmax)
            ascat_sph = np.dot(asm, einc) # in par/perp basis
            ascatsq = (np.abs(ascat_sph)**2).sum()
            return ascatsq * np.sin(theta) * np.cos(theta)

        integral = _integrate4pi(costhetawt)

        asym = integral / optics.wavevec**2 # need to divide by cscat
        return asym

    def _calc_cross_sections(self, scatterer, optics):
        """
        Calculate scattering, absorption, and extinction cross
        sections, and asymmetry parameter for sphere clusters
        with polarized incident light.

        The extinction cross section is calculated from the optical 
        theorem. The scattering cross section is calculated by 
        numerical quadrature of the scattered field, and the absorption
        cross section is calculated from the difference of the extinction
        cross section and the scattering cross section.

        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            sphere cluster to compute for
        
        Returns
        -------
        cross_sections : array (4)
            Dimensional scattering, absorption, and extinction
            cross sections, and <cos \theta>
        """
        # normalize the polarization
        pol = optics.polarization / np.sqrt((optics.polarization**2).sum())
        # calculate amn coefficients
        amn, lmax = self._scsmfo_setup(scatterer, optics)

        cext = self._calc_cext(scatterer, optics, amn, lmax)
        cscat = self._calc_cscat(scatterer, optics, amn, lmax)
        cabs = cext - cscat
        asym = self._calc_asym(optics, amn, lmax) / cscat

        return np.array([cscat, cabs, cext, asym])


def _asm_far(theta, phi, amn, lmax):
        """
        far field amplitude scattering matrix for fixed angles
        """
        asm = np.roll(uts_scsmfo.asm(amn, lmax, theta, phi), 
                      -1).reshape((2,2)) * -0.5 # correction factor
        return asm

def _integrate4pi(integrand):
    integral, error = dblquad(integrand, 0, 2 * np.pi, lambda theta:0., 
                              lambda theta:np.pi)
    return integral

class MultisphereFieldNaN(UnrealizableScatterer):
    def __str__(self):
        return "Fields computed with Multisphere are NaN, this probably represents a failure of \
the code to converge, check your scatterer."


class MultisphereExpansionNaN(Exception):
    def __str__(self):
        return ("Internal expansion for Multisphere coefficients contains "
                "NaN.  This probably means your scatterer is unphysical.")

class ConvergenceFailureMultisphere(Exception):
    def __str__(self):
        return ("Multisphere calculations failed to converge, this probably means "
                "your scatterer is unphysical, or possibly just huge")
