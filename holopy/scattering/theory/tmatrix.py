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
Compute holograms using Mishchenko's T-matrix method for axisymmetric scatterers.  Currently uses

.. moduleauthor:: Anna Wang <annawang@seas.harvard.edu>
"""

import numpy as np
import subprocess
import tempfile
import os
import shutil
import copy
from ..scatterer import Sphere, Spheroid, Cylinder
from ..errors import TheoryNotCompatibleError, TmatrixFailure, DependencyMissing

from .scatteringtheory import ScatteringTheory
try:
    from .mie_f import mieangfuncs
except:
    pass

class Tmatrix(ScatteringTheory):
    """
    Computes scattering using the axisymmetric T-matrix solution by Mishchenko
    with extended precision.

    It can calculate scattering from axisymmetric scatterers such as cylinders and spheroids.
    Calculations for particles that are very large or have high aspect ratios may not converge.


    Attributes
    ----------
    delete : bool (optional)
        If true (default), delete the temporary directory where we store the
        input and output file for the fortran executable

    Notes
    -----
    Does not handle near fields.  This introduces ~5% error at 10 microns.

    """
    def __init__(self, delete=True):
        self.delete = delete
        path, _ = os.path.split(os.path.abspath(__file__))
        self.tmatrixlocation = os.path.join(path, 'tmatrix_f', 'S')
        if os.name == 'nt':
            self.tmatrixlocation += '.exe' 
        if not os.path.isfile(self.tmatrixlocation):
            raise DependencyMissing('Tmatrix')

        super().__init__()

    def _can_handle(self, scatterer):
        return isinstance(scatterer, Sphere) or isinstance(scatterer, Cylinder) or isinstance(scatterer, Spheroid)

    def _run_tmat(self, temp_dir):
        if os.name == 'nt':
            cmd = ['S.exe']
        else:
            cmd = ['./S']
        subprocess.check_call(cmd, cwd=temp_dir)
        return

    def _raw_scat_matrs(self, scatterer, pos, medium_wavevec, medium_index):
        temp_dir = tempfile.mkdtemp()
        shutil.copy(self.tmatrixlocation, temp_dir)

        angles = pos.T[:, 1:] * 180/np.pi
        outf = open(os.path.join(temp_dir, 'tmatrix_tmp.inp'), 'wb')

        med_wavelen = 2*np.pi/medium_wavevec
        if isinstance(scatterer, Sphere):
            rxy = scatterer.r
            rz = scatterer.r
            iscyl = False
            scatterer = copy.copy(scatterer)
            scatterer.rotation = (0,0,0)
        elif isinstance(scatterer, Spheroid):
            rxy = scatterer.r[0]
            rz = scatterer.r[1]
            iscyl = False
        elif isinstance(scatterer.Cylinder):
            rxy = scatterer.d/2
            rz = scatterer.h/2
            iscyl = True
        else:
            # cleanup and raise error
            outf.close()
            shutil.rmtree(temp_dir)
            raise TheoryNotCompatibleError(self, scatterer)

        # write the info into the scattering angles file in the following order:
        outf.write((str((3/2)**iscyl*(rz*rxy**2)**(1/3.))+'\n').encode('utf-8'))
        outf.write((str(med_wavelen)+'\n').encode('utf-8'))
        outf.write((str(scatterer.n.real/medium_index)+'\n').encode('utf-8'))
        outf.write((str(scatterer.n.imag/medium_index)+'\n').encode('utf-8'))
        outf.write((str(rxy/rz)+'\n').encode('utf-8'))
        outf.write((str(scatterer.rotation[2]*180/np.pi)+'\n').encode('utf-8'))
        outf.write((str(scatterer.rotation[1]*180/np.pi)+'\n').encode('utf-8'))
        outf.write((str(-1 - iscyl)+'\n').encode('utf-8'))
        outf.write((str(angles.shape[0])+'\n').encode('utf-8'))

        # Now write all the angles
        np.savetxt(outf, angles)
        outf.close()

        self._run_tmat(temp_dir)
        try:
            tmat_result = np.loadtxt(os.path.join(temp_dir, 'tmatrix_tmp.out'))
        except FileNotFoundError:
            #No output file
            raise TmatrixFailure(os.path.join(temp_dir, 'log'))
        if len(tmat_result)==0:
            #Output file is empty
            raise TmatrixFailure(os.path.join(temp_dir, 'log'))

        # columns in result are
        # s11.r s11.i s12.r s12.i s21.r s21.i s22.r s22.i
        # should be
        # s11 s12
        # s21 s22

        # Combine the real and imaginary components from the file into complex
        # numbers. Then scale by -ki due to Mishchenko's conventions in eq 5. of
        # Mishchenko, Applied Optics (2000).
        s = tmat_result[:,0::2] + 1.0j*tmat_result[:,1::2]
        s = s*(-2j*np.pi/med_wavelen)
        # Now arrange them into a scattering matrix, noting that Mishchenko's basis
        # vectors are different from the B/H, so we need to take that into account:
        scat_matr = np.array([[s[:,0], s[:,1]], [-s[:,2], -s[:,3]]]).transpose()

        if self.delete:
            shutil.rmtree(temp_dir)

        return scat_matr

    def _raw_fields(self, pos, scatterer, medium_wavevec, medium_index, illum_polarization):

        if not (np.array(illum_polarization)[:2] == np.array([1,0])).all():
            raise ValueError("Our implementation of Tmatrix scattering can only handle [1,0] polarization. Adjust your reference frame accordingly.")

        scat_matr = self._raw_scat_matrs(scatterer, pos, medium_wavevec=medium_wavevec, medium_index=medium_index)
        fields = np.zeros_like(pos.T, dtype = scat_matr.dtype)

        for i, point in enumerate(pos.T):
            kr, theta, phi = point
            # TODO: figure out why postfactor is needed -- it is not used in dda.py
            postfactor = np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
            escat_sph = mieangfuncs.calc_scat_field(kr, phi, np.dot(scat_matr[i],postfactor),[1,0])
            fields[i] = mieangfuncs.fieldstocart(escat_sph, theta, phi)
        return fields.T
