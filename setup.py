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

'''
setup.py

uses numpy.distutils instead of standard distutils so that we can
build fortran extensions

Notes:

The t-matrix and mie scattering codes require working Fortran 90 and
C compilers, as well as f2py and cython. On Ubuntu, you will need the
"gfortran" and "python-dev" packages installed.
'''

import setuptools
import subprocess
import os
from os.path import join
from numpy.distutils.core import setup, Extension


# this will automatically build the scattering extensions, using the
# setup.py files located in their subdirectories
def configuration(parent_package='',top_path=''):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None,parent_package,top_path)

    pkglist=setuptools.find_packages()
    print(pkglist)
    for i in pkglist:
        config.add_subpackage(i)

    config.add_data_files(['.',['AUTHORS']])
    config.add_data_files(join('holopy','scattering','tests','gold','full_data','*.h5'))
    config.add_data_files(join('holopy','scattering','tests','gold','*.yaml'))
    config.add_data_files(join('holopy','core/tests','exampledata','*.h5'))
    config.add_data_files(join('holopy','core/tests','exampledata','*.jpg'))
    config.add_data_files(join('holopy','propagation','tests','gold','full_data','*.h5'))
    config.add_data_files(join('holopy','propagation','tests','gold','*.yaml'))
    if os.name == "nt":
        config.add_data_files(join('holopy', 'scattering', 'theory', 'tmatrix_f', 'S.exe'))
    else:
        config.add_data_files(join('holopy', 'scattering', 'theory', 'tmatrix_f', 'S'))

    config.get_version()
    return config

__version__ = 'unknown'
try:
    from holopy import __version__
except ImportError:
    # no version specified, or file got deleted in bzr
    pass

if __name__ == "__main__":

    #make Tmatrix fortran code
    if os.name == 'nt':
	    subprocess.check_call('gfortran -o S.exe S.lp.f lpq.f',cwd=join('holopy','scattering','theory','tmatrix_f'))
    else:
    	subprocess.check_call(['make'], cwd=join('holopy','scattering','theory','tmatrix_f'))

    requires=[l for l in open("requirements.txt").readlines() if l[0] != '#']
    setup(configuration=configuration,
          name='HoloPy',
          version=__version__,
          description='Holography in Python',
          install_requires=requires,
          author='Manoharan Lab, Harvard University',
          author_email='vnm@seas.harvard.edu',
          url='http://manoharan.seas.harvard.edu/holopy',
          license='GNU GPL',
          test_suite='nose.collector',
          package=['HoloPy'])
