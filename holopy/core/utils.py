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
Misc utility functions to make coding more convenient

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""


import os
import shutil
import errno
import numpy as np
from copy import copy
import itertools
import xarray as xr
import schwimmbad

def ensure_array(x):
    if isinstance(x, xr.DataArray):
        if x.shape==():
            if len(x.coords)==0:
                return np.array([x.item()])
            else:
                return x.expand_dims(x.coords)
        else:
            return x
    elif np.isscalar(x) or isinstance(x, bool) or (isinstance(x, np.ndarray) and x.shape==()):
        return np.array([x])
    elif x is None:
        return None
    else:
        return np.array(x)

def ensure_listlike(x):
    if x is None:
        return []
    try:
        iter(x)
        return x
    except TypeError:
        return [x]

def mkdir_p(path):
    '''
    Equivalent to mkdir -p at the shell, this function makes a
    directory and its parents as needed, silently doing nothing if it
    exists.
    '''
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else: raise # pragma: no cover

def dict_without(d, keys):
    """
    Exclude a list of keys from a dictionary.

    Silently ignores any key in keys that is not in the dict (this is
    intended to be used to make sure a dict does not contain specific
    keys)
    Parameters
    ----------
    d : dict
        The dictionary to operate on
    keys : list(string)
        The keys to exclude
    returns : d2
        A copy of dict without any of the specified keys
    """
    d = copy(d)
    for key in ensure_array(keys):
        try:
            del d[key]
        except KeyError:
            pass
    return d

def is_none(o):
    """
    Check if something is None.

    This can't be done with a simple is check anymore because numpy decided that
    array is None should do an element wise comparison.

    Parameters
    ----------
    o : object
        Anything you want to see if is None
    """

    return isinstance(o, type(None))

def updated(d, update={}, filter_none=True, **kwargs):
    """Return a dictionary updated with keys from update

    Analgous to sorted, this is an equivalent of d.update as a
    non-modifying free function

    Parameters
    ----------
    d : dict
        The dict to update
    update : dict
        The dict to take updates from

    """
    d = copy(d)
    for key, val in itertools.chain(update.items(), kwargs.items()):
        if val is not None or filter_none is False:
            d[key] = val

    return d

def repeat_sing_dims(indict, keys = 'all'):
    if keys == 'all':
        subdict = indict
    else:
       subdict = {key: indict[key] for key in keys}

    subdict = {key: ensure_array(val) for key, val in subdict.items()}
    maxlen = max([len(val) for val in subdict.values()])

    subdict={key:np.repeat(val, maxlen) for key, val in subdict.items() if len(val)==1}

    return updated(indict, subdict)

def choose_pool(parallel):
    """
    This is a remake of schwimmbad.choose_pool with a single argument that has more options.
    """
    if parallel is None:
        return schwimmbad.SerialPool()
    if parallel is 'mpi':
        pool = schwimmbad.MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        return pool
    if parallel is 'all':
        threads = os.cpu_count()
        return choose_pool(threads)
    if parallel is 'auto':
        if schwimmbad.MPIPool.enabled():
            return choose_pool('mpi')
        else:
            return choose_pool('all')
    if isinstance(parallel, int):
        return schwimmbad.MultiPool(parallel)
    raise TypeError("Could not interpret 'parallel' argument. Use an integer, 'mpi', 'all', 'auto', or None")

