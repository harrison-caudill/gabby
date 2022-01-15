import datetime
import gc
import lmdb
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pprint
import subprocess
import struct
import sys
import tletools
import scipy
import scipy.interpolate
import scipy.signal

from .defs import *
from .utils import *

"""
"""


class Propagator(object):
    """Parent class for data propagators.
    """
    pass


class StatsPropagator(object):
    """Forward propagator using basic statistical distribution.

    Using historical data regarding decay rates as a function of
    apogee and perigee,
    """

    def __init__(self, cfg, moral_decay, ):
        self.cfg = cfg
        self.db_env = env
        self.db_apt = apt
        self.db_tle = tle
        self.db_scope = scope


def keplerian_period(A, P):
    """Determines the period of a keplerian ellipitical earth orbit.

    A: <np.array> or float
    P: <np.array> or float
    returns: <np.array> or float

    Does NOT take into account oblateness or the moon.
    """
    Re = (astropy.constants.R_earth/1000.0).value
    RA = A+Re
    RP = P+Re
    e = (RA-RP) / (RA+RP)

    # These are all in meters
    a = 1000*(RA+RP)/2
    mu = astropy.constants.GM_earth.value
    T = 2 * np.pi * (a**3/mu)**.5 / 60.0

    return T
