from .utils import mkdir_p
from .utils import setup_logging
from .undertaker import Undertaker
from .gabby_plotter import GabbyPlotter
from .gabby_data_model import GabbyDataModel
from .bstar_plotter import BStarPlotter
from .cache import GabbyCache
from .transformer import Jazz
from .transformer import Optimus
from .moral_decay import MoralDecay
from .tle import TLE
from .db import GabbyDB
from .faker import FakeDB
from .propagator import keplerian_period
from .propagator import StatsPropagator
from .db import CloudDescriptor

from .defs import *

"""Package for reading TLE data and producing useful plots/animations.


This one will be useful for anyone who wants to unpack a TLE using
TLE_STRUCT_FMT

pack_vals = [n, ndot, nddot, bstar, tle_num, inc,
             raan, ecc, argp, mean_anomaly, rev_num]

# TLE Format in the DB in order of appearance in the struct packing
# Value        fmt Description/units
# n             f  Mean Motion (revs/day)
# ndot          f  dn/dt
# nddot         f  d^2n/dt^2
# bstar         f  B* drag term
# tle_num       i  Observation number
# inc           f  Inclination (deg)
# raan          f  Right Ascension (deg)
# ecc           f  Eccentricity (dimensionless, 0-1)
# argp          f  Argument of Perigee
# mean_anomaly  f  Mean anomaly (phase in deg)
# rev_num       i  Number of orbits from epoch at time of observation



"""
