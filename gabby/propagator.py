import datetime
import gc
import json
import hashlib
import lmdb
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pprint
import scipy
import scipy.interpolate
import scipy.signal
import struct
import subprocess
import sys
import tletools

from .defs import *
from .utils import *
from .transformer import Jazz

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

    def __init__(self,
                 global_cache=None,
                 tgt_cache=None,
                 db=None,
                 cfg=None,
                 tgt=None):
        self.global_cache = global_cache
        self.tgt_cache = tgt_cache
        self.db = db
        self.cfg = cfg
        self.tgt = tgt
        self._init_global_stats()

    def _deriv_cache_name(self, stats_cfg):
        tmp = dict(stats_cfg.items())

        vals = [(k, tmp[k]) for k in sorted(tmp.keys())]
        m = hashlib.sha256()
        m.update(str(vals).encode())
        return 'deriv-' + m.hexdigest()

    def _init_global_stats(self):
        """Ensures the instance has a copy of MoralDecay in memory.

        Just a thin wrapper around the cache/transformer.
        """

        logging.info(f"Initializing Global Fragment Statistics")

        # Jazz will do all the heavy lifting here
        jazz = Jazz(self.cfg,
                    global_cache=self.global_cache,
                    tgt_cache=self.tgt_cache)

        stats_cfg = self.cfg['stats']
        base_frags = json.loads(stats_cfg['historical-asats'])
        fragments = self.db.find_daughter_fragments(base_frags)
        apt = self.db.load_apt(fragments)

        name = self._deriv_cache_name(stats_cfg)

        if name in self.global_cache:
            logging.info(f"  Found previous value in global cache")
            return self.global_cache[name]
        else:
            logging.info(f"  Stats not found in cache -- building anew")


            filtered, deriv = jazz.filtered_derivatives(apt,
                                                        min_life=1.0,
                                                        dt=SECONDS_IN_DAY,
                                                        fltr=jazz.lpf())

            # logging.info(f"  Saving derivatives to cache")
            # self.global_cache[name] = deriv


        sys.exit(0)

        
        # flt = jazz.apply_filter(apt, jazz.lpf())

    # def decay_rates(self, positions, derivatives, Ns,
    #                 mesh_output=None):
    #     moral_decay = jazz.decay_rates()

    def propagate(tgt, data):
        """Propagates the <data> forward according to <tgt>.

        Global statistics necessary to perform the forward propagation
        are collected in the initialization phase.
        """
        assert(False)


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
