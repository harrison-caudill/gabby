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
                 tgt=None,
                 drag=None):
        self.global_cache = global_cache
        self.tgt_cache = tgt_cache
        self.db = db
        self.cfg = cfg
        self.tgt = tgt
        self.drag = drag
        self._init_global_stats()

    def _cfg_hash(self, cfg):
        tmp = dict(cfg.items())

        vals = [(k, tmp[k]) for k in sorted(tmp.keys())]
        m = hashlib.sha256()
        m.update(str(vals).encode())
        return m.hexdigest()

    def _sats_hash(self, cfg):
        sats = json.loads(cfg['historical-asats'])
        m = hashlib.sha256()
        m.update((','.join(sorted(sats))).encode())
        return m.hexdigest()

    def _deriv_cache_name(self, stats_cfg):
        return 'deriv-' + self._sats_hash(stats_cfg)

    def _decay_cache_name(self, stats_cfg):
        return 'moral_decay-' + self._cfg_hash(stats_cfg)

    def _filtered_cache_name(self, stats_cfg):
        return 'filtered-' + self._sats_hash(stats_cfg)

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

        decay_name = self._decay_cache_name(stats_cfg)
        if decay_name in self.global_cache:
            logging.info(f"  Found moral decay in the cache")
            decay = self.decay = self.global_cache[decay_name]
        else:
            logging.info(f"  Cache is free from moral decay, let's make some")

            base_frags = json.loads(stats_cfg['historical-asats'])
            fragments = self.db.find_daughter_fragments(base_frags)
            self.apt = apt = self.db.load_apt(fragments)

            filtered_name = self._filtered_cache_name(stats_cfg)
            deriv_name = self._deriv_cache_name(stats_cfg)

            if deriv_name in self.global_cache:
                logging.info(f"  Found filtered/derivative values in global cache")
                deriv = self.global_cache[deriv_name]
                filtered = self.global_cache[filtered_name]
            else:
                logging.info(f"  Stats not found in cache -- building anew")
                
                filtered, deriv = jazz.filtered_derivatives(apt,
                                                            min_life=1.0,
                                                            dt=SECONDS_IN_DAY,
                                                            fltr=jazz.lpf())
                logging.info(f"  Saving derivatives to cache")
                self.global_cache[deriv_name] = deriv
                self.global_cache[filtered_name] = filtered

            # Uncomment for debug plots
            idx = 1
            apt.plot('output/apt.png', idx, title=f"positions {deriv.fragments[idx]}")
            filtered.plot('output/filtered.png', idx, title=f"filtered {deriv.fragments[idx]}")
            deriv.plot('output/deriv.png', idx, title=f"derivative {deriv.fragments[idx]}")

            self.decay = decay = jazz.decay_rates(apt, filtered, deriv,
                                                  drag=self.drag)

            logging.info(f"  Adding a little moral decay to the cache")
            self.global_cache[decay_name] = decay

        decay.plot_mesh('output/mesh.png', data='median')
        #decay.plot_dA_vs_P('output/avp-%(i)2.2d.png')

    def propagate(self, data, fwd=True, rev=True):
        """Propagates the <data> forward according to <tgt>.

        Global statistics necessary to perform the forward propagation
        are collected in the initialization phase.
        """

        logging.info(f"Propagating")

        L = data.L
        N = data.N

        dt = data.dt.total_seconds()/24/3600.0

        decay_alt = self.tgt.getint('decay-altitude')

        if rev:
            # If we're doing reverse propagation then we assume that
            # all of the fragments come into scope at the time of the
            # incident.
            for i in range(L):
                data.scope_start[data.names[i]] = data.incident_ts

        # The beginning is a good place to begin
        t = data.start_ts

        # FIXME: Can do multiprocessing across fragments

        # Forward pass
        for i in range(N-1):
            logging.info(f"  Propagating Frame: {i+1} / {N}")
            for j in range(L):
                frag = data.names[j]
                A = data.As[i][j]
                P = data.Ps[i][j]

                if A and not data.As[i+1][j]:
                    # We have data now, but not in the future.  We
                    # should evaluate this frame for propagation.

                    # The perigee is already at or below the decay
                    # altitude, so we're going to drop it off the map
                    # now.
                    if P <= decay_alt:
                        data.scope_end[frag] = t
                        data.valid[i+1][j] = 0
                        continue

                    # Find the indexes into the tables
                    idx_A, idx_P = self.decay.index_for(A, P)

                    # Find the decay rates (dA/dt and dP/dt)
                    rate_A = self.decay.median[0][idx_A][idx_P]
                    rate_P = self.decay.median[1][idx_A][idx_P]
                    if abs(rate_P) > abs(rate_A):
                        # FIXME: This is a problem with Moral Decay.
                        # Sometimes the perigee decay rate exceeds the
                        # apogee decay rate.  At a glance, this seems
                        # to happen when we have few data points to go
                        # on so noise in the data has an outsized
                        # impact.  When this happens, as a hack, we
                        # swap it round.  This issue will be fixed
                        # when we switch to a history-informed
                        # physical model in the non-descript future.
                        tmp = rate_P
                        rate_P = rate_A
                        rate_A = tmp

                    # Compute the new A/P values
                    A = A + dt * rate_A
                    P = P + dt * rate_P

                    # Because the apogee decay right is higher we can
                    # sometimes judge ourselves just over the line and
                    # invert the apogee/perigee.
                    if A >= P:
                        data.As[i+1][j] = A
                        data.Ps[i+1][j] = P
                    else:
                        data.As[i+1][j] = P
                        data.Ps[i+1][j] = A

                    data.Ts[i+1][j] = keplerian_period(data.As[i+1][j],
                                                       data.Ps[i+1][j])
                    data.valid[i+1][j] = 1
            t += dt

        # Update the number of valid values
        data.Ns = np.sum(data.valid, axis=1, dtype=np.int64)


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
