import copy
import datetime
import gc
import json
import hashlib
import lmdb
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
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


class StatsPropagatorContext(object):

    def __init__(self, prop, data, fwd, rev, prop_after_obs):

        self.fwd = fwd
        self.rev = rev
        self.prop_after_obs = prop_after_obs

        self.As = data.As
        self.Ps = data.Ps
        self.Ns = data.Ns
        self.Ns_obs = None
        self.Ts = data.Ts
        self.valid = data.valid
        self.scope_start = data.scope_start
        self.scope_end = data.scope_end
        self.L = data.L
        self.N = data.N
        self.dt = data.dt.total_seconds()/24/3600.0
        self.start_ts = data.start_ts
        self.names = data.names
        self.decay = prop.decay

        self.decay_alt = prop.tgt.getint('decay-altitude')

        if rev:
            # If we're doing reverse propagation then we assume that
            # all of the fragments come into scope at the time of the
            # incident.
            for i in range(self.L):
                data.scope_start[data.names[i]] = data.incident_ts

        self.decay_alts = np.zeros(self.L) + self.decay_alt

        # If we're dropping them before they fully decay, then we'll
        # want to first find the altitude of the last observation.
        self.drop_early = prop.tgt.getboolean('drop-early-losses')
        if self.drop_early:
            for j in range(self.L):
                for i in range(self.N-1):
                    P = data.Ps[i][j]
                    if P and not data.Ps[i+1][j]:
                        self.decay_alts[j] = P
                        break


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
                                                            dt=SECONDS_IN_DAY)
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

    def propagate(self, data,
                  fwd=True,
                  rev=True,
                  prop_after_obs=False,
                  n_threads=1):
        """Propagates the <data> forward according to <tgt>.

        prop_after_obs: Means that we start propagating as soon as we
                        observe the fragment.  That way we can compare
                        the propagator to the observations.

        Global statistics necessary to perform the forward propagation
        are collected in the initialization phase.
        """

        logging.info(f"Propagating")

        ctx = StatsPropagatorContext(self, data, fwd, rev, prop_after_obs)

        data.Ns_obs = data.Ns

        if n_threads > 1:
            # Interleave generation so that they're generated roughly
            # in order in parallel rather than one block at a time.
            # [ 0,  N+0,  2N+0, ...]
            # [ 1,  N+1,  2N+1, ...]
            # ...
            # [N-1, 2N-1, 3N-1, ...]
            tmp = np.linspace(0, ctx.L-1, ctx.L, dtype=np.int)
            indexes = [tmp[i::n_threads] for i in range(n_threads)]

            # Can't stop the work...
            work = []
            for i in range(n_threads):
                c = copy.deepcopy(ctx)
                c.indexes = indexes[i]
                work.append(c)

            logging.info(f"  Launching the pool with {n_threads} threads")
            with multiprocessing.Pool(n_threads) as pool:
                retval = pool.map(StatsPropagator._propagate_frame, work)

            # Recombine the results
            for t in range(n_threads):
                for i in work[t].indexes:
                    data.As[:,i] = retval[t].As[:,i]
                    data.Ps[:,i] = retval[t].Ps[:,i]
                    data.Ts[:,i] = retval[t].Ts[:,i]
                    data.valid[:,i] = retval[t].valid[:,i]

            data.Ns = np.sum(ctx.valid, axis=1, dtype=np.int64)

        else:
            ctx.indexes = list(range(ctx.L))
            tmp = StatsPropagator._propagate_frame(ctx)
            data.Ns
            data.Ns = tmp.Ns
            data.As = tmp.As
            data.Ps = tmp.Ps
            data.Ts = tmp.Ts
            data.Vs = tmp.valid

    @classmethod
    def _propagate_frame(cls, ctx):

        # The beginning is a good place to begin
        t = ctx.start_ts

        fwd_prop_start = 0

        # Forward pass
        for i in range(ctx.N-1):
            print(f"  Propagating Frame:{i+1}/{ctx.N} Thread:{ctx.indexes[0]}")
            for j in ctx.indexes:
                frag = ctx.names[j]
                A = ctx.As[i][j]
                P = ctx.Ps[i][j]

                do_prop = (A and (not ctx.As[i+1][j] or ctx.prop_after_obs))

                if do_prop:

                    # Register the index of the first forward-predicted frame
                    if not fwd_prop_start: fwd_prop_start = i+1

                    # We have data now, but not in the future.  We
                    # should evaluate this frame for propagation.

                    # The perigee is already at or below the decay
                    # altitude, so we're going to drop it off the map
                    # now.
                    if P <= ctx.decay_alts[j]:
                        ctx.scope_end[frag] = t
                        ctx.valid[i+1][j] = 0
                        continue

                    # Find the indexes into the tables
                    idx_A, idx_P = ctx.decay.index_for(A, P)

                    dat = ctx.decay.mean

                    # Find the decay rates (dA/dt and dP/dt)
                    rate_A = dat[0][idx_A][idx_P]
                    rate_P = dat[1][idx_A][idx_P]
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
                    A = A + ctx.dt * rate_A
                    P = P + ctx.dt * rate_P

                    # Because the apogee decay right is higher we can
                    # sometimes nudge ourselves just over the line and
                    # invert the apogee/perigee.
                    if A >= P:
                        ctx.As[i+1][j] = A
                        ctx.Ps[i+1][j] = P
                    else:
                        ctx.As[i+1][j] = P
                        ctx.Ps[i+1][j] = A

                    ctx.Ts[i+1][j] = keplerian_period(ctx.As[i+1][j],
                                                      ctx.Ps[i+1][j])
                    ctx.valid[i+1][j] = 1

            t += ctx.dt

        # Update the number of valid values and preserve the original
        ctx.Ns_obs = ctx.Ns
        ctx.Ns = np.sum(ctx.valid, axis=1, dtype=np.int64)

        # Annotate the beginning of forward propagation
        ctx.fwd_prop_start = fwd_prop_start

        return ctx
