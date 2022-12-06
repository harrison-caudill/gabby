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
from .moral_decay import MoralDecay

"""
"""


class Propagator(object):
    """Parent class for data propagators.
    """
    pass


class StatsPropagatorContext(object):

    def __init__(self, prop, data, fwd, rev,
                 prop_after_obs, decay_alt, drop_early,
                 incident_ts, prop_start_ts):

        self.fwd = fwd
        self.rev = rev
        self.prop_after_obs = prop_after_obs

        self.As = data.As
        self.Ps = data.Ps
        self.Ns = data.Ns
        self.ts = data.ts
        self.Ns_obs = None
        self.Ts = data.Ts
        self.Vs = data.Vs
        self.L = data.L
        self.N = data.N
        self.dt = data.dt.total_seconds()/24/3600.0
        self.start_ts = data.ts[0]
        self.names = data.fragments
        self.decay = prop.decay
        self.prop_start_ts = prop_start_ts
        self.Ds = np.zeros(self.Vs.shape, dtype=np.int8) # Decayed

        self.decay_alts = np.zeros(self.L) + decay_alt

        self.incident_idx = np.searchsorted(data.ts, incident_ts)
        if prop_start_ts is not None:
            self.prop_start_idx = np.searchsorted(data.ts, prop_start_ts)
        else:
            # Never hit this
            self.prop_start_idx = self.N + 1

        # If we're dropping them before they fully decay, then we'll
        # want to first find the altitude of the last observation.
        self.drop_early = drop_early
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

    def __init__(self, moral_decay):
        self.decay = moral_decay

    @classmethod
    def from_config(cls, cfg, db, cache=None):
        """Creates a StatsPropagator from the global config and DB.

        Just a thin wrapper around the transformer.
        """
        return StatsPropagator(Jazz.moral_decay_from_cfg(cfg, db, cache=cache))

    def propagate(self, data, incident_d,
                  drop_early=False,
                  fwd=True,
                  rev=True,
                  prop_start=None,
                  prop_after_obs=False,
                  n_threads=1,
                  decay_alt=200):
        """Propagates the <data> forward

        prop_after_obs: Means that we start propagating as soon as we
                        observe the fragment.  That way we can compare
                        the propagator to the observations.

        Global statistics necessary to perform the forward propagation
        are collected in the initialization phase.
        """

        logging.info(f"Propagating")

        incident_ts = dt_to_ts(incident_d)
        prop_start_ts = None
        if prop_start: prop_start_ts = dt_to_ts(prop_start)

        ctx = StatsPropagatorContext(self, data, fwd, rev, prop_after_obs,
                                     decay_alt, drop_early, incident_ts,
                                     prop_start_ts)

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
                retval = pool.map(StatsPropagator._propagate_fragment, work)

            # Recombine the results
            for t in range(n_threads):
                for i in work[t].indexes:
                    data.As[:,i] = retval[t].As[:,i]
                    data.Ps[:,i] = retval[t].Ps[:,i]
                    data.Ts[:,i] = retval[t].Ts[:,i]
                    data.Vs[:,i] = retval[t].Vs[:,i]

            data.Ns = np.sum(ctx.Vs, axis=1, dtype=np.int64)

        else:
            ctx.indexes = list(range(ctx.L))
            tmp = StatsPropagator._propagate_fragment(ctx)
            data.Ns
            data.Ns = tmp.Ns
            data.As = tmp.As
            data.Ps = tmp.Ps
            data.Ts = tmp.Ts
            data.Vs = tmp.Vs

    @classmethod
    def _propagate_fragment(cls, ctx):

        thread_n = ctx.indexes[0]

        if ctx.fwd:
            cls._fwd_propagate_fragment(ctx)

        if ctx.rev:
            cls._rev_propagate_fragment(ctx)

        # Update the number of valid values and preserve the original
        ctx.Ns_obs = ctx.Ns
        ctx.Ns = np.sum(ctx.Vs, axis=1, dtype=np.int64)

        return ctx


    @classmethod
    def _fwd_propagate_fragment(cls, ctx):


        thread_n = ctx.indexes[0]
        fwd_prop_start = None
        #decayed = np.zeros(ctx.As.shape[1], dtype=np.int8)
        for i in range(ctx.N-1):
            print(f"  Fwd Propagate:{i+1}/{ctx.N} Thread:{thread_n}")
            for j in ctx.indexes:
                A = ctx.As[i][j]
                P = ctx.Ps[i][j]

                # print(f"Trying: {i}, {j}")

                # print(f"P[{i}][{j}] = {P} ({ctx.Vs[i][j]})")

                # if P and P <= 200:
                #     print(f"DECAY???  P:{P} V:{ctx.Vs[i][j]}")

                do_prop = (ctx.Vs[i][j] # Do we have a valid starting point
                           and (not ctx.Vs[i+1][j]
                                or ctx.prop_after_obs
                                or i >= ctx.prop_start_idx))

                if do_prop:
                    # Register the index of the first forward-predicted frame
                    if fwd_prop_start is None: fwd_prop_start = i+1

                    # We have data now, but not in the future.  We
                    # should evaluate this frame for propagation.

                    # The perigee is already at or below the decay
                    # altitude, so we're going to drop it off the map
                    # now.
                    #print(f"WAAAAT {P} {ctx.decay_alts[j]}")
                    if P <= ctx.decay_alts[j]:
                        # print("DECAAAYYYYYYEEEED")
                        # print(f"  P:{P} V:{ctx.Vs[i][j]}")
                        # print(f"  Invalidating: [{i}:][{j}]")
                        ctx.Vs[i:,j] = 0
                        ctx.As[i:,j] = 0
                        ctx.Ps[i:,j] = 0
                        ctx.Ts[i:,j] = 0
                        #decayed[j] = 1
                        continue

                    # Ensure we trigger next frame
                    ctx.Vs[i+1][j] = 1

                    # Find the indexes into the tables
                    idx_A, idx_P = ctx.decay.index_for(A, P)

                    dat = ctx.decay.mean

                    # Find the decay rates (dA/dt and dP/dt)
                    rate_A = dat[0][idx_A][idx_P]
                    rate_P = dat[1][idx_A][idx_P]

                    #rate_A, rate_P = ctx.decay.rates(dat, A, P)

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
                    A += ctx.dt * rate_A
                    P += ctx.dt * rate_P

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

        # Annotate the beginning of forward propagation
        ctx.fwd_prop_start = fwd_prop_start
        #print(f"==============> {len(decayed)} - {np.sum(decayed)}")

    @classmethod
    def _rev_propagate_fragment(cls, ctx):

        thread_n = ctx.indexes[0]

        for i in range(ctx.N-1, ctx.incident_idx, -1):
            print(f"  Rev Propagate:{i+1}/{ctx.N} Thread:{ctx.indexes[0]}")
            for j in ctx.indexes:

                # Basic info about the frame
                frag = ctx.names[j]
                A = ctx.As[i][j]
                P = ctx.Ps[i][j]

                if not A: continue

                idx_A, idx_P = ctx.decay.index_for(A, P)
                rate_A = ctx.decay.mean[0][idx_A][idx_P]
                rate_P = ctx.decay.mean[1][idx_A][idx_P]
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

                if ctx.Vs[i][j] and not ctx.Vs[i-1][j]:

                    # Compute the new A/P values
                    A -= ctx.dt * rate_A
                    P -= ctx.dt * rate_P

                    # Because the apogee decay right is higher we can
                    # sometimes nudge ourselves just over the line and
                    # invert the apogee/perigee.
                    if A >= P:
                        ctx.As[i-1][j] = A
                        ctx.Ps[i-1][j] = P
                    else:
                        ctx.As[i-1][j] = P
                        ctx.Ps[i-1][j] = A

                    assert(0 < ctx.Ps[i-1][j] <= ctx.As[i-1][j])

                    ctx.Ts[i-1][j] = keplerian_period(ctx.As[i-1][j],
                                                      ctx.Ps[i-1][j])
                    ctx.Vs[i-1][j] = 1
