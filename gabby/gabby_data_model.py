import copy
import datetime
import gc
import json
import lmdb
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pickle
import subprocess
import struct
import sys
import tletools

from .model import DataModel
from .defs import *
from .utils import *
from .transformer import Jazz
from .cache import GabbyCache


class GabbyDataModel(DataModel):
    """Raw data employed by the Gabby Plotter.

    Data is contained within NxL Numpy Arrays of type float32.
    ts: timestamps of the frame
    As: Apogee
    Ps: Perigee
    Ts: Period
    Ns: Array of length N indicating how many fragments are valid in that frame
    Vs: NxL indication of which observations are valid
    """

    def __init__(self, fragments, ts, As, Ps, Ts, Ns, Vs, dt):
        self.fragments = fragments
        self.ts = ts
        self.As = As
        self.Ps = Ps
        self.Ts = Ts
        self.Ns = Ns
        self.Vs = Vs
        self.dt = dt

        self.start_d = ts_to_dt(ts[0])
        self.end_d = ts_to_dt(ts[-1])

        self.N = len(ts)
        self.L = len(As[0])

        # In case we're doing forward propagation, we'll need the
        # starting offset for graphing purposes.
        self.fwd_prop_start = None

    @classmethod
    def cache_name(cls, tgt):
        return f"gabby-model-{cfg_hash(tgt)}"

    @classmethod
    def from_cfg(cls, tgt, db):
        # The list of base satellites who's daughters are to be
        # tracked.  It has to be a list so that we can handle
        # collisions and intentional detonations.
        target_des = json.loads(tgt['intldes'])

        # Pull in the time boundaries from the config file
        start_d = parse_date_d(tgt['start-date'])
        end_d = parse_date_d(tgt['end-date'])

        # Time step between images
        dt = datetime.timedelta(days=tgt.getint('plot-period'))

        return cls.from_db(db=db, des=target_des,
                           start_d=start_d, end_d=end_d, dt_d=dt)

    @classmethod
    def from_db(cls,
                db=None,
                des=None,
                start_d=None, end_d=None, dt_d=None):
        """Loads the data from the database.
        """

        # Slightly redundant, but doing it this way ensures that the
        # order of the fragments is the same that everybody else will
        # be using.
        fragments = db.find_daughter_fragments(des)

        # CloudDescriptor with the APT values
        apt = db.load_apt(fragments)

        # Get our array dimensions
        L = len(fragments)
        N = int(math.ceil((end_d - start_d)/dt_d))+1

        # Get the integer timestamps to use for indexing
        start_ts = dt_to_ts(start_d)
        end_ts = dt_to_ts(end_d)

        # Time delta between gabby frames in seconds
        dt_s = int(dt_d.total_seconds())

        # Load scope
        #scope_start, scope_end = db.load_scope(des)

        # Linear array of timestamps, one for each frame
        srch_ts = np.arange(start_ts, end_ts+dt_s, dt_s, dtype=int)
        cmp_low_ts = srch_ts - 1
        cmp_high_ts = srch_ts + 1
        assert(N == len(srch_ts))

        # Convenience array in the same shape as the output timestamp
        # values allowing adding/subtracting.
        tmp = np.concatenate([srch_ts for i in range(L)]).reshape((L, N))
        tgt_ts = np.transpose(tmp)

        logging.info(f"  L: {L} fragments")
        logging.info(f"  N: {N} gabby frames")
        logging.info(f"  Start:    {start_d}")
        logging.info(f"  End:      {end_d}")
        logging.info(f"  ")


        # Before and after values for the timestamp and Apogee/Perigee
        before_ts = np.zeros((N, L), dtype=np.int32)
        after_ts = np.zeros((N, L), dtype=np.int32)

        before_A = np.zeros((N, L), dtype=np.float32)
        after_A = np.zeros((N, L), dtype=np.float32)

        before_P = np.zeros((N, L), dtype=np.float32)
        after_P = np.zeros((N, L), dtype=np.float32)

        before_T = np.zeros((N, L), dtype=np.float32)
        after_T = np.zeros((N, L), dtype=np.float32)

        valid = np.zeros((N, L), dtype=np.int8)

        bounds = np.zeros((L, 2), dtype=np.int32)

        # NOTE: Turns out that the Fengyun debris contains 3 negative
        # values for 99025CKQ, 99025MB, and 99025WC.  I should
        # probably pre-filter the DB to eliminate negative values.  In
        # the meantime, I can clip them.
        apt.A = np.clip(apt.A, 0, None)
        apt.P = np.clip(apt.P, 0, None)

        # The big main loop
        for frag_idx in range(L):
            frag = fragments[frag_idx]
            logging.info(f"    Finding straddling points for {frag} ({frag_idx+1}/{L})")

            # Number of observations for this fragment
            n_frag_obs = apt.N[frag_idx]

            # If we only have one observation, there's no point in
            # using it.
            if 2 > n_frag_obs: continue

            # tAPT observations for this fragment
            frag_t = apt.t[frag_idx][:n_frag_obs]
            frag_A = apt.A[frag_idx][:n_frag_obs]
            frag_P = apt.P[frag_idx][:n_frag_obs]
            frag_T = apt.T[frag_idx][:n_frag_obs]

            # There are 9 options for the first observation of
            # consequence A-I are the options, and R is the
            # gabby-frame Reference line.
            # A: +-------+
            # B:   +-------+
            # C:     +-------+
            # D:           +-------+
            # E:               +-------+
            # F:                   +-------+
            # G:                       +-------+
            # H:                           +-------+
            # I:                             +-------+
            # R:           +---+---+---+---+

            # If this fragment does not have any viable entries (A and
            # I), or has only a single point of intersection (B and
            # H), we can just skip it.
            if frag_t[-1] <= srch_ts[0]: continue # Option A/B
            if frag_t[0] >= srch_ts[-1]: continue # Option H/I

            # Now we only need consider the non-zero intersection
            # sets:
            # C:           +--+
            # D:           +-------+
            # E:               +-------+
            # F:                   +-------+
            # G:                       +---+
            # R:           +---+---+---+---+

            # That makes C/D and F/G equivalent.  Since the beginning
            # and ending are treated independently of one another, we
            # don't care about the case where the beginning/end of the
            # ranges are the same.

            # That brings us down to 3 cases:
            # D:           +-------+
            # E:               +-------+
            # F:                   +-------+
            # R:           +---+---+---+---+
            # X:   0000000001..............NNNNNNNNN

            # Since we're treating the beginning/ending independently
            # of one another, that means we have two cases for
            # starting and two for ending: on/after for both.

            # Select the offsets within the observations that match
            # the target timestamps for the gabby frames.  idx_after
            # is of length N (one entry per gabby frame) and holds the
            # offsets within the observations for the first
            # observation on/after the target timestamp.
            idx_before = np.searchsorted(frag_t, cmp_high_ts) - 1
            idx_eq = np.searchsorted(frag_t, srch_ts)

            a = np.searchsorted(idx_before, -1, side='right')
            b = np.searchsorted(idx_eq, n_frag_obs, side='left')
            bounds[frag_idx][0] = a
            bounds[frag_idx][1] = b

            for gabby_idx in range(a, b, 1):

                before = idx_before[gabby_idx]
                after = idx_eq[gabby_idx]

                assert(frag_t[before] <= srch_ts[gabby_idx])
                assert(frag_t[after] >= srch_ts[gabby_idx])
                assert(frag_t[before] != srch_ts[gabby_idx] or
                       before == after)

                before_ts[gabby_idx][frag_idx] = frag_t[before]
                after_ts[gabby_idx][frag_idx] = frag_t[after]
                before_A[gabby_idx][frag_idx] = frag_A[before]
                after_A[gabby_idx][frag_idx] = frag_A[after]
                before_P[gabby_idx][frag_idx] = frag_P[before]
                after_P[gabby_idx][frag_idx] = frag_P[after]
                before_T[gabby_idx][frag_idx] = frag_T[before]
                after_T[gabby_idx][frag_idx] = frag_T[after]

        logging.info(f"  Computing temporal offsets")
        dt = after_ts - before_ts
        pct_before = 1.0 - (tgt_ts - before_ts) / dt
        pct_after = 1.0 - (after_ts - tgt_ts) / dt
        pct_before = np.where(np.isnan(pct_before), 1, pct_before)
        pct_after = np.where(np.isnan(pct_before), 0, pct_after)

        # Remove nans and inf's
        pct_before = np.where(np.isnan(pct_before), 0, pct_before)
        pct_after = np.where(np.isnan(pct_after), 0, pct_after)
        pct_before = np.where(np.abs(pct_before) == math.inf, 0, pct_before)
        pct_after = np.where(np.abs(pct_after) == math.inf, 0, pct_after)

        assert(np.all(pct_before >= 0))
        assert(np.all(pct_after >= 0))
        assert(np.all(before_A >= 0))
        assert(np.all(after_A >= 0))
        assert(np.all(before_P >= 0))
        assert(np.all(after_P >= 0))


        logging.info(f"  Interpolating Values")
        A = np.zeros((N, L), dtype=np.float32)
        P = np.zeros((N, L), dtype=np.float32)
        T = np.zeros((N, L), dtype=np.float32)
        for frag_idx in range(L):
            a = bounds[frag_idx][0]
            b = bounds[frag_idx][1]
            valid[a:b,frag_idx] = 1

            before = before_A[a:b, frag_idx] * pct_before[a:b, frag_idx]
            after = after_A[a:b, frag_idx] * pct_after[a:b, frag_idx]
            A[a:b, frag_idx] = before + after

            before = before_P[a:b, frag_idx] * pct_before[a:b, frag_idx]
            after = after_P[a:b, frag_idx] * pct_after[a:b, frag_idx]
            P[a:b, frag_idx] = before + after

            before = before_T[a:b, frag_idx] * pct_before[a:b, frag_idx]
            after = after_T[a:b, frag_idx] * pct_after[a:b, frag_idx]
            T[a:b, frag_idx] = before + after

        t = srch_ts
        Ns = np.sum(valid, axis=1, dtype=np.int64)

        logging.info(f"  We win!")

        return GabbyDataModel(fragments, t, A, P, T, Ns, valid, dt_d)
