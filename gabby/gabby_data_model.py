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
import pprint
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
    """

    cache_name = 'gabby_data_model'

    def __init__(self, tgt, tgt_cache=None):

        self.tgt = tgt

        # The list of base satellites who's daughters are to be
        # tracked.  It has to be a list so that we can handle
        # collisions and intentional detonations.
        self.target_des = json.loads(self.tgt['intldes'])

        # Mask off the original rocket body, as it's a distraction
        self.mask = json.loads(self.tgt['mask']) if 'mask' in self.tgt else None

        # Pull in the time boundaries from the config file
        self.start_d = parse_date_d(self.tgt['start-date'])
        self.end_d = parse_date_d(self.tgt['end-date'])
        self.incident_d = parse_date_d(self.tgt['incident'])

        # Time step between images
        self.dt = datetime.timedelta(days=self.tgt.getint('plot-period'))

        # In case we're doing forward propagation, we'll need the
        # starting offset for graphing purposes.
        self.fwd_prop_start = None

    def fetch_from_db(self, db):
        """Loads the data from the database.
        """

        # First, find when pieces come into scope and when they go out
        self.scope_start, self.scope_end = db.load_scope(self.target_des)

        # Slightly redundant, but doing it this way ensures that the
        # order of the fragments is the same that everybody else will
        # be using.
        self.names = fragments = db.find_daughter_fragments(self.target_des)

        # CloudDescriptor with the APT values
        apt = self.apt = db.load_apt(fragments)

        # Get our array dimensions
        L = self.L = len(fragments)
        N = self.N = int(math.ceil((self.end_d - self.start_d)/self.dt))+1

        # For propagation purposes, it's nice to know the first and
        # last observation.  This will hold the index into the array
        # of the first and last entries.
        boundaries = self.boundaries = np.zeros((self.L, 2), dtype=int)

        # Get the integer timestamps to use for indexing
        self.start_ts = start_ts = dt_to_ts(self.start_d)
        self.incident_ts = incident_ts = dt_to_ts(self.incident_d)
        self.end_ts = end_ts = dt_to_ts(self.end_d)

        # Time delta between gabby frames in seconds
        dt_s = int(self.dt.total_seconds())

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
        logging.info(f"  Start:    {self.start_d}")
        logging.info(f"  Incident: {self.end_d}")
        logging.info(f"  End:      {self.end_d}")
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

        # The big main loop
        for frag_idx in range(L):
            frag = fragments[frag_idx]
            logging.info(f"    Fetching data for {frag} ({frag_idx+1}/{L})")

            # Number of observations for this fragment
            n_frag_obs = self.apt.N[frag_idx]

            # If we only have one observation, there's no point in
            # using it.
            if 2 > n_frag_obs: continue

            # tAPT observations for this fragment
            frag_t = self.apt.t[frag_idx][:n_frag_obs]
            frag_A = self.apt.A[frag_idx][:n_frag_obs]
            frag_P = self.apt.P[frag_idx][:n_frag_obs]
            frag_T = self.apt.T[frag_idx][:n_frag_obs]

            # print("==============================")
            # print(f"Fragment Timestamps: {frag_t}")
            # print(f"Search Timestamps:   {srch_ts}")

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

            # print(f"Fragment is in scope")

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

            # print(f"Index Before:        {idx_before}")
            # print(f"Index Equal:         {idx_eq}")
            # print(f"Diff:                {np.diff(idx_eq)}")

            a = np.searchsorted(idx_before, -1, side='right')
            b = np.searchsorted(idx_eq, n_frag_obs, side='left')
            bounds[frag_idx][0] = a
            bounds[frag_idx][1] = b
            # print(f"a:                   {a}")
            # print(f"b:                   {b}")

            for gabby_idx in range(a, b, 1):

                before = idx_before[gabby_idx]
                after = idx_eq[gabby_idx]
                # print(f"Range[{gabby_idx}]: {before} - {after}")

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

        logging.info(f"  Registering the resulting data")
        self.ts = t
        self.As = A
        self.Ps = P
        self.Ts = T
        self.Ns = Ns
        self.valid = valid


