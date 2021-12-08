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
import scipy.signal

from .defs import *
from .utils import *

"""

=== A note about binning ===

If we bin evenly across the entire range, then we run the risk of
allowing outliers (e.g. below) to force the bulk of the observations
to be crowded:

 #
 #
 #
 #
 #
###                                           #

For the moment, we're still going to do uniform spacing of bins at
roughly 1km intervals which is assumed to be the materiality threshold.

"""


class Jazz(object):

    def __init__(self,
                 cfg,
                 db_path,
                 db_env=None):

        # Preserve the basic inputs
        self.db_path = db_path
        self.cfg = cfg

        # Build the DB handles
        if db_env: self.db_env = db_env
        else: self.db_env = lmdb.Environment(self.db_path,
                                             max_dbs=len(DB_NAMES),
                                             map_size=int(DB_MAX_LEN))
        self.db_gabby = self.db_env.open_db(DB_NAME_GABBY.encode(),
                                            dupsort=False)
        self.db_idx = self.db_env.open_db(DB_NAME_IDX.encode(),
                                          dupsort=False)
        self.db_scope = self.db_env.open_db(DB_NAME_SCOPE.encode(),
                                            dupsort=False)

    def _prior_des(self):
        # Find the prior ASATs to use for producing the histograms
        prior_des = self.cfg['historical-asats'].strip().split(',')
        prior_des = [s.strip() for s in prior_des]
        return prior_des

    def lifetime_stats(self,
                       cutoff,
                       n_apogee=5,
                       n_perigee=5,
                       n_time=5,
                       weight_bins=True,
                       min_life=1.0):
        """Finds historical lifetime stats as a function of A/P.

        Examines all previous debris fragments associated with
        in-space collisions (ASAT and natural alike) and finds the
        probability histogram associated with decay time as a function
        of both apogee and perigee.  It stops at the cutoff date.

        Returns: hist

        n_apogee: number of apogee bins in the histogram
        n_perigee: number of perigee bins in the histogram
        n_time: number of time bins in the histogram
        weight_bins: use uniformly-weighted bins for apogee and perigee
        min_life: only consider fragments with a minimum lifetime (days)
        """

        prior_des = self._prior_des()

        # Only need a read-only transaction for this
        txn = lmdb.Transaction(self.db_env, write=False)
        try:

            start_reg = {}
            end_reg = {}
            a_reg = {}
            p_reg = {}
            t_reg = {}

            cursor = txn.cursor(db=self.db_scope)

            cutoff_ts = cutoff.timestamp()

            # Load all the values into memory
            for satellite in prior_des:
                cursor.set_range(satellite.encode())
                for k, v in cursor:
                    frag = k.decode()
                    if satellite not in frag: break

                    start, end, = struct.unpack('ii', v)

                    # We only consider fully-decayed fragments
                    if end > cutoff_ts: continue

                    life = end - start

                    # We ignore single-observation fragments
                    if life < min_life*24*3600: continue

                    start_reg[frag] = start
                    end_reg[frag] = end

                    apt_bytes = txn.get(fmt_key(start, frag), db=self.db_gabby)
                    a, p, t, = struct.unpack('fff', apt_bytes)

                    a_reg[frag] = a
                    p_reg[frag] = p
                    t_reg[frag] = life

            # Relinquish our handle and clear our pointer
            txn.commit()
            txn = None

            N = len(a_reg)
            assert(len(start_reg) == len(end_reg) == len(p_reg) == len(t_reg))
            logging.info(f"  Found {N} compliant fragments")

            # Find our ranges
            min_a = min([a_reg[k] for k in a_reg])
            max_a = max([a_reg[k] for k in a_reg])
            min_p = min([p_reg[k] for k in p_reg])
            max_p = max([p_reg[k] for k in p_reg])
            min_t = min([t_reg[k] for k in t_reg])
            max_t = max([t_reg[k] for k in t_reg])

            # Define our bins
            if weight_bins:
                A = sorted([a_reg[x] for x in a_reg])
                P = sorted([p_reg[x] for x in p_reg])
                T = sorted([t_reg[x] for x in t_reg])

                a_bins = np.zeros(n_apogee)
                p_bins = np.zeros(n_perigee)
                t_bins = np.zeros(n_time)

                a_idx = [int(x) for x in np.linspace(0, N, n_apogee+1)][:-1]
                p_idx = [int(x) for x in np.linspace(0, N, n_perigee+1)][:-1]
                t_idx = [int(x) for x in np.linspace(0, N, n_time+1)][:-1]

                for i in range(n_apogee): a_bins[i] = A[a_idx[i]]
                for i in range(n_perigee): p_bins[i] = P[p_idx[i]]
                for i in range(n_time): t_bins[i] = T[t_idx[i]]

            else:
                a_bins = np.linspace(min_a, max_a, n_apogee)
                p_bins = np.linspace(min_p, max_p, n_perigee)
                t_bins = np.linspace(min_t, max_t, n_time)

            logging.info(f"  Apogee:   {int(min_a)}-{int(max_a)}")
            bins_s = str([int(x) for x in a_bins])
            logging.info(f"  {bins_s}\n")
            logging.info(f"  Perigee:  {int(min_p)}-{int(max_p)}")
            bins_s = str([int(x) for x in p_bins])
            logging.info(f"  {bins_s}\n")
            logging.info(f"  Lifetime: {int(min_t)}-{int(max_t)}")
            bins_s = str([datetime.datetime.utcfromtimestamp(x)
                          for x in t_bins])

            # Assign counts to the bins
            decay_hist = np.zeros((n_apogee, n_perigee, n_time,))
            for frag in a_reg:
                a = a_reg[frag]
                p = p_reg[frag]
                t = t_reg[frag]

                i = np.digitize(a, a_bins)-1
                j = np.digitize(p, p_bins)-1
                k = np.digitize(t, t_bins)-1

                decay_hist[i][j][k] += 1

            # Normalize the distribution
            for i in range(n_apogee):
                for j in range(n_perigee):
                    s = sum(decay_hist[i][j])
                    if not s: continue
                    decay_hist[i][j] /= s

            # for i in range(n_apogee):
            #     a_start = a_bins[i]
            #     logging.info(f"Decay Histogram (Apogee: {int(a_bins[i])})")
            #     logging.info(f"  Total: {sum(decay_hist[i])}")
            #     logging.info(f"  Total: {sum(sum(decay_hist[i]))}")
            #     bin_s = pprint.pformat(decay_hist[i]).replace('\n', '\n  ')
            #     logging.info(f"  {bin_s}")

            return decay_hist

        finally:
            if txn: txn.commit()

    def derivatives(self,
                    step_A=1, step_P=1, step_T=1,
                    step_dA=.1, step_dP=.1, step_dT=.001,
                    min_life=1.0):
        """Finds the derivatives of A, P, and T.

        step_A: km
        step_P: km
        step_T: minutes
        min_life: only consider fragments with a minimum lifetime (days)
        """

        logging.info(f"Finding derivatives")

        prior_des = self._prior_des()

        logging.info(f"  Listing fragments and their scopes")

        # Only need a read-only transaction for this
        txn = lmdb.Transaction(self.db_env, write=True)
        try:

            # designator => <int> timestamp when it comes into scope
            start_reg = {}

            # designator => <int> timestamp when it goes out of scope
            end_reg = {}

            # designator => <float> value when it comes into scope (init val)
            A_reg = {}
            P_reg = {}
            T_reg = {}

            # Cursor to walk the scope DB
            scope_cur = txn.cursor(db=self.db_scope)

            # Find the scope of all fragments of concern
            for satellite in prior_des:
                scope_cur.set_range(satellite.encode())
                for k, v in scope_cur:
                    frag = k.decode()
                    if satellite not in frag: break

                    start, end, = struct.unpack('ii', v)

                    life = end - start

                    # We ignore single-observation fragments
                    if life < min_life*24*3600: continue

                    start_reg[frag] = start
                    end_reg[frag] = end

                    apt_bytes = txn.get(fmt_key(start, frag), db=self.db_gabby)
                    A, P, T, = struct.unpack('fff', apt_bytes)

                    A_reg[frag] = A
                    P_reg[frag] = P
                    T_reg[frag] = T

            # Number of fragments we're looking at
            Nf = len(A_reg)
            assert(len(start_reg) == len(end_reg) == len(P_reg) == len(T_reg))
            logging.info(f"  Found {Nf} compliant fragments")

            # The primary reason we need the apogee and perigee is to
            # determine the effective range of our histogram.  We
            # should probably NOT do it this way, because data quality
            # issues can get some pretty whacked-out values which
            # could lead to the bins being too tightly packed unless
            # we do equally-weighted bins (in which case we have
            # another problem).
            min_A = min([A_reg[k] for k in A_reg])
            max_A = max([A_reg[k] for k in A_reg])
            min_P = min([P_reg[k] for k in P_reg])
            max_P = max([P_reg[k] for k in P_reg])
            min_T = min([T_reg[k] for k in T_reg])
            max_T = max([T_reg[k] for k in T_reg])

            # Compute the entire set of derivatives.  This set will be
            # quite large (on the order of 100m entries).
            logging.info(f"  Computing derivatives")
            index_cur = txn.cursor(db=self.db_idx)
            gabby_cur = txn.cursor(db=self.db_gabby)
            derivatives = {}

            tmp_t = np.zeros(40*365*2, dtype=np.int32)
            tmp_A = np.zeros(40*365*2, dtype=np.float32)
            tmp_P = np.zeros(40*365*2, dtype=np.float32)
            tmp_T = np.zeros(40*365*2, dtype=np.float32)

            # Here we go
            N = 0
            for frag in A_reg:
                logging.info(f"  Investigating fragment: {frag}")
                index_cur.set_range(frag.encode())
                off = len(frag)+1
                idx = 0
                des = (frag+',').encode()
                
                for k, v in index_cur:
                    if not k.startswith(des): break
                    cur_t = int(k[off:])
                    cur_A, cur_P, cur_T, = struct.unpack('fff', v)

                    tmp_t[idx] = cur_t
                    tmp_A[idx] = cur_A
                    tmp_P[idx] = cur_P
                    tmp_T[idx] = cur_T

                    idx += 1
                # Done with this fragment

                # Use numpy to parallelize the computation of the
                # differentials
                t = tmp_t[:idx]
                A = tmp_A[:idx]
                P = tmp_P[:idx]
                T = tmp_T[:idx]

                dt = np.diff(t)
                dA = np.diff(A)
                dP = np.diff(P)
                dT = np.diff(T)

                A = (A[:-1] + A[1:])/2
                P = (P[:-1] + P[1:])/2
                T = (T[:-1] + T[1:])/2

                dAdt = dA/dt
                dPdt = dP/dt
                dTdt = dT/dt
                derivatives[frag] = {
                    't': t,
                    'A': A,
                    'P': P,
                    'T': T,
                    'dt': dt,
                    'dA': dA,
                    'dP': dP,
                    'dT': dT,
                    'dAdt': dAdt,
                    'dPdt': dPdt,
                    'dTdt': dTdt,
                    }
                N += idx
            # Done with the big loop

            # Find the min and max values
            max_dAdt = min_dAdt = derivatives[frag]['dAdt'][0]
            max_dPdt = min_dPdt = derivatives[frag]['dPdt'][0]
            max_dTdt = min_dTdt = derivatives[frag]['dTdt'][0]
            max_A = min_A = derivatives[frag]['A'][0]
            max_P = min_P = derivatives[frag]['P'][0]
            max_T = min_T = derivatives[frag]['T'][0]
            for frag in derivatives:
                max_dAdt = min(max(max_dAdt,
                                   np.max(derivatives[frag]['dAdt'])),
                               0)

                min_dAdt = min(min_dAdt, np.min(derivatives[frag]['dAdt']))

                max_dPdt = min(max(max_dPdt,
                                   np.max(derivatives[frag]['dPdt'])),
                               0)
                min_dPdt = min(min_dPdt, np.min(derivatives[frag]['dPdt']))

                max_dTdt = min(max(max_dTdt,
                                   np.max(derivatives[frag]['dTdt'])),
                               0)
                min_dTdt = min(min_dTdt, np.min(derivatives[frag]['dTdt']))

                max_A = min(max(max_A, np.max(derivatives[frag]['A'])), 0)
                min_A = min(min_A, np.min(derivatives[frag]['A']))

                max_P = min(max(max_P, np.max(derivatives[frag]['P'])), 0)
                min_P = min(min_P, np.min(derivatives[frag]['P']))

                max_T = min(max(max_T, np.max(derivatives[frag]['T'])), 0)
                min_T = min(min_T, np.min(derivatives[frag]['T']))

            logging.info(f"  Total number of observations {N}")
            logging.info(f"  max_dAdt: {max_dAdt}")
            logging.info(f"  min_dAdt: {min_dAdt}")
            logging.info(f"  max_dPdt: {max_dPdt}")
            logging.info(f"  min_dPdt: {min_dPdt}")
            logging.info(f"  max_dTdt: {max_dTdt}")
            logging.info(f"  min_dTdt: {min_dTdt}")

            logging.info(f"  Collating the derivatives in their bins")
            nA = int(math.ceil(abs(max_A-min_A)/step_A))
            nP = int(math.ceil(abs(max_P-min_P)/step_P))
            nT = int(math.ceil(abs(max_T-min_T)/step_T))
            ndA = int(math.ceil(abs(max_dAdt-min_dAdt)/step_dA))
            ndP = int(math.ceil(abs(max_dPdt-min_dPdt)/step_dP))
            ndT = int(math.ceil(abs(max_dTdt-min_dTdt)/step_dT))
            
            ret_A = np.zeros((nA, nP, ndA), dtype=np.int32)
            ret_P = np.zeros((nA, nP, ndP), dtype=np.int32)
            ret_T = np.zeros((nA, nP, ndT), dtype=np.int32)

            for frag in derivatives:
                cur = derivatives[frag]
                off_A = cur['A'] - min_A
                off_P = cur['P'] - min_P
                off_T = cur['T'] - min_T
                off_dA = cur['dA'] - min_dAdt
                off_dP = cur['dP'] - min_dPdt
                off_dT = cur['dT'] - min_dTdt
                for i in range(0, len(cur['dA'])):

                    idx_A = min(max(off_A[i]/step_A, 0), nA-1)
                    idx_P = min(max(off_P[i]/step_P, 0), nP-1)
                    idx_T = min(max(off_T[i]/step_T, 0), nT-1)

                    idx_dA = min(max(off_dA[i]/step_dA, 0), nA-1)
                    idx_dP = min(max(off_dP[i]/step_dP, 0), nP-1)
                    idx_dT = min(max(off_dT[i]/step_dT, 0), nT-1)

                    ret_A[idx_A][idx_P][idx_dA] += 1
                    ret_P[idx_P][idx_P][idx_dP] += 1
                    ret_T[idx_T][idx_P][idx_dT] += 1

                break
            
            return derivatives, ret_A, ret_P, ret_T

        finally:
            if txn: txn.commit()
