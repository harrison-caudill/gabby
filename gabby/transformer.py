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

    def __init__(self, cfg, env, apt, tle, scope):
        self.cfg = cfg
        self.db_env = env
        self.db_apt = apt
        self.db_tle = tle
        self.db_scope = scope

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

                    apt_bytes = txn.get(fmt_key(start, frag), db=self.db_apt)
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


    def resample(self, X, Y, dx, sub='linear'):
        """Resamples the aperiodic signal to periodic sampling.

        The TLE observations are aperiodic, but most of the cool
        signal processing routines assume a regularly-sampled signal.
        """
        Xr = np.arange(X[0], X[-1], dx, dtype=X.dtype)

        if sub == 'cubic':
            f = scipy.interpolate.interp1d(X, Y, kind='cubic')
            Yr = f(Xr)
        else:
            Yr = np.interp(Xr, X, Y)

        return Xr, Yr

    def lpf(self):
        k = 10000
        n_taps = 127
        fltr = np.arange(-1*(n_taps//2), n_taps//2+1, 1) * np.pi / k
        fltr = (1/k) * np.sinc(fltr)
        fltr /= np.sum(fltr)
        return fltr

    def derivatives(self,
                    dP=1,
                    min_life=1.0,
                    dt=SECONDS_IN_DAY,
                    fltr=None,
                    cache_dir=None):
        """Finds A'(P), and P'(P)
        """

        if cache_dir:
            cache_data_path = os.path.join(cache_dir, "deriv_data.np")
            cache_meta_path = os.path.join(cache_dir, "deriv_meta.pickle")
            if os.path.exists(cache_data_path):
                with open(cache_data_path, 'rb') as fd:
                    logging.info(f"Loading derivatives from: {cache_data_path}")
                    pos = np.load(fd)
                    deriv = np.load(fd)
                    N = np.load(fd)
                with open(cache_meta_path, 'rb') as fd:
                    meta = pickle.load(fd)
                return meta['fragments'], pos, deriv, N

        start_time = datetime.datetime.now()

        # Only need a read-only transaction for this
        txn = lmdb.Transaction(self.db_env, write=False)

        base_des = self._prior_des()
        fragments = find_daughter_fragments(base_des, txn, self.db_scope)

        L = len(fragments)

        logging.info(f"Finding derivatives for {L} fragments")

        # Load the APT values for all of the prior fragments
        to, Ao, Po, To, No = load_apt(fragments, txn, self.db_apt,
                                      cache_dir=cache_dir)
        logging.info(f"  Finished loading APT values")

        # (p)repared values
        tp = []
        Ap = []
        Pp = []
        Tp = []
        Np = np.zeros(L, dtype=np.int)

        for i in range(L):

            # (r)esampled
            Nr = No[i]
            tr, Ar = self.resample(to[i][:Nr], Ao[i][:Nr], dt)
            tr, Pr = self.resample(to[i][:Nr], Po[i][:Nr], dt)
            tr, Tr = self.resample(to[i][:Nr], To[i][:Nr], dt)
            Nr = len(tr)

            if fltr is not None and Nr > len(fltr):
                tr = tr[len(fltr)//2:-1*(len(fltr)//2)]
                Ar = np.convolve(Ar, fltr, mode='valid')
                Pr = np.convolve(Pr, fltr, mode='valid')
                Tr = np.convolve(Tr, fltr, mode='valid')
                Nr = len(tr)

            tp.append(tr)
            Ap.append(Ar)
            Pp.append(Pr)
            Tp.append(Tr)
            Np[i] = Nr

            if 0 == L%1000:
                logging.info("  Resampled and filtered {i} fragments")

        # Find the actual (d)erivatives and put all the filtered
        # values into a single numpy array
        N = np.where(Np > 0, Np-1, 0)
        ret_filtered = np.zeros((L, 4, np.max(Np)-1), dtype=np.float32)
        for i in range(L):
            ret_filtered[i][0][:N[i]] = tp[i][1:]
            ret_filtered[i][1][:N[i]] = Ap[i][1:]
            ret_filtered[i][2][:N[i]] = Pp[i][1:]
            ret_filtered[i][3][:N[i]] = Tp[i][1:]
        ret_deriv = np.zeros((L, 4, np.max(Np)-1), dtype=np.float32)
        for i in range(L):
            ret_deriv[i][0][:N[i]] = tp[i][1:]
            ret_deriv[i][1][:N[i]] = np.diff(Ap[i]) / dt
            ret_deriv[i][2][:N[i]] = np.diff(Pp[i]) / dt
            ret_deriv[i][3][:N[i]] = np.diff(Tp[i]) / dt

        end_time = datetime.datetime.now()

        elapsed = int((end_time-start_time).seconds * 10)/10.0
        logging.info(f"  Finished finding derivatives in {elapsed} seconds")

        # Cache the values
        if cache_dir:
            cache_data_path = os.path.join(cache_dir, "deriv_data.np")
            cache_meta_path = os.path.join(cache_dir, "deriv_meta.pickle")
            with open(cache_data_path, 'wb') as fd:
                logging.info(f"Saving derivatives to: {cache_data_path}")
                np.save(fd, ret_filtered)
                np.save(fd, ret_deriv)
                np.save(fd, N)
            with open(cache_meta_path, 'wb') as fd:
                meta = {
                    'dP': dP,
                    'min_life': min_life,
                    'dt': dt,
                    'fltr': fltr,
                    'base': base_des,
                    'fragments': fragments,
                    }
                pickle.dump(meta, fd)

        retval = (fragments, ret_filtered, ret_deriv, N)
        return retval

    def decay_rates(self,
                    positions,
                    derivatives,
                    Ns,
                    ignore_frac=.01,
                    n_A_bins=100,
                    n_P_bins=100,
                    n_D_bins=100,
                    min_apogee=150,
                    max_apogee=1500,
                    min_perigee=100,
                    max_perigee=1000):

        N = np.sum(Ns)
        logging.info(f"Quantifying Moral Decay from {N} samples")

        # FIXME: Any normalization steps for things like B* compared
        # to mean would happen at this stage.

        # We don't care about fragment separation so just concatenate
        # everything
        L = len(positions)
        Ap = np.zeros(N, dtype=np.float32)
        Pp = np.zeros(N, dtype=np.float32)
        Ad = np.zeros(N, dtype=np.float32)
        Pd = np.zeros(N, dtype=np.float32)
        j = 0
        for i in range(L):
            Ap[j:j+Ns[i]] = positions[i][1][:Ns[i]]
            Pp[j:j+Ns[i]] = positions[i][2][:Ns[i]]
            Ad[j:j+Ns[i]] = derivatives[i][1][:Ns[i]]
            Pd[j:j+Ns[i]] = derivatives[i][2][:Ns[i]]
            j += Ns[i]

        logging.info("  Partitioning derivatives")
        start = datetime.datetime.now().timestamp()
        n_skip = int(ignore_frac * np.sum(N))
        Ad_part = np.partition(Ad, n_skip)
        Ad_min = Ad[n_skip]
        Ad_part = np.partition(Ad, N-n_skip)
        Ad_max = Ad[N-n_skip]

        Pd_part = np.partition(Pd, n_skip)
        Pd_min = Pd[n_skip]
        Pd_part = np.partition(Pd, N-n_skip)
        Pd_max = Pd[N-n_skip]
        end = datetime.datetime.now().timestamp()

        logging.info("  Clipping the derivative arrays")
        Ad_step = (Ad_max-Ad_min)/(n_D_bins-1) # n_bins, not n_steps
        Pd_min = Pd[n_skip]
        Pd_max = Pd[-1*n_skip]
        Pd_step = (Pd_max-Pd_min)/(n_D_bins-1) # n_bins, not n_steps
        # Place all ignored values into flanking bins
        Ad = np.clip(Ad, Ad_min-Ad_step, Ad_max+Ad_step)
        Pd = np.clip(Pd, Pd_min-Pd_step, Pd_max+Pd_step)

        logging.info("  Clipping the Apogee/Perigee arrays")
        A_min = min_apogee
        A_max = max_apogee
        P_min = min_perigee
        P_max = max_perigee
        P_step = (P_max-P_min)/(n_P_bins-1)
        A_step = (A_max-A_min)/(n_A_bins-1)
        Ap = np.clip(Ap, A_min-A_step*.9, A_max+A_step*.9)
        Pp = np.clip(Pp, P_min-P_step*.9, P_max+P_step*.9)

        logging.info("  Discretizing the bin numbers")
        Ap = (Ap - min_apogee) / A_step
        Ad = (Ad - Ad_min) / Ad_step
        Pp = (Pp - min_perigee) / P_step
        Pd = (Pd - Pd_min) / Pd_step

        # We add one so that they are valid indices into an array
        Ap = np.round(Ap, decimals=0).astype(np.int) + 1
        Ad = np.round(Ad, decimals=0).astype(np.int) + 1
        Pp = np.round(Pp, decimals=0).astype(np.int) + 1
        Pd = np.round(Pd, decimals=0).astype(np.int) + 1

        logging.info("  Constructing a sorted universal key/value int64")
        # <bin-A><bin-P><derivative-bin>

        start = datetime.datetime.now().timestamp()
        bits_A = int(math.ceil(math.log(n_A_bins, 2)))
        bits_P = int(math.ceil(math.log(n_P_bins, 2)))
        bits_D = int(math.ceil(math.log(n_D_bins, 2)))
        univ = np.zeros(N, dtype=np.int64)
        univ += Ap
        univ *= 2**bits_P
        univ += Pp
        univ *= 2**bits_D
        univ_A = univ + Ad
        univ_P = univ + Pd

        end = datetime.datetime.now().timestamp()
        logging.info(f"    Universalizing took: {int((end-start)*1000)}ms")

        start = datetime.datetime.now().timestamp()
        np.sort(univ_A)
        np.sort(univ_P)
        end = datetime.datetime.now().timestamp()
        logging.info(f"    Sorting that took: {int((end-start)*1000)}ms")

        logging.info("  Indexing")
        start = datetime.datetime.now().timestamp()
        index = np.zeros((2, n_A_bins+2, n_P_bins+2, n_D_bins+2), dtype=np.int)
        for i in range(n_A_bins+2):
            for j in range(n_P_bins+2):
                for k in range(n_D_bins+2):
                    srch = (i)<<(bits_P+bits_D) | (j)<<bits_D | k
                    index[0][i][j][k] = np.searchsorted(univ_A, srch)
                    index[1][i][j][k] = np.searchsorted(univ_P, srch)
        end = datetime.datetime.now().timestamp()
        logging.info(f"    Indexing took: {int((end-start)*1000)}ms")

        logging.info("  Binning")
        start = datetime.datetime.now().timestamp()

        retval = np.zeros((2, n_A_bins, n_P_bins, n_D_bins),
                       dtype=np.float32)

        for i in range(1, n_A_bins+1, 1):
            for j in range(1, n_P_bins+1, 1):
                tot_A = 0.0
                tot_P = 0.0
                for k in range(2, n_D_bins+2, 1):
                    cur = index[0][i][j][k] - index[0][i][j][k-1]
                    retval[0][i-1][j-1][k-2] = cur
                    tot_A += cur
                    cur = index[1][i][j][k] - index[1][i][j][k-1]
                    retval[1][i-1][j-1][k-2] = cur
                    tot_P += cur
                if tot_A: retval[0][i-1][j-1] /= tot_A
                if tot_P: retval[1][i-1][j-1] /= tot_P
        end = datetime.datetime.now().timestamp()
        logging.info(f"    Binning took: {int((end-start)*1000)}ms")

        bins_A = np.linspace(Ad_min, Ad_max, n_D_bins)
        bins_P = np.linspace(Pd_min, Pd_max, n_D_bins)
        return retval, bins_A, bins_P
