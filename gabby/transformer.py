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

|  #
|  #
|  #
| ###                         #
+-------------------------------

For the moment, we're still going to do uniform spacing of bins at
roughly 1km intervals which is assumed to be the materiality
threshold.  We're also going to clip outliers.

|      #
|     ###
|   #######
| ################
+------------------

"""


class MoralDecay(object):
    """Explores the depths of moral (and orbial) decay rates.

    decay_hist:
      [A=0,P=1][A-bin][P-bin][D-bin]

      Derivative bins are from D_min-D_max for the specific
      combination of A/P A and
    """

    def __init__(self, decay_hist, Ap, Ad, Pp, Pd):
        self.bins_A = np.linspace(Ad_min, Ad_max, n_D_bins)
        self.bins_P = np.linspace(Pd_min, Pd_max, n_D_bins)

    def _cdf():
        # CDF
        cdf = np.copy(moral_decay)
        for i in range(n_A):
            for j in range(n_P):
                cdf[0][i][j] = np.cumsum(cdf[0][i][j])
                cdf[1][i][j] = np.cumsum(cdf[1][i][j])

    def _percentiles():
        # Percentiles
        pct = np.copy(cdf)
        for i in range(n_A):
            for j in range(n_P):
                pct[0][i][j] = self._inverse(pct[0][i][j])
                pct[1][i][j] = self._inverse(pct[1][i][j])


class Jazz(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def _prior_des(self):
        # Find the prior ASATs to use for producing the histograms
        prior_des = self.cfg['historical-asats'].strip().split(',')
        prior_des = [s.strip() for s in prior_des]
        return prior_des

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
        """A basic low-pass filter for decay rates.

        Assumes samp_rate == 1 day
        """
        k = 10000
        n_taps = 127
        fltr = np.arange(-1*(n_taps//2), n_taps//2+1, 1) * np.pi / k
        fltr = (1/k) * np.sinc(fltr)
        fltr /= np.sum(fltr)
        return fltr

    # def derivatives(self,
    #                 priors=None,
    #                 dP=1,
    #                 min_life=1.0,
    #                 dt=SECONDS_IN_DAY,
    #                 fltr=None,
    #                 cache_dir=None):
    #     """Finds A'(P), and P'(P)
    #     """

    #     start_time = datetime.datetime.now()

    #     # Only need a read-only transaction for this
    #     txn = lmdb.Transaction(self.db_env, write=False)

    #     if priors: base_des = priors
    #     else: base_des = self._prior_des()
    #     fragments = find_daughter_fragments(base_des, txn, self.db_scope)

    #     L = len(fragments)

    #     logging.info(f"Finding derivatives for {L} fragments")

    #     # Load the APT values for all of the prior fragments
    #     to, Ao, Po, To, No = load_apt(fragments, txn, self.db_apt,
    #                                   cache_dir=cache_dir)
    #     logging.info(f"  Finished loading APT values")

    #     # (p)repared values
    #     tp = []
    #     Ap = []
    #     Pp = []
    #     Tp = []
    #     Np = np.zeros(L, dtype=np.int)

    #     for i in range(L):

    #         # (r)esampled
    #         Nr = No[i]
    #         tr, Ar = self.resample(to[i][:Nr], Ao[i][:Nr], dt)
    #         tr, Pr = self.resample(to[i][:Nr], Po[i][:Nr], dt)
    #         tr, Tr = self.resample(to[i][:Nr], To[i][:Nr], dt)
    #         Nr = len(tr)

    #         if fltr is not None and Nr > len(fltr):
    #             tr = tr[len(fltr)//2:-1*(len(fltr)//2)]
    #             Ar = np.convolve(Ar, fltr, mode='valid')
    #             Pr = np.convolve(Pr, fltr, mode='valid')
    #             Tr = np.convolve(Tr, fltr, mode='valid')
    #             Nr = len(tr)

    #         tp.append(tr)
    #         Ap.append(Ar)
    #         Pp.append(Pr)
    #         Tp.append(Tr)
    #         Np[i] = Nr

    #         if 0 == L%1000:
    #             logging.info("  Resampled and filtered {i} fragments")

    #     # Find the actual (d)erivatives and put all the filtered
    #     # values into a single numpy array
    #     N = np.where(Np > 0, Np-1, 0)
    #     ret_filtered = np.zeros((L, 4, np.max(Np)-1), dtype=np.float32)
    #     for i in range(L):
    #         ret_filtered[i][0][:N[i]] = tp[i][1:]
    #         ret_filtered[i][1][:N[i]] = Ap[i][1:]
    #         ret_filtered[i][2][:N[i]] = Pp[i][1:]
    #         ret_filtered[i][3][:N[i]] = Tp[i][1:]
    #     ret_deriv = np.zeros((L, 4, np.max(Np)-1), dtype=np.float32)
    #     for i in range(L):
    #         ret_deriv[i][0][:N[i]] = tp[i][1:]
    #         ret_deriv[i][1][:N[i]] = np.diff(Ap[i]) / dt
    #         ret_deriv[i][2][:N[i]] = np.diff(Pp[i]) / dt
    #         ret_deriv[i][3][:N[i]] = np.diff(Tp[i]) / dt

    #     end_time = datetime.datetime.now()

    #     elapsed = int((end_time-start_time).seconds * 10)/10.0
    #     logging.info(f"  Finished finding derivatives in {elapsed} seconds")

    #     retval = (fragments, ret_filtered, ret_deriv, N)
    #     return retval

    # def __concatenate(self, arr, N, prekeys=None, postkeys=None):
    #     """
    #     +-         + +-   -+
    #     |  1  2  3 | |  3  |
    #     |  4  0  0 | |  1  | => [1 2 3 4 7 8]
    #     |  7  8  0 | |  2  |
    #     +-        -+ +-   -+
    #     """

    #     print()
    #     print('='*60)
    #     pprint.pprint(N)
    #     print(arr.shape)
    #     print('='*60)

    #     L = len(arr)
    #     retval = np.zeros(np.sum(N), dtype=arr.dtype)
    #     j = 0
    #     for i in range(L):
    #         tmp = arr[i]
    #         for k in prekeys: tmp = tmp[k]
    #         retval[j:j+N[i]] = tmp[:N[i]]
    #         j += N[i]
    #     return retval

    def _percentile_values(self, arr, low_frac, high_frac):
        """Finds the lower and upper bounds after pruning the fraction.
        """

        N = len(arr)

        if low_frac:
            # Find the low-end value
            off = int(N * low_frac)
            tmp = np.partition(arr, off)
            low = tmp[off]
            del tmp
        else:
            low = np.min(arr)

        if high_frac:
            # Find the high-end value
            off = N - off - 1
            tmp = np.partition(arr, off)
            high = tmp[off]
            del tmp
        else:
            high = np.max(arr)

        return (low, high)

    def clip_to_flanks(self, arr, n_bins,
                       low_clip=None, high_clip=None,
                       min_val=None, max_val=None):
        """Clips values so that binning puts them in flanking bins.

        min/max values are bounded by the clip_frac as well as by the
        min/max values passed in.
        """

        # Find our actual clipping values
        clip_min, clip_max, = self._percentile_values(arr, low_clip, high_clip)

        # Apply any hard constraints
        if min_val is not None:
            clip_min = max(clip_min, min_val)
            clip_max = max(clip_max, min_val)
        if max_val is not None:
            clip_min = min(clip_min, max_val)
            clip_max = min(clip_max, max_val)

        step = (clip_max-clip_min)/(n_bins-1)

        # We're going to clip the values to ensure they fall into one
        # of the flanking bins.  That lets us use exactly the same
        # machinery later on during the binning process but without
        # including the clipped values in the actual results (the
        # inner bins).
        retval = np.clip(arr, clip_min-step*.9, clip_max+step*.9)

        return clip_min, clip_max, step, retval

    def _flanking_digitize(self, arr, min_val, step):
        """Digitizes the values to a bin with extremes in flanking bins

        Once you've clipped to flanks, this will indexes into bins.
        The lower flanking bin will be offset 0, and the peak offset
        will be N+1.  If done properly, then the real data is between
        1 and N.
        """
        assert(0 < step)
        tmp = (arr - min_val) / step
        tmp = np.round(tmp, decimals=0).astype(np.int) + 1
        return tmp

    def __concat_and_digitize(self, positions, derivatives, Ns, off, n_bins,
                              min_val=None, max_val=None,
                              low_clip=None, high_clip=None):

        print(positions.shape)
        sys.exit(0)
        p = self.__concatenate(positions, Ns, subkeys=[off])
        p_min, p_max, dp, p = self.clip_to_flanks(p, n_bins,
                                                  min_val=min_val,
                                                  max_val=max_val)
        p = self._flanking_digitize(p, p_min, dp)

        d = self.__concatenate(derivatives, Ns, subkeys=[off])
        d_min, d_max, dd, d = self.clip_to_flanks(Ad, n_D_bins,
                                                  low_clip=low_clip,
                                                  high_clip=high_clip)
        d = self._flanking_digitize(d, d_min, dd)

        return (p_min, p_max, dp, p,
                d_min, d_max, dd, d,)

    def __universalize(Ap, Ad, Pp, Pd, n_A_bins, n_P_bins, n_D_bins):
        logging.info("  Constructing a sorted universal key/value int64")
        # <bin-A><bin-P><derivative-bin>

        start = datetime.datetime.now().timestamp()
        bits_A = int(math.ceil(math.log(n_A_bins, 2)))+1
        bits_P = int(math.ceil(math.log(n_P_bins, 2)))+1
        bits_D = int(math.ceil(math.log(n_D_bins, 2)))+1

        shift_A = bits_P + bits_D
        shift_P = bits_D
        shift_D = 0
        mask_A = ((1<<bits_A)-1)<<shift_A
        mask_P = ((1<<bits_P)-1)<<shift_P
        mask_D = ((1<<bits_D)-1)<<shift_D

        univ = np.zeros(N, dtype=np.int64)
        univ |= (Ap << shift_A)
        univ |= (Pp << shift_P)
        univ_A = univ | (Ad << shift_D)
        univ_P = univ | (Pd << shift_D)

        end = datetime.datetime.now().timestamp()
        logging.info(f"    Universalizing took: {int((end-start)*1000)}ms")

        logging.info(f"  Sorting the universalized array")
        start = datetime.datetime.now().timestamp()
        univ_A.sort()
        univ_P.sort()
        end = datetime.datetime.now().timestamp()
        logging.info(f"    Sorting that took: {int((end-start)*1000)}ms")

        logging.info("  Indexing sorted universalized array")
        start = datetime.datetime.now().timestamp()
        index = np.zeros((2, n_A_bins+2, n_P_bins+2, n_D_bins+2), dtype=np.int)
        for i in range(n_A_bins+2):
            for j in range(n_P_bins+2):
                for k in range(n_D_bins+2):
                    srch = (i<<shift_A) | (j<<shift_P) | (k<<shift_D)
                    index[0][i][j][k] = np.searchsorted(univ_A, srch)
                    index[1][i][j][k] = np.searchsorted(univ_P, srch)
        end = datetime.datetime.now().timestamp()
        logging.info(f"    Indexing took: {int((end-start)*1000)}ms")

        return index, univ_A, univ_P

    def __bin_universalized_array(index, n_A_bins, n_P_bins, n_D_bins,):
        logging.info("  Binning dA/dP")
        start = datetime.datetime.now().timestamp()

        moral_decay = np.zeros((2, n_A_bins, n_P_bins, n_D_bins),
                               dtype=np.float32)

        for i in range(1, n_A_bins+1, 1):
            for j in range(1, n_P_bins+1, 1):
                tot_A = 0.0
                tot_P = 0.0
                for k in range(2, n_D_bins+2, 1):
                    cur = index[0][i][j][k] - index[0][i][j][k-1]
                    moral_decay[0][i-1][j-1][k-2] = cur
                    tot_A += cur
                    cur = index[1][i][j][k] - index[1][i][j][k-1]
                    moral_decay[1][i-1][j-1][k-2] = cur
                    tot_P += cur
                if tot_A: moral_decay[0][i-1][j-1] /= tot_A
                if tot_P: moral_decay[1][i-1][j-1] /= tot_P
        end = datetime.datetime.now().timestamp()
        logging.info(f"    Binning took: {int((end-start)*1000)}ms")

    def plot_mesh():
        Z = np.zeros((n_A_bins, n_P_bins), dtype=np.int)
        for i in range(n_A_bins):
            for j in range(n_P_bins):
                Z[i][j] = np.sum(moral_decay[0][i][j])

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        c = ax.pcolor(Z)
        fig.colorbar(c, ax=ax)
        fig.savefig(mesh_output)

    def decay_rates(self, positions, derivatives, Ns,
                    mesh_output=None):
        """Bins the decay rate distributions.


        (p)ositions:
        [
         [0] = tp: [frag-number][off] = time
         [1] = Ap: [frag-number][off] = apogee value
         [2] = Pp: [frag-number][off] = perigee value
        ]

        [3] = Np: [frag-number][off] = number of observations




        retval: [A'=0,P'=1][A-bin][P-bin][D-bin] = d(A/P)/dt

        dt is defined in the call to derivatives() defaulting to 1 day.



        derivatives: same as positions, but it's (d/dt)(A|P) and same time vals

        positions:   (L, N, 3)
        derivatives: (L, N-1, 3)

        Ns: [frag-number] = number of observations of that fragment
        """

        f = self.__concat_and_digitize

        Ap_min = self.cfg.getint('min-apogee')
        Ap_max = self.cfg.getint('max-apogee')
        Ad_low = self.cfg.getfloat('apogee-deriv-low-prune')
        Ad_high = self.cfg.getfloat('apogee-deriv-high-prune')
        n_A_bins = self.cfg.getint('n-apogee-bins')
        (Ap_min, Ap_max, Adp, Ap,
         Ad_min, Ad_max, Add, Ad,) = f(positions, derivatives, Ns, 1, n_A_bins,
                                       min_val=Ap_min, max_val=Ap_max,
                                       low_clip=Ad_low, high_clip=Ad_high)

        Pp_min = self.cfg.getint('min-perigee')
        Pp_max = self.cfg.getint('max-perigee')
        Pd_low = self.cfg.getfloat('perigee-deriv-low-prune')
        Pd_high = self.cfg.getfloat('perigee-deriv-high-prune')
        n_P_bins = self.cfg.getint('n-perigee-bins')
        (Pp_min, Pp_max, Pdp, Pp,
         Pd_min, Pd_max, Pdd, Pd,) = f(positions, derivatives, Ns, 2, n_P_bins,
                                       min_val=Pp_min, max_val=Pp_max,
                                       low_clip=Pd_low, high_clip=Pd_high)

        logging.info(f"Quantifying Moral Decay")

        __universalize(Ap, Ad, Pp, Pd, n_A_bins, n_P_bins, n_D_bins)

        __bin_universalized_array(index, n_A_bins, n_P_bins, n_D_bins,)

        # FIXME: Any normalization steps for things like B* compared
        # to mean would happen at this stage.

        if mesh_output: self.plot_mesh()

        return MoralDecay(moral_decay, Ap, Ad, Pp, Pd)
