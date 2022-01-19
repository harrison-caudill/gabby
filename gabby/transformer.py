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
from .db import CloudDescriptor

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

    def __init__(self, decay_hist, resampled, derivatives,
                 Ap_min, Ap_max, dAp, Ad_min, Ad_max, dAd,
                 Pp_min, Pp_max, dPp, Pd_min, Pd_max, dPd):

        self.decay_hist = decay_hist
        self.resampled = resampled
        self.derivatives = derivatives

        self.Ap_min = Ap_min
        self.Ap_max = Ap_max
        self.dAp = dAp
        self.Ad_min = Ad_min
        self.Ad_max = Ad_max
        self.dAd = dAd

        self.Pp_min = Pp_min
        self.Pp_max = Pp_max
        self.dPp = dPp
        self.Pd_min = Pd_min
        self.Pd_max = Pd_max
        self.dPd = dPd

        self.n_A_bins = len(decay_hist[0])
        self.n_P_bins = len(decay_hist[0][0])
        self.n_D_bins = len(decay_hist[0][0][0])
        self.bins_A = np.linspace(Ad_min, Ad_max, self.n_D_bins)
        self.bins_P = np.linspace(Pd_min, Pd_max, self.n_D_bins)

        self.cdf = self._cdf()
        self.percentiles = self._percentiles()

    def plot_mesh(self, path):
        Z = np.zeros((self.n_A_bins, self.n_P_bins), dtype=np.int)
        for i in range(self.n_A_bins):
            for j in range(self.n_P_bins):
                cur = np.sum(self.decay_hist[1][i][j])
                Z[i][j] = cur

        Z = np.where(Z > 0, 1, 0)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        c = ax.pcolor(Z)
        fig.colorbar(c, ax=ax)
        fig.savefig(path)

    def plot_dA_vs_P(self, path):
        #Z = np.sum(self.decay_hist[0], axis=0) * self.dPd + self.Pd_min

        for i in range(self.n_A_bins):
            fpath = path % {'i':i}
            logging.info(f"  Generating AVP image: {fpath}")
            Z = np.transpose(self.decay_hist[0][i]) * self.dPd + self.Pd_min
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.boxplot(Z)
            fig.savefig(fpath)
            fig.clf()
            plt.close(fig)
            gc.collect()

    def _cdf(self):
        cdf = np.copy(self.decay_hist)
        for i in range(self.n_A_bins):
            for j in range(self.n_P_bins):
                cdf[0][i][j] = np.cumsum(cdf[0][i][j])
                cdf[1][i][j] = np.cumsum(cdf[1][i][j])
        return cdf

    def _percentiles(self):
        pct = np.copy(self.cdf)
        for i in range(2):
            for An in range(self.n_A_bins):
                for Pn in range(self.n_P_bins):
                    last = 0
                    for Dn in range(self.n_D_bins-1):
                        srch = float(Dn)/self.n_D_bins
                        assert(1 >= srch)
                        assert(1+1e-3 >= self.cdf[i][An][Pn][Dn])
                        val = np.searchsorted(self.cdf[i][An][Pn], srch)
                        val = min(val, self.n_D_bins-1)
                        assert(val >= last)
                        last = val
                        pct[i][An][Pn][Dn] = val
                    pct[i][An][Pn][self.n_D_bins-1] = self.n_D_bins-1
        return pct

class Jazz(object):

    def __init__(self, cfg, global_cache=None, tgt_cache=None):
        self.cfg = cfg
        self.global_cache = global_cache
        self.tgt_cache = tgt_cache

    def resample(self, X, Y, dx, sub='linear'):
        """Resamples the aperiodic signal to periodic sampling.

        The TLE observations are aperiodic, but most of the cool
        signal processing routines assume a regularly-sampled signal.
        """
        Xr = np.arange(X[0], X[-1]+dx, dx, dtype=X.dtype)

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

    def filtered_derivatives(self, apt,
                             min_life=1.0,
                             dt=SECONDS_IN_DAY,
                             fltr=None):
        """Finds A'(P), and P'(P) potentially filtering along the way.

        This will do a resample to match dt also.
        """

        start_time = datetime.datetime.now()

        logging.info(f"Finding derivatives for {apt.L} fragments")

        # (f)iltered values
        tf = []
        Af = []
        Pf = []
        Tf = []
        Nf = np.zeros(apt.L, dtype=np.int)

        # Resample and Filter
        for i in range(apt.L):
            # (r)esampled
            Nr = apt.N[i]
            tr, Ar = self.resample(apt.t[i][:Nr], apt.A[i][:Nr], dt)
            tr, Pr = self.resample(apt.t[i][:Nr], apt.P[i][:Nr], dt)
            tr, Tr = self.resample(apt.t[i][:Nr], apt.T[i][:Nr], dt)
            Nr = tr.shape[0]

            if fltr is not None and Nr > len(fltr):
                # Copy here so we don't have to deal with numpy references
                tr = np.copy(tr[len(fltr)//2:-1*(len(fltr)//2)])
                Ar = np.convolve(Ar, fltr, mode='valid')
                Pr = np.convolve(Pr, fltr, mode='valid')
                Tr = np.convolve(Tr, fltr, mode='valid')
                Nr = tr.shape[0]

            # be sure to decrement the numpy refcount
            tf.append(tr); tr = None
            Af.append(Ar); Ar = None
            Pf.append(Pr); Pr = None
            Tf.append(Tr); Tr = None
            Nf[i] = Nr; Nr = None

            if i and 0 == i%1000:
                logging.info(f"  Resampled and filtered {i} fragments")

        M = max(Nf)
        for i in range(apt.L):
            tf[i].resize(M)
            Af[i].resize(M)
            Pf[i].resize(M)
            Tf[i].resize(M)

        tf = np.concatenate(tf).reshape((apt.L, M))
        Af = np.concatenate(Af).reshape((apt.L, M))
        Pf = np.concatenate(Pf).reshape((apt.L, M))
        Tf = np.concatenate(Tf).reshape((apt.L, M))
        Nf = Nf # It's fine as it is

        filtered = CloudDescriptor(fragments=apt.fragments,
                                   t=tf[:,1:],
                                   A=Af[:,1:],
                                   P=Pf[:,1:],
                                   T=Tf[:,1:],
                                   N=np.clip(Nf-1, 0, None))

        # (d)erivative values
        td = tf[:,1:]
        Ad = SECONDS_IN_DAY * np.diff(Af) / dt
        Pd = SECONDS_IN_DAY * np.diff(Pf) / dt
        Td = SECONDS_IN_DAY * np.diff(Tf) / dt
        Nd = np.clip(Nf-1, 0, None)

        deriv = CloudDescriptor(fragments=apt.fragments,
                                t=td, A=Ad, P=Pd, T=Td, N=Nd)

        assert(np.all(filtered.N == deriv.N))
        assert(np.all(filtered.t == deriv.t))

        end_time = datetime.datetime.now()

        elapsed = int((end_time-start_time).seconds * 10)/10.0
        logging.info(f"  Finished finding derivatives in {elapsed} seconds")

        return filtered, deriv

    def _concatenate(self, arr, N):
        """
        +-         + +-   -+
        |  1  2  3 | |  3  |
        |  4  0  0 | |  1  | => [1 2 3 4 7 8]
        |  7  8  0 | |  2  |
        +-        -+ +-   -+
        """

        L = len(arr)
        retval = np.zeros(np.sum(N), dtype=arr.dtype)
        j = 0
        for i in range(L):
            tmp = arr[i]
            retval[j:j+N[i]] = tmp[:N[i]]
            j += N[i]
        return retval

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

    def __concat_and_digitize(self, pos, deriv,
                              min_val=None,
                              max_val=None,
                              low_clip=None,
                              high_clip=None,
                              n_p_bins=None,
                              n_D_bins=None,
                              key=None):
        """Convenience function for decay_rates.

        pos: CloudDescriptor
        deriv: CloudDescriptor
        key: <str>, either 'A' or 'P' to select apogee or perigee

        It's a lot of repeated drudgery so we combine it here.
        """


        # Clipping the arrays necessitates having access to a single
        # concatenated array.
        p = self._concatenate(getattr(pos, key), pos.N)
        p_min, p_max, dp, p = self.clip_to_flanks(p, n_p_bins,
                                                  min_val=min_val,
                                                  max_val=max_val)
        p = self._flanking_digitize(p, p_min, dp)

        d = self._concatenate(getattr(deriv, key), deriv.N)
        d_min, d_max, dd, d = self.clip_to_flanks(d, n_D_bins,
                                                  low_clip=low_clip,
                                                  high_clip=high_clip)
        d = self._flanking_digitize(d, d_min, dd)

        return (p_min, p_max, dp, p,
                d_min, d_max, dd, d,)

    def __universalize(self, Ap, Ad, Pp, Pd, n_A_bins, n_P_bins, n_D_bins):
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

        N = len(Ap)
        assert(N == len(Ap) == len(Ad) == len(Pp) == len(Pd))

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

    def __bin_universalized_array(self, index, n_A_bins, n_P_bins, n_D_bins):
        """

        index: [0=dA/dt][A][P][dx/dt]
        The index holds the offsets into the universalized and sorted arrays.
        """
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

        return moral_decay

    def decay_rates(self, apt, resampled, deriv):
        """Bins the decay rate distributions.

        retval: [A'=0,P'=1][A-bin][P-bin][D-bin] = d(A/P)/dt

        dt is defined in the call to derivatives() defaulting to 1 day.
        """

        assert(np.all(resampled.N == deriv.N))
        assert(np.all(resampled.t == deriv.t))

        sec = self.cfg['stats']
        n_A_bins = sec.getint('n-apogee-bins')
        n_D_bins = sec.getint('n-deriv-bins')

        kwargs = {
            'min_val': sec.getint('min-apogee'),
            'max_val': sec.getint('max-apogee'),
            'low_clip': sec.getfloat('apogee-deriv-low-prune'),
            'high_clip': sec.getfloat('apogee-deriv-high-prune'),
            'n_p_bins': n_A_bins,
            'n_D_bins': n_D_bins,
            'key': 'A',
            }
        dig_A = self.__concat_and_digitize(resampled, deriv, **kwargs)
        (Ap_min, Ap_max, dAp, Ap,
         Ad_min, Ad_max, dAd, Ad,) = dig_A

        n_P_bins = sec.getint('n-perigee-bins')
        kwargs = {
            'min_val': sec.getint('min-perigee'),
            'max_val': sec.getint('max-perigee'),
            'low_clip': sec.getfloat('perigee-deriv-low-prune'),
            'high_clip': sec.getfloat('perigee-deriv-high-prune'),
            'n_p_bins': n_P_bins,
            'n_D_bins': n_D_bins,
            'key': 'P',
            }
        dig_P = self.__concat_and_digitize(resampled, deriv, **kwargs)
        (Pp_min, Pp_max, dPp, Pp,
         Pd_min, Pd_max, dPd, Pd,) = dig_P

        logging.info(f"Quantifying Moral Decay")

        index, univ_A, univ_P, = self.__universalize(Ap, Ad, Pp, Pd,
                                                     n_A_bins,
                                                     n_P_bins,
                                                     n_D_bins)
        moral_decay = self.__bin_universalized_array(index,
                                                     n_A_bins,
                                                     n_P_bins,
                                                     n_D_bins,)

        # FIXME: Any normalization steps for things like B* compared
        # to mean would happen at this stage.

        return MoralDecay(moral_decay, resampled, deriv,
                          Ap_min, Ap_max, dAp, Ad_min, Ad_max, dAd,
                          Pp_min, Pp_max, dPp, Pd_min, Pd_max, dPd)
