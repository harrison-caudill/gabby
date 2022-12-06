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
from .moral_decay import MoralDecay

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


class Jazz(object):
    """Finds Moral Decay in the data.
    """

    def __init__(self, cfg):
        self.cfg = cfg

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
        k = 100000
        n_taps = 127
        fltr = np.arange(-1*(n_taps//2), n_taps//2+1, 1) * np.pi / k
        fltr = (1/k) * np.sinc(fltr)
        fltr /= np.sum(fltr)
        return fltr


    @classmethod
    def deriv_cache_name(cls, stats_cfg):
        return 'deriv-' + sats_hash(stats_cfg)


    @classmethod
    def filtered_cache_name(cls, stats_cfg):
        return 'filtered-' + sats_hash(stats_cfg)

    @classmethod
    def moral_decay_from_cfg(cls, cfg, db, cache=None):
        stats_cfg = cfg['stats']

        # Start by checking the cache
        decay_name = MoralDecay.cache_name(stats_cfg)
        if cache and decay_name in cache:
            logging.info(f"  Found moral decay in the cache")
            decay = cache[decay_name]
            decay.plot_mesh('output/mesh-A.png', axis='A')
            decay.plot_mesh('output/mesh-P.png', axis='P')
            return decay

        logging.info(f"Transformers, more than meets the eye!")

        # Jazz will do all the heavy lifting here
        jazz = Jazz(cfg)

        # names for cache-lookups
        filtered_name = Jazz.filtered_cache_name(stats_cfg)
        deriv_name = Jazz.deriv_cache_name(stats_cfg)

        base_frags = json.loads(stats_cfg['historical-asats'])
        fragments = db.find_daughter_fragments(base_frags)
        apt = db.load_apt(fragments)

        if cache and deriv_name in cache:
            logging.info(f"  Found filtered/derivative values in global cache")
            deriv = cache[deriv_name]
            filtered = cache[filtered_name]
        else:
            logging.info(f"  Stats not found in cache -- building anew")
            filtered, deriv = jazz.filtered_derivatives(apt,
                                                        min_life=1.0,
                                                        dt=SECONDS_IN_DAY,
                                                        fltr=jazz.lpf())
            logging.info(f"  Saving derivatives to cache")
            cache[deriv_name] = deriv
            cache[filtered_name] = filtered

        # FIXME: Deal with solar-activity compensation later
        decay = jazz.decay_rates(apt, filtered, deriv, drag=None)
        if cache:
            logging.info(f"  Adding a little moral decay to the cache")
            cache[decay_name] = decay

        return decay

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
        logging.info("  Binning dX/dP")
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

    def _slow_decay_rates(self, apt, resampled, deriv):
        """Very slowly and tediously bin the data.

        This is largely here to aid unit testing.
        """

        sec = self.cfg['stats']

        n_A_bins = sec.getint('n-apogee-bins')
        Ap = self._concatenate(resampled.A, resampled.N)
        Ap_min = sec.getint('min-apogee')
        Ap_max = sec.getint('max-apogee')
        Ap_bins = np.linspace(Ap_min, Ap_max, n_A_bins)

        n_P_bins = sec.getint('n-perigee-bins')
        Pp = self._concatenate(resampled.P, resampled.N)
        Pp_min = sec.getint('min-perigee')
        Pp_max = sec.getint('max-perigee')
        Pp_bins = np.linspace(Pp_min, Pp_max, n_P_bins)

        n_D_bins = sec.getint('n-deriv-bins')

        Ad = self._concatenate(deriv.A, resampled.N)
        low_clip = sec.getfloat('apogee-deriv-low-prune')
        high_clip = sec.getfloat('apogee-deriv-high-prune')
        Ad_min, Ad_max, = self._percentile_values(Ad, low_clip, high_clip)
        Ad_bins = np.linspace(Ad_min, Ad_max, n_D_bins)

        Pd = self._concatenate(deriv.P, resampled.N)
        low_clip = sec.getfloat('perigee-deriv-low-prune')
        high_clip = sec.getfloat('perigee-deriv-high-prune')
        Pd_min, Pd_max, = self._percentile_values(Pd, low_clip, high_clip)
        Pd_bins = np.linspace(Pd_min, Pd_max, n_D_bins)

        data = [[Ad, Ad_bins],
                [Pd, Pd_bins]]

        # Numpy's digitize will take the Ap_max value and place it
        # beyond the final bin.  So what we do is to move anything
        # above our max value to be below the min.  That way, when we
        # run the digitize, all the values that are too high end up
        # being in the 0'th (discarded) bin.  Then we can clip the
        # value on the high end to be the final bin which should only
        # show up when the value is exactly the maximum.
        Ap_clipped = np.where(Ap>Ap_max, Ap_min-1, Ap)
        Ap_step = (Ap_max-Ap_min)/n_A_bins
        tmp_bins = np.concatenate(([Ap_min-Ap_step/2], Ap_bins))
        Ap_binned = np.digitize(Ap_clipped, tmp_bins, right=True)
        Ap_binned = np.clip(Ap_binned, 0, n_A_bins) - 1

        Ad_clipped = np.where(Ad>Ad_max, Ad_min-1, Ad)
        Ad_step = (Ad_max-Ad_min)/n_D_bins
        tmp_bins = np.concatenate(([Ad_min-Ad_step/2], Ad_bins))
        Ad_binned = np.digitize(Ad_clipped, tmp_bins, right=True)

        Pp_clipped = np.where(Pp>Pp_max, Pp_min-1, Pp)
        Pp_step = (Pp_max-Pp_min)/n_P_bins
        tmp_bins = np.concatenate(([Pp_min-Pp_step/2], Pp_bins))
        Pp_binned = np.digitize(Pp_clipped, tmp_bins, right=True)

        Pd_clipped = np.where(Pd>Pd_max, Pd_min-1, Pd)
        Pd_step = (Pd_max-Pd_min)/n_D_bins
        tmp_bins = np.concatenate(([Pd_min-Pd_step/2], Pd_bins))
        Pd_binned = np.digitize(Pd_clipped, tmp_bins, right=True)

        retval = np.zeros((2, n_A_bins+1, n_P_bins+1, n_D_bins+1),
                          dtype=np.float32)

        logging.info(f"Binning: {len(Ap_binned)}")
        for i in range(len(Ap_binned)):
            retval[0][Ap_binned[i]][Pp_binned[i]][Ad_binned[i]] += 1
            retval[1][Ap_binned[i]][Pp_binned[i]][Pd_binned[i]] += 1
            if not i % 1000 and i:
                logging.info(f"  Binned: {i//1000}k")

        return (Ap_min, Ap_max, Ap_step, Ap,
                Ad_min, Ad_max, Ad_step, Ad,
                Pp_min, Pp_max, Pp_step, Pp,
                Pd_min, Pd_max, Pd_step, Pd,
                retval[:,1:,1:,1:])

    def decay_rates(self, apt, resampled, deriv, drag=None):
        """Bins the decay rate distributions.

        retval: [A'=0,P'=1][A-bin][P-bin][D-bin] = d(A/P)/dt

        dt is defined in the call to derivatives() defaulting to 1 day.
        """

        logging.info(f"Reticulating Splines") # Tee hee
        if drag: drag.normalize_decay_rates(deriv)

        deriv.A = np.where(deriv.A > 0, 0, deriv.A)
        deriv.P = np.where(deriv.P > 0, 0, deriv.P)

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
        
        # logging.info(f"Slowly Quantifying Moral Decay")
        # start = datetime.datetime.now().timestamp()

        # (Ap_min, Ap_max, dAp, Ap,
        #  Ad_min, Ad_max, dAd, Ad,
        #  Pp_min, Pp_max, dPp, Pp,
        #  Pd_min, Pd_max, dPd, Pd,
        #  moral_decay) = self._slow_decay_rates(apt, resampled, deriv)
        # end = datetime.datetime.now().timestamp()
        # logging.info(f"    Slow Version took: {int((end-start)*1000)}ms")

        # FIXME: Any normalization steps for things like B* compared
        # to mean would happen at this stage.

        return MoralDecay(moral_decay,
                          Ap_min, Ap_max, dAp, Ad_min, Ad_max, dAd,
                          Pp_min, Pp_max, dPp, Pd_min, Pd_max, dPd)


class SolarDrag(object):
    """Container for historical solar-induced drag.

    Mostly, we just look at global percentiles of the bstar values.
    """

    def __init__(self, t, pct, vals, mean, fenceposts, t_step):
        """
        t: timestamp of the beginning of the bin
        pct: percentiles in vals (e.g. 0, .25, .5, .75, 1)
        vals: each percentile in each time-bin
        mean: mean for each time-bin
        fenceposts: boundaries of the bins
        t_step: nominal spacing between bins

        Data Types:
        t: np array of length <n-bins>
        pct: np array of length <n-pct>
        vals: np array of shape <n-pct> x <n-bins>
        mean: np array of length <n-bins>
        fenceposts: np array of length <n-bins>+1
        t_step: float

        """
        self.t = t
        self.pct = pct
        self.vals = vals
        self.mean = mean
        self.fenceposts = fenceposts

        self.t_step = t_step
        self.n_bins = len(mean)
        self.n_pct = len(pct)

        self.bstar_factor = self.vals / np.mean(self.vals[:, 2])

    def normalize_decay_rates(self, deriv):
        """Normalizes decay rates as a function of solar activity at the time.

        deriv: CloudDescriptor
        """

        logging.info(f"Normalizing Derivatives for Solar Activity")

        N = math.prod(deriv.t.shape)
        t = np.reshape(deriv.t, N)
        A = np.reshape(deriv.A, N)
        P = np.reshape(deriv.A, N)

        # Bin numbers inside of the solar activity table
        b = np.floor(t / self.t_step).astype(np.uint64)

        # Ugh...it's super slow to do a single loop for lookups, and
        # it's also super slow to do an np.where for each bin.  So
        # we'll do the old combine-into-a-single-uint64 trick.  it'll
        # be <bin-number><offset>.  Sort by bin number, replace whole
        # ranges of the bin numbers with the right factor, then
        # reverse the values to be <offset><factor> resort and remove
        # the <offset>.

        # Number of bits we need for the <offset>
        n_bits = math.ceil(math.log(N+1, 2))

        # Combined value of <bin-number><offset>
        off = np.linspace(0, N-1, N, dtype=np.uint64)
        C = b << n_bits + off
        C = np.sort(C)

        # Order in which the values appear
        o = C & ((1<<n_bits)-1)

        # Sorted bin numbers
        b = C >> n_bits

        # Locations of the first bin to have a given value
        idx = np.searchsorted(b, np.linspace(0, self.n_bins, self.n_bins+1))

        # The scaling factor as a function of bin number
        F = 1.0 / np.clip(self.bstar_factor[:, 2], 1.0e-4, None)

        # The factor to be applied, sorted by bin number
        f = np.zeros(N, dtype=np.float64)
        for i in range(self.n_bins):
            f[idx[i]:idx[i+1]] = F[i]

        # Recombine and sort
        n_bits = math.ceil(math.log(np.max(F), 2))
        C = np.sort((o << n_bits).astype(np.float64) + f)

        # Grab the newly reordered values
        f = C - (off<<n_bits)

        # The scaled derivatives
        retval_A = (A * f).reshape(deriv.A.shape)
        retval_P = (P * f).reshape(deriv.P.shape)

        # Assign the results
        deriv.A = retval_A
        deriv.P = retval_P

    def plot_bstar(self, output, log=True):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        #if log: ax.set_yscale('log')
        ax.set_ylabel('B*')
        ax.set_xlabel('Date')
        fig.suptitle('B* Variation over Time')
        X = [ts_to_dt(x) for x in self.t]
        for i in range(self.n_bins): assert(self.vals[i, -1])
        #ax.plot(X, self.vals[:, 1], label='25th Pct')
        ax.plot(X, self.vals[:, 2], label='Median')
        #ax.plot(X, self.vals[:, 3], label='75th Pct')
        ax.hlines(np.mean(self.vals[:, 2]), X[0], X[-1], label='Mean of the Median')
        ax.legend()
        fig.savefig(output)

class Optimus(object):
    """Normalizes for atmospheric density.

    Space-Track.org's data is not as advertised.  They do NOT provide
    TLEs or OMM data; they provide something close to it.  The B* drag
    term is supposed to be a statement about the spacecraft, but is
    instead a statement about the combination of the spacecraft and
    the current space weather.  It makes some amount of sense that
    they'd combine the numbers in this fashion since the TLE format
    does not include a term for space weather otherwise.

    The Optimus class exists to first find the mean B* value as a
    function of time, and then to normalize the B* values against that
    mean.
    """

    def __init__(self, base, cache, db):
        self.base = base
        self.fragments = db.find_daughter_fragments(base)
        self.cache = cache
        self.db = db

    def _load_bstar(self, cache=True):
        logging.info(f"Finding B* History")

        txn = self.db.txn(write=False)
        cursor = txn.cursor(db=self.db.db_tle)
        cursor.first()

        M = 1024
        N = 0
        B = np.zeros(M, dtype=np.float32)
        T = np.zeros(M, dtype=np.int)


        b_name = f"bstar-raw-B"
        t_name = f"bstar-raw-T"
        if cache and b_name in self.cache:
            B = self.cache[b_name]
            T = self.cache[t_name]

        else:
            # Gather all of the bstar values
            for k, v in cursor:
                b = unpack_tle(v)[TLE_OFF_BSTAR]
                T[N] = parse_key(k)[1]
                B[N] = b
                N += 1
                if N >= M:
                    M *= 2
                    B.resize(M)
                    T.resize(M)

                if 0 == N % 1e6:
                    logging.info(f"  Loaded {int(N/1e6)} M B* entries")

            B = B[:N]
            T = T[:N]

            if cache:
                self.cache[b_name] = B
                self.cache[t_name] = T

        return B, T

    def bstar_percentiles(self,
                          dt_days=10,
                          pct=[0, .25, .5, .75, 1],
                          cache=True):
        """Finds the percentiles for bstar values for all TLEs available.

        We aren't actually finding the mean, because it's a massively
        skewed sample.  However, for the purpose of a first-order
        approximation, it should be fine.  We'll literally just add up
        all the bstar values and divide by N for each time bucket.
        """

        name = f"solar-drag-{dt_days}"
        if cache and name in self.cache: return self.cache[name]

        pct = np.array(pct, dtype=np.float32)

        d, t, B = self.db.cache_tle_field(TLE_OFF_BSTAR)

        # # Remove any 0/negative values of B* as they aren't useful here
        # mask = np.where(B <= 0, 0, 1)
        # t *= mask
        # B *= mask
        # B = (B[B != 0]).astype(np.float64)
        # t = (t[t != 0]).astype(np.uint64)
        N = len(B)

        # We need the min and max for a number of reasons
        t_min = np.min(t)
        t_max = np.max(t)
        t -= t_min

        # Sort the B* values
        logging.info("  Sorting B* Values")
        s = np.sum(B)

        # Bits used by timestamps
        t_bits = math.ceil(math.log(np.max(t)+1, 2))

        # Bits that will be used by the B* values
        B_bits = 64 - t_bits

        # Max value
        B_max = (1 << B_bits) - 1

        # Occupied range of B* values
        B_used = np.max(B) - np.min(B)

        # Since we're quantizing the value of B*, let's find our
        # quantizer value
        B_states = B_max + 1
        B_quant = B_used / B_states

        # Additive factor to B
        B_off = np.min(B)

        # Multiplicative factor
        B_factor = float(B_max) / B_used

        # Combine and sort
        C = np.sort((t<<B_bits).astype(np.uint64)
                    + (B_factor*(B+B_off)).astype(np.uint64))

        # Reconstruct
        t = (C >> B_bits)
        B = ((C - (t << B_bits)).astype(np.float64) / B_factor) - B_off
        t += t_min

        ### t/B are now sorted in temporal order

        # Enumerate our fence posts/bins.  There will be N+1 fence
        # posts for N buckets.
        t_step = datetime.timedelta(days=dt_days).total_seconds()
        n_bins = math.ceil((t_max - t_min) / t_step)
        print(f"n_bins: {n_bins}")
        print(f"t_min:  {t_min}")
        print(f"t_max:  {t_max}")
        print(f"t_step: {t_step}")
        fenceposts = np.linspace(t_min, t_max, n_bins+1, dtype=np.float64)
        bins = fenceposts[:-1]

        # Index the sorted array
        logging.info(f"Indexing bin locations in times")
        idx = np.searchsorted(t, fenceposts)
        print(len(idx))
        print(len(fenceposts))
        print(n_bins)
        assert(len(idx) == len(fenceposts) == n_bins+1)

        # Number of percentile values being observed
        n_pct = len(pct)

        # Output values
        vals = np.zeros((n_bins, len(pct)), dtype=np.float32)
        mean = np.zeros(n_bins, dtype=np.float32)

        logging.info(f"Binning Data")
        t = np.sort(t)
        for i in range(n_bins):
            logging.info(f"  Binning {i+1}/{n_bins}")
            a = idx[i]
            b = idx[i+1]
            tmp = B[a:b]
            L = len(tmp) - 1 # We subtract 1 so that 1*L is the last entry
            for j in range(n_pct):
                if 0 >= L:
                    vals[i][j] = 0
                    mean[i] = 0
                else:
                    k = int(L*pct[j])
                    vals[i][j] = np.partition(tmp, k)[k]
            mean[i] = np.mean(tmp)
        logging.info(f"  Done Binning Data")

        retval = SolarDrag(t, pct, vals, mean, fenceposts, t_step)

        self.cache[name] = retval

        return retval
