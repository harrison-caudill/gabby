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


class MoralDecay(object):
    """Explores the depths of moral (and orbial) decay rates.

    decay_hist:
      [A=0,P=1][A-bin][P-bin][D-bin]

    resampled, derivatives: <CloudDescriptor>
      Their N's should match

      Derivative bins are from D_min-D_max for the specific
      combination of A/P A and
    """

    def __init__(self, decay_hist,
                 Ap_min, Ap_max, dAp, Ad_min, Ad_max, dAd,
                 Pp_min, Pp_max, dPp, Pd_min, Pd_max, dPd):

        self.decay_hist = decay_hist

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

        self.bins_Ap = np.linspace(Ap_min, Ap_max, self.n_A_bins)
        self.bins_Pp = np.linspace(Pp_min, Pp_max, self.n_P_bins)
        self.bins_Ad = np.linspace(Ad_min, Ad_max, self.n_D_bins)
        self.bins_Pd = np.linspace(Pd_min, Pd_max, self.n_D_bins)

        # FIXME: Square time-domain filters don't have the best
        # frequency-domain responses, but, meh, works fine.
        n = 3
        m = 2*n-1
        kernel=np.ones(m**2).reshape((m,m))
        kernel /= np.sum(kernel)

        # FIXME: Triangular(ish) window works a bit better...don't
        # remember which window function this is and what its OOB
        # rejection is...also don't care right now.
        window = np.array(list(range(1, n, 1))+[n]+list(range(n-1, 0, -1)))
        kernel = np.sqrt(np.outer(window, window))
        kernel /= np.sum(kernel)

        # FIXME: Let's try a Blackman Harris Window
        window = np.abs(np.blackman(m))
        kernel = np.sqrt(np.outer(window, window))
        kernel /= np.sum(kernel)

        self.cdf = self._cdf()
        self.percentiles = self._percentiles()
        self.median = self._median(kernel=kernel)
        self.mean = self._mean(kernel=kernel)

    def index_for(self, A, P):
        idx_A = min(int((A - self.Ap_min) / self.dAp), self.n_A_bins-1)
        idx_P = min(int((P - self.Pp_min) / self.dPp), self.n_P_bins-1)
        return max(idx_A, 0), max(idx_P, 0)

    def _mean(self, kernel=None):
        """Finds the expectation value for each bin.

        At first blush, we don't necessarily care about the full
        probability distribution.
        """

        retval = np.zeros((2, self.n_A_bins, self.n_P_bins), dtype=np.float32)
        for i in range(self.n_A_bins):
            for j in range(self.n_P_bins):
                retval[0][i][j] = np.sum((self.bins_Ad - self.dAd)*self.decay_hist[0][i][j])
                retval[1][i][j] = np.sum((self.bins_Pd - self.dPd)*self.decay_hist[1][i][j])
                assert(0 >= retval[0][i][j])
                assert(0 >= retval[1][i][j])

        if kernel is not None:
            retval[0] = scipy.signal.convolve2d(retval[0], kernel, mode='same')
            retval[1] = scipy.signal.convolve2d(retval[1], kernel, mode='same')

        return retval

    def index_array(self, data, axis='A'):
        """Takes an array of data and gives back the indexes into the bins.

        If you have an array of apogee and perigee values, for
        example, you may want to know which bins they belong to in the
        decay_hist.
        """
        if 'A' == axis:
            min_val = self.Ap_min
            max_val = self.Ap_max
            step = self.dAp
            logging.info(f"Indexing Array")
            logging.info(f"  range: {min_val} => {max_val}")
        else:
            min_val = self.Pp_min
            max_val = self.Pp_max
            step = self.dPp

        retval = (data - min_val) / step

        # make sure an int8 is sufficient
        assert(1<<8 > (max_val-min_val)/step)

        return retval.astype(np.int8)

    def plot_mesh(self, path, data='median', axis='A'):
        """Plots a 2-axis mesh of showing A/P/Z where Z is usually median.

        data: mean or median
        axis: A or P

        Lets you get a quick visual representation of the historical
        values.
        """

        logging.info(f"Plotting PColorMesh to {path}")
        logging.info(f"  {self.Ap_min}, {self.Ap_max}")

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel('Apogee Bin')
        ax.set_xlabel('Perigee Bin')
        fig.suptitle('dA/dt (km/day)')

        idx = 0 if 'A' == axis else 1
        src = self.median if 'median' == data else self.mean

        src = np.zeros(self.mean.shape, dtype=int)
        for a_idx in range(self.n_A_bins):
            for p_idx in range(self.n_P_bins):
                for t in range(2):
                    src[t][a_idx][p_idx] = 1 if self.median[1][a_idx][p_idx] > self.median[0][a_idx][p_idx] else 0

        src = self.median

        c = ax.pcolor(src[idx])

        bins_A = (self.bins_Ap * 100).astype(np.int).astype(np.float32)/100.0
        bins_P = (self.bins_Pp * 100).astype(np.int).astype(np.float32)/100.0

        #ax.set_ylim(bins_A[0], bins_A[-1])
        #ax.set_yticklabels(bins_A)
        #ax.set_xlim(bins_P[0], bins_P[-1])
        #ax.set_xticklabels(bins_P)

        fig.colorbar(c, ax=ax)
        fig.savefig(path)

    def plot_dA_vs_P(self, path):
        """Stem plot of dA/dt vs perigee.

        Since it's usually the perigee value that dominates the loss
        of orbital energy (at least for atmospheric decay which is
        largely what we're tracking here) I have a special method for it.
        """
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
        """Continuous Distribution Function of the frequency histograms.

        returns np array of the same shape as decay_hist, but instead
        of being a PMF in the Z axis, it's a CDF.
        """
        cdf = np.copy(self.decay_hist)
        for i in range(self.n_A_bins):
            for j in range(self.n_P_bins):
                cdf[0][i][j] = np.cumsum(cdf[0][i][j])
                cdf[1][i][j] = np.cumsum(cdf[1][i][j])
        return cdf

    def _percentiles(self):
        """Finds the bin for the given percentile.

        If, for example, you have 20 bins, then you will have 5%
        increments.  The first bin is always 0, and the last bin is
        always n_D_bins - 1.

        returns [2][n_A][n_P][n_D], type=int8
        """
        pct = np.copy(self.cdf).astype(np.int8)
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

    def _median(self, kernel=None):
        """Finds the median value for each bin.

        At first blush, we don't necessarily care about the full
        probability distribution.
        """

        retval = np.zeros((2, self.n_A_bins, self.n_P_bins), dtype=np.float32)
        for i in range(self.n_A_bins):
            for j in range(self.n_P_bins):
                idx = self.percentiles[0][i][j][self.n_D_bins//2]
                retval[0][i][j] = self.bins_Ad[idx] - self.dAd
                idx = self.percentiles[1][i][j][self.n_D_bins//2]
                retval[1][i][j] = self.bins_Pd[idx] - self.dPd

        if kernel is not None:
            retval[0] = scipy.signal.convolve2d(retval[0], kernel, mode='same')
            retval[1] = scipy.signal.convolve2d(retval[1], kernel, mode='same')

        return retval


    @classmethod
    def cache_name(cls, stats_cfg):
        return 'moral_decay-' + cfg_hash(stats_cfg)
