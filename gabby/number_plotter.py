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


class NumberPlotter(object):
    """Plots total pieces of debris.
    """

    def __init__(self,
                 cfg=None,
                 tgt=None,
                 img_dir=None,
                 output_dir=None,
                 cache_dir=None,
                 data=None,
                 db=None):

        # Preserve the basic inputs
        self.cfg = cfg
        self.tgt = tgt
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.data = data
        self.db = db

        self.cache = GabbyCache(cache_dir) if cache_dir else None

        # The colors are meant to be interpreted directly by matplotlib
        self.obs_color = self.tgt['observation-color']
        self.prop_color = self.tgt['propagation-color']

        # The datetime objects along the x-axis of the plot with the
        # number of fragments.
        self.Xt = np.arange(self.data.start_d,
                            self.data.end_d+self.data.dt,
                            self.data.dt)

        self.Xts = np.arange(dt_to_ts(self.data.start_d),
                             dt_to_ts(self.data.end_d+self.data.dt),
                             self.data.dt.total_seconds())

        # The date on which we start showing forward propagation
        if 'fwd-prop-start-date' in self.tgt and self.tgt['fwd-prop-start-date']:
            timestr = self.tgt['fwd-prop-start-date']
            self.fwd_prop_start_dt = parse_date_d(timestr)
            self.fwd_prop_start_ts = dt_to_ts(self.fwd_prop_start_dt)
            for i in range(len(self.Xts)):
                if self.Xts[i] == self.fwd_prop_start_ts:
                    self.fwd_prop_idx = i
                    break
                elif self.Xts[i] > self.fwd_prop_start_ts:
                    self.fwd_prop_idx = i-1
                    break
        else:
            self.fwd_prop_start_date = None
            self.fwd_prop_idx = None

        # Indicates that we're doing a comparison of propagated vs
        # observed values
        self.prop_after_obs = tgt.getboolean('prop-after-observation')

        self.plt_setup()

    def plot(self, n_threads=None):

        # Set up the figure and axes
        fig = plt.figure(figsize=(12, 8))
        fig.set_dpi(self.tgt.getint('dpi'))
        fig.suptitle(self.tgt['name'], y=0.97, fontsize=25)

        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel('Number of Fragments')
        ax.set_xlim(self.data.start_d, self.data.end_d)
        ax.set_ylim(0, self.tgt.getint('max-n-fragments'))
        ax.set_xlabel(self.tgt['copyright'])

        # Plot the number of pieces.  There are two options here: 1)
        # we're plotting a single line with two colors indicating the
        # growth followed by predicted decay and 2) we're plotting a
        # comparison of the predicted decay vs observed decay.

        if self.prop_after_obs:
            # We're going to plot a comparison of our propagator vs
            # observed values.
            ax.plot(self.Xt, self.data.Ns_obs,
                    color=self.obs_color,
                    label='Observations')
            ax.plot(self.Xt, self.data.Ns,
                    color=self.prop_color,
                    label='Propagation')
        elif self.fwd_prop_idx:
            # We're going to plot observations to a given date, then
            # look at decay thereafter in a different color
            obs_idx = self.fwd_prop_idx
            ax.plot(self.Xt[:obs_idx+1], self.data.Ns[:obs_idx+1],
                    color=self.obs_color,
                    label='Observations')
            ax.plot(self.Xt[obs_idx:], self.data.Ns[obs_idx:],
                    color=self.prop_color,
                    label='Propagation')
        else:
            # We're just plotting observations
            ax.plot(self.Xt, self.data.Ns,
                    color=self.obs_color,
                    label='Observations')

        ax.legend(loc=1)

        mkdir_p(self.img_dir)
        fname = self.tgt.name[4:] + '.png'
        fig.savefig(os.path.join(self.img_dir, fname))

    def plt_setup(self):
        """Perform initialization that doesn't serialize.

        This only needs to be once per process.
        """
        # We disable the interactive representation so we don't
        # duplicate effort.  This saves us a considerable amount of
        # time during image generation.
        plt.ioff()

        # We get some pretty bad memory leaks with the default backend
        # when used on mac.  I haven't checked this for windows or
        # linux, so be wary.
        if 'darwin' in sys.platform: matplotlib.use('TkAgg')
