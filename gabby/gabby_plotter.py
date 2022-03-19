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


class GabbyPlotContext(object):
    """Easily serializeable object with necessary data for plotting.

    We break this out into a separate and object to make it easier to
    use multiprocessing to process images.  For convenience, we place
    the logic to load the data in here.  It also makes unit tests
    easier as we don't have to entangle the logic for matplotlib with
    the logic for the data.
    """

    def __init__(self, tgt, data, output_dir, img_dir=None):
        """
        tgt: configparser results for the plot section
        data: FragmentData
        fragments: [str, ...] list of fragment names
        start_d: starting datetime
        end_d: ending datetime
        output_dir: just what it says
        indexes: list of integer indexes for the images to build
        """

        # Preserve the basic inputs
        self.tgt = tgt
        self.data = data
        self.output_dir = output_dir

        # We're going to store the images in a separate directory for
        # cleanliness
        if not img_dir: img_dir = os.path.join(self.output_dir, 'gabby-img')
        self.img_dir = img_dir

        # The colors are meant to be interpreted directly by matplotlib
        self.apogee_color = self.tgt['apogee-color']
        self.perigee_color = self.tgt['perigee-color']

        # The datetime objects along the x-axis of the plot with the
        # number of fragments.
        self.Xt = np.arange(self.data.start_d,
                            self.data.end_d+self.data.dt,
                            self.data.dt)

        self.Xts = np.arange(dt_to_ts(self.data.start_d),
                             dt_to_ts(self.data.end_d+self.data.dt),
                             self.data.dt.total_seconds())

        # The date on which we start showing forward propagation
        if 'fwd-prop-start-date' in self.tgt:
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

    def fetch_from_db(self, db):
        """Fetches any necessary data from the DB for the plot context.

        This largely fetches the latest APT values for the
        comparators.  The rest of the data will have already been
        fetched.
        """

        # Basic read-only transaction
        txn = db.txn()

        # Loads the comparators
        static_comparators = json.loads(self.tgt['static-comparators'])
        logging.info(f"  Using static comparators: {static_comparators}")
        comparators = {}
        for comp in static_comparators:
            des, name, color, = comp
            cur = {'name': name,
                   'color': color,
                   'apt': tuple(db.get_latest_apt(txn, des)),
                   'is_static': True
                   }
            comparators[des] = cur

        txn.commit()

        self.comparators = comparators

    def plt_setup(self):
        """Performs setup of matplotlib objects.

        Matplotlib objects can't be serialized so they have to be
        re-generated for each process.
        """
        # The legend is shared by all and customized.
        self.legend_contents = [
            matplotlib.lines.Line2D([0], [0],
                                    color='white',
                                    markerfacecolor=self.apogee_color,
                                    marker='.',
                                    markersize=12),
            matplotlib.lines.Line2D([0], [0],
                                    color='white',
                                    markerfacecolor=self.perigee_color,
                                    marker='.',
                                    markersize=12),
            ]
        self.legend_labels = [
            'Fragment Apogee',
            'Fragment Perigee',
            ]

        # Process the comparators
        comp_X = []
        comp_Y = []
        comp_C = []
        for des in self.comparators:
            cur = self.comparators[des]
            apt = cur['apt']
            color = cur['color']
            name = cur['name']
            A,P,T, = apt
            O = (A+P)/2
            legend_patch = matplotlib.lines.Line2D([0], [0],
                                                   color='white',
                                                   markerfacecolor=color,
                                                   marker='o',
                                                   markersize=12)
            self.legend_contents.append(legend_patch)
            self.legend_labels.append(name)
            comp_X.append(T)
            comp_Y.append(O)
            comp_C.append(color)

        self.comp_X = comp_X
        self.comp_Y = comp_Y
        self.comp_C = comp_C


class GabbyPlotter(object):

    def __init__(self,
                 cfg=None,
                 tgt=None,
                 img_dir=None,
                 output_dir=None,
                 cache_dir=None,
                 data=None,
                 db=None):
        self.cfg = cfg
        self.tgt = tgt
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.data = data
        self.db = db

        self.cache = GabbyCache(cache_dir) if cache_dir else None

    def plot_prediction(self):
        jazz = Jazz(self.cfg,
                    self.frag_env,
                    self.frag_apt,
                    self.frag_tle,
                    self.frag_scope)
        frags, apt, deriv, N = jazz.derivatives(fltr=jazz.lpf(),
                                                cache_dir=self.cache_dir)

        (moral_decay,
         bins_A, bins_P,
         Ap, Ad, Pp, Pd,) = jazz.decay_rates(apt, deriv, N)

        n_A_bins = self.cfg.getint('n-apogee-bins')
        n_P_bins = self.cfg.getint('n-perigee-bins')
        n_D_bins = self.cfg.getint('n-deriv-bins')

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(Ap, bins=n_A_bins+2)
        fig.savefig(os.path.join(self.output_dir, 'Ap_hist.png'))

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(Pp, bins=n_P_bins+2)
        fig.savefig(os.path.join(self.output_dir, 'Pp_hist.png'))

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(Ad, bins=n_D_bins+2)
        fig.savefig(os.path.join(self.output_dir, 'Ad_hist.png'))

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(Pd, bins=n_D_bins+2)
        fig.savefig(os.path.join(self.output_dir, 'Pd_hist.png'))

        tmp = np.concatenate(Ad)

        fig = plt.figure(figsize=(12, 8))
        fig.set_dpi(self.tgt.getint('dpi'))
        ax = fig.add_subplot(1, 1, 1)
        tmp = np.sort(tmp)
        N = len(tmp)
        tmp = tmp[N//5:-1*(N//5)]
        ax.hist(tmp, bins=100)
        fig.savefig('output/wat.png')
        sys.exit(0)

    def plot_scope(self):
        # Build the plot of debris pieces
        Xt = [ts_to_dt(start_d)]
        for i in range(last_idx): Xt.append(Xt[-1]+dt)
        N = np.zeros(len(Xt))
        idx = 0
        for d in Xt:
            n = 0
            ts = int(d.timestamp())
            for des in scope_start:
                start = scope_start[des]
                end = scope_end[des]
                if start <= ts <= end: n += 1
            N[idx] = n
            idx += 1
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.grid()
        ax.plot(Xt, N)
        img_path = os.path.join(self.output_dir, "num_fragments.png")
        fig.savefig(img_path)

    @classmethod
    def _plt_setup(cls, ctx):
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

        # The cached PLT legend entries, etc need to be built on a
        # per-process basis also.
        ctx.plt_setup()

    @classmethod
    def _mp_gabby_plot(cls, ctx):
        """Multiprocessing method for gabby plots.
        """

        # Since it's a new process, we'll need to reinitialize the
        # logging infrastructure
        logging.basicConfig(level=logging.INFO)

        # This routine has to be run once per os level process
        GabbyPlotter._plt_setup(ctx)

        for idx in ctx.indexes: cls._plot_gabby_frame(ctx, idx)

    @classmethod
    def _plot_gabby_frame(cls, ctx, idx):

        # Set up the figure and axes
        fig = plt.figure(figsize=(12, 8))
        fig.set_dpi(ctx.tgt.getint('dpi'))
        fig.suptitle(ctx.tgt['name'], y=0.97, fontsize=25)

        ax_g = fig.add_subplot(2, 1, 1)
        ax_g.set_xlabel('Orbital Period (minutes)')
        ax_g.set_ylabel('Orbital Altitude (km)')
        ax_g.set_xlim(ctx.tgt.getint('min-orb-period'),
                      ctx.tgt.getint('max-orb-period'))
        ax_g.set_ylim(ctx.tgt.getint('min-orb-alt'),
                      ctx.tgt.getint('max-orb-alt'))
        ax_g.legend(ctx.legend_contents, ctx.legend_labels, loc=1)

        ax_n = fig.add_subplot(2, 1, 2)
        ax_n.set_xlim(ctx.data.start_d, ctx.data.end_d)
        ax_n.set_ylim(0, ctx.tgt.getint('max-n-fragments'))
        ax_n.set_ylabel('Number of Fragments')
        ax_n.set_xlabel(ctx.tgt['copyright'])

        logging.info(f"  Preparing plot ({idx}/{ctx.data.N})")

        # Plot the number of pieces
        if ctx.fwd_prop_idx and idx >= ctx.fwd_prop_idx:
            obs_idx = min(idx+1, ctx.fwd_prop_idx)
            ax_n.plot(ctx.Xt[:obs_idx+1], ctx.data.Ns[:obs_idx+1],
                      color=ctx.perigee_color)
            ax_n.plot(ctx.Xt[obs_idx:idx+1], ctx.data.Ns[obs_idx:idx+1],
                      color=ctx.apogee_color)
        else:
            ax_n.plot(ctx.Xt[:idx+1], ctx.data.Ns[:idx+1],
                      color=ctx.perigee_color)

        # Plot the comparators
        for i in range(len(ctx.comp_X)):
            ax_g.plot(ctx.comp_X[i], ctx.comp_Y[i], 'o',
                      color=ctx.comp_C[i], markersize=12)

        # Plot the gabbards
        ax_g.plot(ctx.data.Ts[idx], ctx.data.As[idx], 
                  '.', color=ctx.apogee_color, markersize=6)
        ax_g.plot(ctx.data.Ts[idx], ctx.data.Ps[idx],
                  '.', color=ctx.perigee_color, markersize=6)

        # Futz with boundaries and location
        fig.tight_layout(h_pad=2)
        fig.subplots_adjust(top=0.9)

        # Save everything
        path = f"{ctx.img_dir}/%*.*d.png"%(len(str(ctx.data.N)),
                                           len(str(ctx.data.N)),
                                           idx)

        fig.savefig(path)
        logging.info(f"  Figure saved to {path}")

        fig.clf()
        plt.close(fig)
        gc.collect()

    def plot(self, n_threads=1):
        """Produces the images, but not the video.

        This will read the config file for all relevant information.
        """

        logging.info(f"Plotting")

        if 'gabby_plot_ctx' in self.cache:
            logging.info(f"  Loading context from cache")
            ctx = self.cache['gabby_plot_ctx']
        else:
            logging.info(f"  Building plot context")
            ctx = GabbyPlotContext(tgt=self.tgt,
                                   img_dir=self.img_dir,
                                   data=self.data,
                                   output_dir=self.output_dir)

            logging.info(f"  Loading data from DB")
            ctx.fetch_from_db(self.db)

            self.cache.put('gabby_plot_ctx', ctx, [])

        ctx.data.apt = None

        # We create a great many individual images
        logging.info(f"  Creating image directory: {ctx.img_dir}")
        mkdir_p(ctx.img_dir)

        logging.info(f"  Preparing to build {ctx.data.N} images")
        if n_threads > 1:

            # Interleave generation so that they're generated roughly
            # in order in parallel rather than one block at a time.
            # [ 0,  N+0,  2N+0, ...]
            # [ 1,  N+1,  2N+1, ...]
            # ...
            # [N-1, 2N-1, 3N-1, ...]
            tmp = np.linspace(0, ctx.data.N-1, ctx.data.N, dtype=np.int)
            indexes = [tmp[i::n_threads] for i in range(n_threads)]

            # Can't stop the work...
            work = []
            for i in range(n_threads):
                c = copy.deepcopy(ctx)
                c.indexes = indexes[i]
                work.append(c)

            logging.info(f"  Launching the pool with {n_threads} threads")
            with multiprocessing.Pool(n_threads) as pool:
                pool.map(GabbyPlotter._mp_gabby_plot, work)
        else:
            # One-time initialization per process
            self._plt_setup(ctx)
            for idx in range(ctx.data.N): self._plot_gabby_frame(ctx, idx)
