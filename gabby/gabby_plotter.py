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
        self.mask = json.loads(self.tgt['mask'])

        # Pull in the time boundaries from the config file
        self.start_d = parse_date_d(self.tgt['start-date'])
        self.end_d = parse_date_d(self.tgt['end-date'])

        # Time step between images
        self.dt = datetime.timedelta(days=self.tgt.getint('plot-period'))

    def fetch_from_db(self, db):
        """Loads the data from the database.
        """

        # First, find when pieces come into scope and when they go out
        self.scope_start, self.scope_end = db.load_scope(self.target_des)

        # The results will be sorted in this order
        fragments = sorted(list(self.scope_start.keys()))
        fragments = [f for f in fragments if f not in self.mask]
        self.names = fragments

        # Basic read-only transaction
        txn = db.txn()

        # Initialize our main cursor
        cursor = txn.cursor(db=db.db_apt)
        cursor.first()

        # Get our array dimensions
        L = self.L = len(fragments)
        N = self.N = int(math.ceil((self.end_d - self.start_d)/self.dt))+1

        logging.info(f"  Shape of data: {N}, {L}")

        # Get the integer timestamps to use for indexing
        start_ts = dt_to_ts(self.start_d)
        end_ts = dt_to_ts(self.end_d)
        dt_s = int(self.dt.total_seconds())
        timestamps = np.arange(start_ts, end_ts+dt_s, dt_s, dtype=np.int)
        assert(N == len(timestamps))

        # Before and after values for the timestamp and Apogee/Perigee
        before_ts = np.zeros((N, L), dtype=np.float32)
        tgt_ts = np.zeros((N, L), dtype=np.float32)
        after_ts = np.zeros((N, L), dtype=np.float32)

        before_A = np.zeros((N, L), dtype=np.float32)
        after_A = np.zeros((N, L), dtype=np.float32)

        before_P = np.zeros((N, L), dtype=np.float32)
        after_P = np.zeros((N, L), dtype=np.float32)

        before_T = np.zeros((N, L), dtype=np.float32)
        after_T = np.zeros((N, L), dtype=np.float32)

        valid = np.zeros((N, L), dtype=np.int8)

        # The big main loop
        j = 0
        for i in range(L):
            frag = fragments[i]
            logging.info(f"    Fetching data for {frag} ({i+1}/{L})")
            frag_b = frag.encode()
            frag_len = len(frag)

            prev_ts = None
            prev_v = None

            # loop through the target timestamps
            for j in range(N):
                tgt_ts = timestamps[j]

                # Seek to the target
                srch = fmt_key(des=frag, ts=tgt_ts)
                cursor.set_range(srch)

                # This is the first on/or AFTER the target
                k, v = cursor.item()

                # If we don't have anything on/after the target, we're
                # done
                if not k[:frag_len] == frag_b: break
                #if not k: break

                des, ts = parse_key(k)

                # If we're beyond the end of the table, or the end of
                # the fragment, we just continue
                #if des != frag: break

                # Since the key is guaranteed to be on or after the
                # search key, we can unconditionally assign the next
                # value here.
                next_ts = ts
                next_v = v

                if ts == tgt_ts:
                    # If we are *exactly* at the target, we can assign
                    # the prev val to 0 and exactly one step away.
                    prev_ts = tgt_ts - dt_s
                    prev_v = v

                else:
                    # We have an observation AFTER, but we need one
                    # before to interpolate.

                    # Move the cursor back for the before timestamp/value
                    cursor.prev()
                    k, v = cursor.item()

                    # If we don't have anything before the current
                    # observation, then we'll just have to hope that
                    # we can find a next->next value.
                    if not k[:frag_len] == frag_b: continue
                    #if not k: continue

                    des, ts = parse_key(k)

                    # Same as above.
                    #if des != frag: continue

                    # We win.  We found a previous entry.
                    prev_ts = ts
                    prev_v = v

                if prev_v:
                    # Register the values
                    A, P, T = unpack_apt(prev_v)
                    before_ts[j][i] = prev_ts
                    before_A[j][i] = A
                    before_P[j][i] = P
                    before_T[j][i] = T
                    A, P, T = unpack_apt(next_v)
                    after_ts[j][i] = next_ts
                    after_A[j][i] = A
                    after_P[j][i] = P
                    after_T[j][i] = T
                    valid[j][i] = 1

                    # We win
                    j += 1

        # We're done with our read-only transaction now
        txn.commit()

        logging.info(f"  Computing temporal offsets")
        dt = after_ts - before_ts
        off_before = (tgt_ts - before_ts)/dt
        off_after = (after_ts - tgt_ts)/dt

        logging.info(f"  Interpolating Values")
        t = tgt_ts
        A = before_A * off_before + after_A * off_after
        P = before_P * off_before + after_P * off_after
        T = before_T * off_before + after_T * off_after

        Ns = np.sum(valid, axis=1, dtype=np.int64)

        logging.info(f"  Removing the nans")
        A = np.where(np.isnan(A), 0, A)
        P = np.where(np.isnan(A), 0, P)
        T = np.where(np.isnan(A), 0, T)

        logging.info(f"  Registering the resulting data")
        self.ts = timestamps
        self.As = A
        self.Ps = P
        self.Ts = T
        self.Ns = Ns
        self.valid = valid


class GabbyPlotContext(object):
    """Easily serializeable object with necessary data for plotting.

    We break this out into a separate and object to make it easier to
    use multiprocessing to process images.  For convenience, we place
    the logic to load the data in here.  It also makes unit tests
    easier as we don't have to entangle the logic for matplotlib with
    the logic for the data.
    """

    def __init__(self, tgt, data, output_dir):
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
        self.img_dir = os.path.join(self.output_dir, 'gabby-img')

        # The colors are meant to be interpreted directly by matplotlib
        self.apogee_color = self.tgt['apogee-color']
        self.perigee_color = self.tgt['perigee-color']

        # The datetime objects along the x-axis of the plot with the
        # number of fragments.
        self.Xt = np.arange(self.data.start_d,
                            self.data.end_d+self.data.dt,
                            self.data.dt)

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
        ax_n.plot(ctx.Xt[:idx+1], ctx.data.Ns[:idx+1])

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
            ctx, _ = self.cache.get('gabby_plot_ctx')
        else:
            logging.info(f"  Building plot context")
            ctx = GabbyPlotContext(tgt=self.tgt,
                                   data=self.data,
                                   output_dir=self.output_dir)

            logging.info(f"  Loading data from DB")
            ctx.fetch_from_db(self.db)

            self.cache.put('gabby_plot_ctx', ctx, [])

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
