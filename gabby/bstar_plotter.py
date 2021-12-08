import datetime
import gc
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

from .defs import *
from .utils import *
from .transformer import Jazz


def _plot_image(arg):
    """Multiprocessable function to generate a single gabby plot.

    This is broken out because matplotlib is not presently threadsafe,
    so we have to use multiprocessing to make it work.

    Xt: x-axis (N datetimes)
    Bs: bstar values [NxL]
    scope: { fragment => (start_idx, end_idx) }
    n_img: number of images we're building in total
    start_d: starting datetime
    end_d: ending datetime
    tgt: configparser results for the plot section
    output_dir: just what it says
    indexes: list of integer indexes for the images to build

    """
    (
        Xt,
        Bs,
        scope,
        n_img,
        start_d,
        end_d,
        tgt,
        output_dir,
        indexes,
        ) = arg

    logging.basicConfig(level=logging.INFO)

    plt.ioff()
    matplotlib.use('TkAgg')


class BStarPlotter(object):

    def __init__(self,
                 cfg=None,
                 tgt=None,
                 output_dir=None,
                 img_dir=None,
                 db_path=None,
                 db_env=None):
        self.cfg = cfg
        self.tgt = tgt
        self.output_dir = output_dir
        self.img_dir = img_dir
        self.db_path = db_path
        load_dbs(self, db_env, db_path)

    def plot(self, n_threads=1):
        """Produces the images, but not the video.

        This will read the config file for all relevant information.
        """

        # We're going to store the images in a separate directory for
        # cleanliness
        img_dir = os.path.join(self.output_dir, "bstar-img")

        target_des = self.tgt['intldes'].strip().split(',')
        target_des = [s.strip() for s in target_des]
        logging.info(f"Preparing to plot data for {target_des} => {img_dir}")

        logging.info(f"  Creating image directory: {img_dir}")
        mkdir_p(img_dir)

        # Read only transaction is all we'll need.
        txn = lmdb.Transaction(self.db_env, write=False)

        # Pull in the time boundaries from the config file
        start_d = datetime.datetime.strptime(self.tgt['start-date'],
                                             self.cfg['timestamp-fmt'])
        end_d = datetime.datetime.strptime(self.tgt['end-date'],
                                           self.cfg['timestamp-fmt'])

        # First, find when pieces come into scope and when they go out
        scope_start, scope_end = load_scope(txn, self.db_scope, target_des)

        # Initialize our main cursor
        cursor = txn.cursor(db=self.db_tle)

        # Time step between images
        dt = datetime.timedelta(days=self.tgt.getint('plot-period'))

        # Figure out the ending index
        last_idx = int((end_d - start_d) / dt)

        # Fetch all the BStar values into RAM
        Xt, Bs, = self._fetch_bstar(txn,
                                    scope_start, scope_end,
                                    start_d, end_d, dt)
        self._plot_time_img(Xt, Bs)

        # FIXME
        sys.exit(0)

    def _plot_time_img(self, Xt, Bs):
        N = len(Bs)
        L = len(Bs[0])
        fig = plt.figure(figsize=(12, 8))
        fig.set_dpi(self.tgt.getint('dpi'))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_yscale("log")

        fig.suptitle(f"Mean/Std B* Values")
        ax.set_ylabel("BStar")
        ax.set_xlabel("Observation Date")

        val = np.zeros(N, dtype=np.float32)
        std = np.zeros(N, dtype=np.float32)
        for i in range(N):
            val[i] = np.mean(Bs[i])
            std[i] = np.std(Bs[i])

        low = np.where(val-std < 0, 0, val-std)
        ax.fill_between(Xt, low, val+std, alpha=0.5, color='red')
        ax.plot(Xt, val, color='black')
        fig.savefig('wat.png')

    def _fetch_bstar(self,
                     txn,
                     scope_start, scope_end,
                     start_d, end_d,
                     dt):
        """Fetches the time-averaged B* values from the DB

        This runs in about 4.5 seconds for Fengyun
        
        === Daily Average ===
        N (days) x L (frag) float32 BStar values

        +
        |    ----          Plot the IQR or similar showing how the
        | --'___ \         distribution changes over time.  Allows passing
        | --'   \ `------  individual arrays to matplotlib without alteration.
        |        `-------
        +----------------+
               Time

        === Distribution Animation ===
        N (days) x L (frag) float32 BStar values

        +
        |    ----          Plot a series of images to be animated like a
        | --'    \         gabby plot.  Allows contiguous memory access for
        |         `------  generating each individual plot.
        |                
        +----------------+
               Time
        """

        # Initialize our main cursor
        cursor = txn.cursor(db=self.db_tle)

        # Get our index into the fragments
        fragments = sorted(list(scope_start.keys()))

        # Get our array dimensions
        N = int(math.ceil((end_d - start_d)/dt))
        L = len(fragments)

        logging.info(f"  Shape of data: {N}, {L}")

        # Get the integer timestamps to use for indexing
        start_ts = int(start_d.timestamp())
        end_ts = int(end_d.timestamp())
        dt_s = int(dt.total_seconds())
        Xt = np.arange(start_ts, end_ts, dt_s)

        # Temporally-averaged B* values for each fragment
        Bs = np.zeros((N, L), dtype=np.float32)

        # The big main loop
        for i in range(L):
            frag = fragments[i]
            logging.info(f"    Fetching data for {frag} ({i+1}/{L})")

            # Loop through the observations
            prefix = (frag+',').encode()
            cursor.set_range(fmt_key(start_ts, frag))

            # The values we accumulate
            prev_v = 0.0
            prev_n = 0

            # The range of timestamps we're looking for
            prev_ts = start_ts + dt_s
            next_ts = prev_ts + dt_s

            k, v, = cursor.item()
            j = 0
            while True:
                # Termination conditions for the inner (time) loop:
                #  * We are out of observations of the fragment
                #  * We are beyond the time window of interest
                if not k.startswith(prefix): break
                ts = int(k[-12:])
                if ts >= end_ts: break

                # Since we start by seeking the cursor to the first
                # key equal to or greater than the starting timestamp,
                # we know that the first encountered timestamp will be
                # >= prev_ts.  So if the current timestamp is less
                # than next, we accumulate observations.
                if ts < next_ts:
                    prev_v += struct.unpack(TLE_STRUCT_FMT, v)[3]
                    prev_n += 1
                    cursor.next()
                    k, v, = cursor.item()

                # Window is behind
                else:
                    step = (ts - next_ts)//dt_s + 1
                    prev_ts += step*dt_s
                    next_ts += step*dt_s

                    # We aren't guaranteed to have accumulated anything
                    if prev_n:
                        # Record the average value
                        Bs[j][i] = prev_v / prev_n

                    # Whether or not we have data, moving the window
                    # means moving its index.
                    j += step

                    # We reset unless we're between the targets
                    prev_v = 0
                    prev_n = 0

        return Xt, Bs
