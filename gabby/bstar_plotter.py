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
        Xt, bs, bounds, = self._fetch_bstar(txn,
                                            scope_start, scope_end,
                                            start_d, end_d, dt)

        pprint.pprint(bounds)

        # FIXME
        sys.exit(0)

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

        # We'll want to find the zeros in this one
        bounds = np.zeros((L, 2), np.int)

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

            # Start by catching up to the start_ts
            while True:
                k, v, = cursor.item()
                if not k.startswith(prefix): break
                ts = int(k[-12:])
                if ts >= start_ts: break
                cursor.next()

            j = 0
            while True:
                # Termination conditions for the inner (time) loop:
                #  * We are out of observations of the fragment
                #  * We are beyond the time window of interest
                if not k.startswith(prefix): break
                ts = int(k[-12:])

                # We use +1 here so that j=0 registers a truth value
                # (1).  We can easily decrement this later.
                if not bounds[i][0]: bounds[i][0] = j+1

                # Cursor is behind
                if ts < prev_ts:
                    cursor.next()
                    k, v, = cursor.item()

                # We're in the window -- accumulate
                elif prev_ts < ts < next_ts:
                    prev_ts = ts
                    (n, ndot, nddot, bstar,
                     tle_num, inc, raan, ecc,
                     argp, mean_anomaly,
                     rev_num) = struct.unpack(TLE_STRUCT_FMT, v)
                    prev_v += bstar
                    prev_n += 1

                # Window is behind
                else:
                    prev_ts = next_ts
                    next_ts += dt_s

                    # We aren't guaranteed to have accumulated anything
                    if prev_n:
                        # Record the average value
                        Bs[j][i] = prev_v / prev_n

                        # Record the finish
                        bounds[i][1] = j

                    # Whether or not we have data, moving the window
                    # means moving its index.
                    j += 1

                    # We reset unless we're between the targets
                    prev_v = 0
                    prev_n = 0

                if ts >= end_ts: break

        # We deliberately made these values be +1 earlier so that a 0
        # would register as a truth value.  Now we decrement.
        for i in range(len(bounds)): bounds[i][0] -= 1

        return Xt, Bs, bounds
