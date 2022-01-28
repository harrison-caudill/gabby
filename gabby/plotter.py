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

    ts: timestamps [NxL]
    As: apogees [NxL]
    Ps: perigees [NxL]
    Ts: periods [NxL]
    Xt: x-axis (N datetimes)
    N: num fragments (N)
    fragments: [str, ...] list of fragment names
    comparators: static comparators for the gabby plot
    n_img: number of images we're building in total
    start_d: starting datetime
    end_d: ending datetime
    tgt: configparser results for the plot section
    output_dir: just what it says
    indexes: list of integer indexes for the images to build

    """
    (
        ts,
        As,
        Ps,
        Ts,
        Xt,
        N,
        fragments,
        comparators,
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

    for img_idx in indexes:
        # Set up the figure and axes
        fig = plt.figure(figsize=(12, 8))
        fig.set_dpi(tgt.getint('dpi'))
        ax_g = fig.add_subplot(2, 1, 1)
        ax_g.set_xlabel('Orbital Period (minutes)')
        ax_g.set_ylabel('Orbital Altitude (km)')
        ax_g.set_xlim(tgt.getint('min-orb-period'),
                      tgt.getint('max-orb-period'))
        ax_g.set_ylim(tgt.getint('min-orb-alt'),
                      tgt.getint('max-orb-alt'))

        ax_n = fig.add_subplot(2, 1, 2)
        ax_n.set_xlim(start_d, end_d)
        ax_n.set_ylim(0, tgt.getint('max-n-fragments'))
        ax_n.set_ylabel('Number of Fragments')
        ax_n.set_xlabel(tgt['copyright'])

        # Find the colors for the gabbard
        apogee_color = tgt['apogee-color']
        perigee_color = tgt['perigee-color']

        # Set up the legend
        legend_contents = [
            matplotlib.lines.Line2D([0], [0],
                                    color='white',
                                    markerfacecolor=apogee_color,
                                    marker='.',
                                    markersize=12),
            matplotlib.lines.Line2D([0], [0],
                                    color='white',
                                    markerfacecolor=perigee_color,
                                    marker='.',
                                    markersize=12),
            ]
        legend_labels = [
            'Fragment Apogee',
            'Fragment Perigee',
            ]

        logging.info(f"  Preparing plot ({img_idx}/{n_img})")

        # Plot the number of pieces
        ax_n.plot(Xt[:img_idx+1], N[:img_idx+1])

        # Plot the comparators
        for des in comparators:
            cur = comparators[des]
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
            legend_contents.append(legend_patch)
            legend_labels.append(name)
            ax_g.plot(T, O, 'o', color=color, markersize=12)

        # Now that we have all of the legend contents from the static
        # comparators, time to actually add it to the figure.
        ax_g.legend(legend_contents, legend_labels, loc=1)

        ax_g.plot(Ts[img_idx], As[img_idx], 
                  '.', color=apogee_color, markersize=6)
        ax_g.plot(Ts[img_idx], Ps[img_idx],
                  '.', color=perigee_color, markersize=6)

        # Futz with boundaries and location
        fig.tight_layout(h_pad=2)
        fig.subplots_adjust(top=0.9)

        fig.suptitle(tgt['name'], y=0.97, fontsize=25)

        # We're going to store the images in a separate directory for
        # cleanliness
        img_dir = os.path.join(output_dir, "img")

        # Save everything
        path = f"{img_dir}/%*.*d.png"%(len(str(n_img)),
                                       len(str(n_img)),
                                       img_idx)

        fig.savefig(path)
        logging.info(f"  Figure saved to {path}")

        fig.clf()
        plt.close(fig)
        gc.collect()


class Conspirator(object):

    def __init__(self,
                 cfg=None,
                 tgt=None,
                 staging_dir=None,
                 output_dir=None,
                 db_path=None,
                 db_env=None):
        self.cfg = cfg
        self.tgt = tgt
        self.output_dir = output_dir
        self.staging_dir = staging_dir
        self.db_path = db_path

        if db_env: self.db_env = db_env
        else: self.db_env = lmdb.Environment(self.db_path,
                                             max_dbs=len(DB_NAMES),
                                             map_size=int(DB_MAX_LEN))
        self.db_gabby = self.db_env.open_db(DB_NAME_GABBY.encode())
        self.db_idx = self.db_env.open_db(DB_NAME_IDX.encode())
        self.db_scope = self.db_env.open_db(DB_NAME_SCOPE.encode())

    def get_latest_apt(self, txn, des):
        """Returns the latest APT for the given designator
        """
        cursor = txn.cursor(db=self.db_scope)
        cursor.set_range(des.encode())
        key, scope = cursor.item()
        start, end = struct.unpack('ii', scope)

        key = fmt_key(end, des)
        tmp = txn.get(key, db=self.db_gabby)
        if not tmp:
            key = fmt_key(start, des)
            tmp = txn.get(key, db=self.db_gabby)
            if not tmp:
                print(key)
                sys.exit(0)
        a, p, t, = struct.unpack('fff', tmp)
        del cursor
        return (a, p, t,)

    def plot(self, n_threads=1):
        """Produces the images, but not the video.

        This will read the config file for all relevant information.
        """

        # We're going to store the images in a separate directory for
        # cleanliness
        img_dir = os.path.join(self.output_dir, "img")

        target_des = self.tgt['intldes'].strip().split(',')
        target_des = [s.strip() for s in target_des]
        logging.info(f"Preparing to plot data for {target_des} => {img_dir}")

        logging.info(f"  Creating image directory: {img_dir}")
        mkdir_p(img_dir)

        # Read only transaction is all we'll need.
        txn = lmdb.Transaction(self.db_env, write=False)

        # Log the time so we can track how long it's going to take
        # plot_start = datetime.datetime.now()

        # Pull in the time boundaries from the config file
        start_d = datetime.datetime.strptime(self.tgt['start-date'],
                                             self.cfg['timestamp-fmt'])
        end_d = datetime.datetime.strptime(self.tgt['end-date'],
                                           self.cfg['timestamp-fmt'])

        # Find the designators we're using for comparison
        static_comparators = self.tgt['static-comparators'].strip().split(',')
        static_comparators = [s.strip() for s in static_comparators]
        logging.info(f"  Using static comparators: {static_comparators}")
        comparators = {}
        for comp in static_comparators:
            des, name, color, = comp.split('|')
            cur = {'name': name,
                   'color': color,
                   'apt': tuple(self.get_latest_apt(txn, des)),
                   'is_static': True
                   }
            comparators[des] = cur

        # Mask off the original rocket body, as it's a distraction
        mask = self.tgt['mask'].strip().split(',')
        mask = [s.strip() for s in mask]
        logging.info(f"  Masking off: {mask}")

        # First, find when pieces come into scope and when they go out
        logging.info(f"  Finding the scope of all fragments")
        scope_start = {}
        scope_end = {}
        scope_cursor = txn.cursor(db=self.db_scope)
        for des, scope in scope_cursor:
            des = des.decode()
            if des in mask: continue
            for prefix in target_des:
                if des.startswith(prefix):
                    start, end = struct.unpack('ii', scope)
                    scope_start[des] = start
                    scope_end[des] = end

        # Initialize our main cursor
        cursor = txn.cursor(db=self.db_idx)

        # Time step between images
        dt = datetime.timedelta(days=self.tgt.getint('plot-period'))

        # Figure out the ending index
        last_idx = int((end_d - start_d) / dt)

        # Build the plot of debris pieces
        Xt = [start_d]
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

        # Find the raw values
        t, A, P, T, = self._prep_plot_data(txn,
                                           scope_start, scope_end,
                                           start_d, end_d, dt)
        fragments = sorted(list(scope_start.keys()))

        logging.info(f"  Preparing to build {last_idx+1} images")

        n_img = int(math.ceil((end_d - start_d)/dt))+1


        if n_threads > 1:
            indexes = []
            n_per = int(math.ceil(n_img/n_threads))
            img_idx = 0
            logging.info(f"  Divvying up the work")
            for i in range(n_img):
                if not (i % n_per): indexes.append([])
                indexes[-1].append(i)
            logging.info(f"  Done divvying up work")
            assert(len(indexes) == n_threads)

            work = [(
                t,
                A,
                P,
                T,
                Xt,
                N,
                fragments,
                comparators,
                n_img,
                start_d,
                end_d,
                self.tgt,
                self.output_dir,
                idx,)
                    for idx in indexes]

            logging.info(f"  Launching the pool with {n_threads} threads")
            with multiprocessing.Pool(n_threads) as pool:
                pool.map(_plot_image, work)
        else:
            for i in range(n_img):
                _plot_image((t, A, P, T,
                            Xt, N,
                            fragments, comparators,
                            n_img,
                            start_d, end_d, self.tgt, self.output_dir, [i]))

    def _prep_plot_data(self,
                        txn,
                        scope_start, scope_end,
                        start_d, end_d,
                        dt):

        # Initialize our main cursor
        cursor = txn.cursor(db=self.db_idx)

        # Get our index into the fragments
        fragments = sorted(list(scope_start.keys()))

        # Get our array dimensions
        L = len(fragments)
        N = int(math.ceil((end_d - start_d)/dt))+1

        logging.info(f"  Shape of data: {N}, {L}")

        # Get the integer timestamps to use for indexing
        start_ts = int(start_d.timestamp())
        end_ts = int(end_d.timestamp())
        dt_s = int(dt.total_seconds())
        timestamps = np.arange(start_ts, end_ts+dt_s, dt_s)

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

        # The big main loop
        for i in range(L):
            frag = fragments[i]
            logging.info(f"    Fetching data for {frag} ({i+1}/{L})")

            # Loop through the observations
            prefix = (frag+',').encode()
            cursor.set_range(prefix)

            # The starting value we linearly interpolate from
            prev_v = None
            prev_ts = 0

            # The next targeted timestamp
            next_ts = int(start_ts)

            k, v, = cursor.item()
            j = 0
            while True:
                if not k.startswith(prefix): break
                ts = int(k[-12:])
                if ts > end_ts: break

                if ts < next_ts:
                    prev_ts = ts
                    prev_v = v
                    cursor.next()
                    k, v, = cursor.item()
                else:
                    after_ts[j][i] = ts
                    tgt_ts[j][i] = next_ts
                    if prev_v:
                        A, P, T = struct.unpack('fff', prev_v)
                        before_A[j][i] = A
                        before_P[j][i] = P
                        before_T[j][i] = T
                        before_ts[j][i] = prev_ts
                        A, P, T = struct.unpack('fff', v)
                        after_A[j][i] = A
                        after_P[j][i] = P
                        after_T[j][i] = T

                    next_ts += dt_s
                    j += 1

        logging.info(f"  Computing temporal offsets")
        dt = after_ts - before_ts
        off_before = (tgt_ts - before_ts)/dt
        off_after = (after_ts - tgt_ts)/dt

        logging.info(f"  Interpolating Values")
        t = tgt_ts
        A = before_A * off_before + after_A * off_after
        P = before_P * off_before + after_P * off_after
        T = before_T * off_before + after_T * off_after

        logging.info(f"  Removing the nans")
        A = np.where(np.isnan(A), 0, A)
        P = np.where(np.isnan(A), 0, P)
        T = np.where(np.isnan(A), 0, T)

        return t, A, P, T
