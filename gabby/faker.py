#!/usr/bin/env python

import astropy.constants
import lmdb
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import tempfile

from .defs import *
from .utils import *
from .db import GabbyDB
from .undertaker import Undertaker

class FakeDB(object):
    """Builds and manages a fake DB.
    """

    def __init__(self, cfg, tgt,
                 output_dir=None,
                 db_path=None):

        # If we want it to persist, we pass in a directory, otherwise
        # we generate a tempdir here
        if not db_path: db_path = tempfile.TemporaryDirectory().name
        self.db_path = db_path
        self.db = GabbyDB(path=db_path)
        self.output_dir = output_dir

        self.cfg = cfg
        self.tgt = tgt

    def build_single(self):
        """Builds and stores the tAPT values for a single fragment.

        A brain-dead naive function is used to make apogee/perigee
        curves that sorta-kinda 'look' like the real values is used.
        At some point I should rip that out and put in a proper decay
        model.  The <life> is the number of days until it reaches
        <decay-alt>.  The <start> is the date/time of the first APT
        value.

        Consumes:
          * single-intldes
          * single-A
          * single-P
          * single-life
          * single-start
          * single-output
          * single-decay-alt
        """

        des = self.tgt['single-intldes']
        A0 = self.tgt.getfloat('single-A')
        P0 = self.tgt.getfloat('single-P')
        t0_date = parse_date(self.tgt['single-start'])
        t0 = dt_to_ts(t0_date)
        L0 = self.tgt.getint('single-life')
        h1 = self.tgt.getfloat('single-decay-alt')


        if 'single-output' in self.tgt:
            dirname = self.output_dir
            fname = self.tgt['single-output'] % {'des':des}
            if dirname: output = os.path.join(dirname, fname)
            else: output = fname
        else: output = None

        logging.info(f"Building single satellite fragment:")
        logging.info(f"  Designator: {des}")
        logging.info(f"  Apogee:     {A0}")
        logging.info(f"  Perigee:    {P0}")
        logging.info(f"  Lifetime:   {L0}")
        logging.info(f"  Decay Alt:  {h1}")
        logging.info(f"  Start Date: {t0_date}")

        t, A, P, T = self._fake_sat(A0, P0, t0, L0,
                                    decay_alt=h1,
                                    output_path=output)
        self._fill_apt(des, zip(t, A, P, T))

    def build_linear(self):
        """Builds and stores the tAPT values for a single linear-decay.

        Pretty similar to the build_single method, but instead of
        trying to sorta look like a decay profile, it's just a couple
        of straight lines.  That makes derivative computations (read:
        verifications) easy.

        Consumes:
          * linear-intldes
          * linear-A
          * linear-P
          * linear-life
          * linear-start
          * linear-output
          * linear-decay-alt
        """

        des = self.tgt['linear-intldes']
        A0 = self.tgt.getfloat('lienar-A')
        P0 = self.tgt.getfloat('linear-P')
        t0_date = parse_date(self.tgt['linear-start'])
        t0 = dt_to_ts(t0_date)
        L0 = self.tgt.getint('linear-life')
        h1 = self.tgt.getfloat('linear-decay-alt')


        if 'linear-output' in self.tgt:
            dirname = self.output_dir
            fname = self.tgt['linear-output'] % {'des':des}
            if dirname: output = os.path.join(dirname, fname)
            else: output = fname
        else: output = None

        logging.info(f"Building satellite fragment with linear decay:")
        logging.info(f"  Designator: {des}")
        logging.info(f"  Apogee:     {A0}")
        logging.info(f"  Perigee:    {P0}")
        logging.info(f"  Lifetime:   {L0}")
        logging.info(f"  Decay Alt:  {h1}")
        logging.info(f"  Start Date: {t0_date}")

        t = np.linspace(0, L0, L0)
        A = np.linspace(A0, h1, L0)
        P = np.linspace(P0, h1, L0)
        T = self._T(A, P)
        self._fill_apt(des, zip(t, A, P, T))

    def build_norm(self):
        """Builds and stores a normal variate of fragments.

        The periods will be computed as a function of apogee and
        perigee.  The variance can only be 0 if the number of
        fragments is 1.  If the variance is 0, then the exact numbers
        will be used.

        Consumes:
          * norm-intldes
          * norm-A
          * norm-P
          * norm-life
          * norm-n
          * norm-dev-frac
          * start-date
        """

        N = self.tgt.getint('norm-n')
        rho = self.tgt.getfloat('norm-dev-frac')
        des = self.tgt['norm-intldes']
        A0 = self.tgt.getfloat('norm-A')
        P0 = self.tgt.getfloat('norm-P')
        L0 = self.tgt.getint('norm-life')
        t0_date = parse_date(self.tgt['start-date'])
        t0 = int(t0_date.timestamp())

        logging.info(f"Building Normal variate constellation:")
        logging.info(f"  N:          {N}")
        logging.info(f"  Rho:        {rho}")
        logging.info(f"  Designator: {des}")
        logging.info(f"  Apogee:     {A0}")
        logging.info(f"  Perigee:    {P0}")
        logging.info(f"  Lifetime:   {L0}")
        logging.info(f"  Start Date: {t0_date}")

        for i in range(N):
            # FIXME: select A, P, and L from a normal distribution
            tapt = self._fake_sat(A, P, t0, L)
            self._fill_apt(des, zip(tapt))
            assert(False) # Unimplemented

    def _T(self, A, P):
        Re = (astropy.constants.R_earth/1000.0).value
        RA = A+Re
        RP = P+Re
        e = (RA-RP) / (RA+RP)

        # These are all in meters
        a = 1000*(RA+RP)/2
        mu = astropy.constants.GM_earth.value
        T = 2 * np.pi * (a**3/mu)**.5
        # T is now in seconds

        T /= 60.0
        # T is now in minutes which is waht the DB expects

        return T

    def _fake_sat(self, A0, P0, t0, lifetime,
                  decay_alt=100,
                  moon_wobble_amp=2,
                  moon_wobble_decay_coeff=2,
                  output_path=None,
                  frag=None):

        logging.info(f"  Building fragment: A={A0} P={P0} t={t0} L={lifetime}")

        h1 = decay_alt

        t = np.linspace(0, lifetime, lifetime)

        k = np.log(P0 - h1 + 1) / lifetime
        base = P0 - np.exp(k * t) + 1

        k = np.log(A0-P0+1)/lifetime
        A_correction = -1 * (P0-A0+1) * np.exp(-1*k*t)

        moon_decay = np.exp((-1.0 * moon_wobble_decay_coeff * t / lifetime))
        moon_cos = np.cos(t*np.pi/(56))
        moon_wobble = moon_decay * moon_wobble_amp * moon_cos

        A = base + A_correction + moon_wobble
        P = base + moon_wobble
        T = self._T(A, P)

        if output_path: plot_apt(frag, (t, A, P, T), output_path)

        logging.debug(f"    First APT: A={A[0]} P={P[0]} T={T[0]}")
        logging.debug(f"    Last APT:  A={A[-1]} P={P[-1]} T={T[-1]}")

        return t*SECONDS_IN_DAY+t0, A, P, T

    def _fill_apt(self, frag, tapt):
        """Fills the DB with APT values.

        frag: '99025ABC'
        tapt: [[timestamp(float), A, P, T], ...]

        The timestamp will be casted to an integer before use.

        Imports the APT values to the DB, then builds the scope index.
        """

        txn = self.db.txn(write=True)
        for wat in tapt:
            ts, A, P, T, = wat

            key = fmt_key(des=frag, ts=ts)
            apt_bytes = pack_apt(A=A, P=P, T=T)
            txn.put(key, apt_bytes,
                    db=self.db.db_apt,
                    overwrite=True)
        txn.commit()

        undertaker = Undertaker(db_path=self.db_path)
        undertaker.index_db()
