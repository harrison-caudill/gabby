#!/usr/bin/env python

import datetime
import numpy as np
import pprint
import pytest
import tempfile

from .fixtures import *
from ..utils import *
from ..tle import TLE


class TestUndertaker(object):

    def test_calendar_offsets(undertaker):

        offsets = Undertaker._build_calendar_offsets()

        # Check leap years
        # https://www.thelists.org/list-of-leap-years.html

        leap_years = [60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 0, 4,
                      8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52,
                      56,]
        for year in leap_years:
            assert(0 < offsets[year][365])
        for year in range(100):
            if year in leap_years: continue
            assert(0 == offsets[year][365])

        cur = EPOCH
        dt = datetime.timedelta(days=1)
        end_date = datetime.datetime(2057, 1, 1, tzinfo=datetime.timezone.utc)
        while cur < end_date:
            year_start = datetime.datetime(cur.year, 1, 1,
                                           tzinfo=datetime.timezone.utc)
            year_off = cur.year % 100
            off = (cur - year_start)
            day_off = off.days
            off = cur - EPOCH
            assert(offsets[year_off][day_off] == off.total_seconds())
            cur += dt

    def test_tle(self, undertaker):
        """Whitebox test
        """

        def __APT(n, ecc):
            """From space-track documentation
            https://www.space-track.org/documentation#faq
            """
            min_per_day = 24*60
            sec_per_day = min_per_day*60
            mu = 398600.4418
            a = (mu/(n*2*np.pi/sec_per_day)**2)**(1/3)
            A = (a * (1 + ecc)) - 6378.135
            P = (a * (1 - ecc)) - 6378.135
            T = min_per_day / n
            return A, P, T

        # Verify basic writes
        test_tles = []
        cat_number = 12345
        launch_number = 1
        tlefile = tempfile.NamedTemporaryFile()
        fd = open(tlefile.name, 'w')
        obs_time = datetime.datetime.now(tz=datetime.timezone.utc)
        for year in [2000, 1956, 1957, 1958, 1999]:
            t = TLE(cat_number=cat_number,
                    launch_year=year,
                    launch_number=launch_number,
                    epoch=obs_time,
                    n=15,
                    ndot=-1.0e-5,
                    nddot=0.0,
                    bstar=1e-5,
                    tle_num=1,
                    inc=51.6,
                    raan=0.0,
                    ecc=1.0e-6,
                    argp=0.0,
                    mean_anomaly=0.0,
                    rev_num=1)
            cat_number += 1
            launch_number += 1
            test_tles.append(t)
            lines = str(t).split('\n')
            tlestr = ''.join([
                lines[1], '\\', '\n',
                lines[2], '\n',
                ])
            fd.write(tlestr)
        fd.close()
        undertaker.load_tlefile(tlefile.name, store_tles=True)

        fragments = [t.intldes for t in test_tles]
        taptn = undertaker.db.load_apt(fragments)
        for i in range(len(test_tles)):
            t = test_tles[i]
            A, P, T = __APT(t.n, t.eccentricity)
            assert(abs(A - taptn[1][i][0]) < 1)
            assert(abs(P - taptn[2][i][0]) < 1)
            assert(abs(T - taptn[3][i][0]) < 1)

        # base_des: None, [A], [A, B]
        # force: true/false
        # format: conforming, missing_checksum
        # designator: 99025AB, 99025 AB
