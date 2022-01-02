import datetime
import lmdb
import logging
import json
import numpy as np
import os
import pickle
import pprint
import subprocess
import struct
import sys
import tletools

from .defs import *

from . import utils


class Undertaker(object):
    """Brings data into the DB.
    """

    def __init__(self, db_path):
        self.db_path = db_path
        utils.load_dbs(self, db_path, None)
        self.db_env = self.full_env
        self.db_tle = self.full_tle
        self.db_apt = self.full_apt
        self.db_scope = self.full_scope

    def index_db(self):
        self._build_scope()

    def load_json(self, path, store_tles=False, base_des=None, force=False):
        """Loads the JSON-based TLE data into the DB.

        The import process is heavily duplicated because of the speed
        requirements of the TLE file import.
        """

        offsets = self._build_calendar_offsets

        logging.info(f"  Importing json data from {path}")
        txn = lmdb.Transaction(self.full_env, write=True)

        with open(path, 'r') as fd: data = json.load(fd)

        for datum in data['data']:
            des = datum['INTLDES']
            ts = parse_date(datum['EPOCH'])

            ### INLINE-CODE
            key = ("%s,%12.12d"%(des, ts)).encode()
            apogee = float(datum['APOGEE'])
            perigee = float(datum['PERIGEE'])
            period = float(datum['PERIOD'])
            apt_bytes = pack_apt(A=apogee, P=perigee, T=period)

            txn.put(key, apt_bytes,
                    db=self.full_apt,
                    overwrite=True)

            n = float(datum['MEAN_MOTION'])
            ndot = float(datum['MEAN_MOTION_DOT'])
            nddot = float(datum['MEAN_MOTION_DDOT'])
            bstar = float(datum['BSTAR'])
            tle_num = int(datum['ELEMENT_SET_NO'])
            inc = float(datum['INCLINATION'])
            raan = float(datum['RA_OF_ASC_NODE'])
            ecc = float(datum['ECCENTRICITY'])
            argp = float(datum['ARG_OF_PERICENTER'])
            mean_anomaly = float(datum['MEAN_ANOMALY'])
            rev_num = int(datum['REV_AT_EPOCH'])
            tle_bytes = pack_tle(n=n, ndot=ndot, nddot=nddot,
                                 bstar=bstar, tle_num=tle_num, inc=inc,
                                 raan=raan, ecc=ecc, argp=argp,
                                 mean_anomaly=mean_anomaly, rev_num=rev_num)
            txn.put(key, tle_bytes,
                    db=self.full_tle,
                    overwrite=True)

    def _build_calendar_offsets(self):
        logging.info("  Generating Time Lookup Table")
        # To make the timestamp computation go more quickly, we're
        # going to do it this way.  We DON'T keep track of leap
        # seconds, only leap days.
        offsets = np.zeros([100,366])
        for year in range(100):
            d = datetime.datetime(year=year+o,
                                  month=1,
                                  day=1,
                                  tzinfo=datetime.timezone.utc)
            for day in range(366):
                o = 2000 if year+1900 < EPOCH_YEAR else 1900
                try:
                    dt = datetime.timedelta(days=day)
                    offsets[year][day] = dt_to_ts(d + dt)
                except ValueError:
                    # This happens when there isn't a leap year
                    continue

        return offsets

    def load_tlefile(self, path, store_tles=False, base_des=None, force=False):
        """Loads the TLE file into the DB.

        Loads the TLE file and saves its entries in both the DB and
        the index.

        Because we're processing north of a billion records, we'll
        want to optimize things a bit.  Instead of using datetime to
        compute timestamps, we're going to use integers and floats.
        Instead of using the TLE class, we'll do the parsing and
        computations manually.  We also won't incur an additional
        jalr.  We're going to stop short of doing this natively
        though.  This is now efficient enough that the DB overhead is
        the dominant factor.

        If you specify a <base_des>, then only fragments stemming from
        that base_des will be imported.  This allows us to do things
        like do full TLE imports for a subset of the overall set, or
        use SGP4 to propagate a full orbit to determine the actual APT
        values, or ... without spending a decade importing the values.
        """

        offsets = self._build_calendar_offsets

        MINUTES_PER_DAY = 24*60
        SECONDS_PER_DAY = MINUTES_PER_DAY*60

        logging.info(f"  Importing data from {path}")
        txn = lmdb.Transaction(self.full_env, write=True)
        N = 0
        skipped = 0
        start = datetime.datetime.now().timestamp()
        non_conforming = []
        with open(path) as fd:
            while True:
                # Read the lines from the TLE
                line_1 = fd.readline().strip()
                line_2 = fd.readline().strip()

                if not (len(line_1) == 70 and len(line_2) == 69):
                    # readline returns 0-length when we're done
                    if not len(line_1): break

                    # In many cases, it's just missing the checksum
                    if line_1[62:64] != '0 ':
                        # TLE is non-conforming
                        non_conforming.append((line_1, line_2,))
                        continue

                # Find the designator
                des = line_1[9:17].strip().replace(' ', '')

                # We're matching the on the base designator
                if base_des and line_1[9:14] != base_des: continue

                # Compute the timestamp (signed integer, seconds since epoch)
                year = int(line_1[18:20])
                day_part = float(line_1[20:32])
                day = int(day_part)
                seconds = int((day_part - day) * SECONDS_PER_DAY)
                ts = int(offsets[year][day-1] + seconds)

                # INLINE-KEY
                key = ("%s,%12.12d"%(des, ts)).encode()

                # Increment the processed count now, since we may skip
                # further action.
                N += 1

                if 0 == (N % 100000):
                    now = datetime.datetime.now().timestamp()
                    proc = now - start
                    logging.info(f"    Completed {N} TLEs in %.1fs: {int(N/proc)}" % proc)

                # If we've already procssed this entry, just move along
                if not force and txn.get(key, db=self.db_apt):
                    skipped += 1
                    continue

                try:
                    n = float(line_2[52:63])
                    ecc = float('.'+line_2[26:33])
                    b = 42241.122 * n**(-2.0/3)
                except ValueError:
                    # TLE is non-conforming
                    non_conforming.append((line_1, line_2,))
                    continue

                period = (MINUTES_PER_DAY / n)
                apogee = b*(1+ecc)-6378
                perigee = b*(1-ecc)-6378

                if store_tles:
                    try:
                        # Find the remaining items from the TLE for funsies
                        ndot = float(line_1[33:43])

                        s = line_1[44:52]
                        nddot = float(s[0] + '.' + s[1:-2] + 'e' + s[-2:]) # f

                        s = line_1[53:61]
                        bstar = float(s[0] + '.' + s[1:-2] + 'e' + s[-2:]) # f
                        tle_num = int(line_1[64:68])  # i
                        inc = float(line_2[8:16])          # f
                        raan = float(line_2[17:25])        # f
                        argp = float(line_2[34:42])        # f
                        mean_anomaly = float(line_2[43:51])   # f
                        rev_num = int(line_2[63:68])  # i
                        pack_vals = [n, ndot, nddot, bstar, tle_num, inc,
                                     raan, ecc, argp, mean_anomaly, rev_num]
                        ### INLINE-CODE
                        tle_bytes = struct.pack(TLE_STRUCT_FMT, *pack_vals)
                        txn.put(key, tle_bytes, db=self.db_tle,
                                overwrite=True)
                    except Exception as e:
                        # some data quality issues here
                        bad_fmt += 1
                        continue

                # Pack the bytes
                ### INLINE-CODE
                apt_bytes = struct.pack(APT_STRUCT_FMT, apogee, perigee, period)
                txn.put(key, apt_bytes,
                        db=self.db_apt,
                        overwrite=True)

        logging.info(f"  Processed: {N}")
        logging.info(f"  Bad Data:  {len(non_conforming)}")
        logging.info(f"  Skipped:   {skipped}")

        txn.commit()

    def _build_scope(self):
        logging.info("Annotating Fragment Scope Table")
        txn = lmdb.Transaction(self.full_env, write=True)

        N = 0

        # Cursor to walk the DB
        cursor = txn.cursor(db=self.db_apt)

        # First key to initialize things:
        cursor.first()

        cur_des = None
        start_ts = None
        last_ts = None

        for key, _ in cursor:
            des, ts, = key.decode().split(',')
            ts = int(ts)

            if des == cur_des: last_ts = ts

            else:
                if cur_des:
                    reg = pack_scope(start=first_ts, end=last_ts)
                    txn.put(cur_des.encode(), reg,
                            db=self.db_scope,
                            overwrite=True)
                cur_des = des
                first_ts = ts
                last_ts = ts

                # Increment the processed count now, since we may skip
                # further action.
                N += 1

                if 0 == (N % 1000):
                    logging.info(f"  Put {N} entries, latest: {des}")

        if cur_des and start_ts is not None and end_ts is not None:
            while txn.delete(cur_des.encode()): pass
            reg = pack_scope(start=first_ts, end=last_ts)
            txn.put(cur_des.encode(), reg,
                    db=self.db_scope,
                    overwrite=True)

        txn.commit()
