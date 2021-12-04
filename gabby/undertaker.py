import datetime
import lmdb
import logging
import numpy as np
import os
import pickle
import subprocess
import struct
import sys
import tletools

from .defs import *


class Undertaker(object):
    """Brings data into the DB.
    """

    def __init__(self,
                 staging_dir=None,
                 db_path=None,
                 db_env=None):
        self.staging_dir = staging_dir
        self.db_path = db_path

        if db_env: self.db_env = db_env
        else: self.db_env = lmdb.Environment(self.db_path,
                                             max_dbs=len(DB_NAMES),
                                             map_size=int(DB_MAX_LEN))
        self.db_tle = self.db_env.open_db(DB_NAME_TLE.encode(), dupsort=False)
        self.db_gabby = self.db_env.open_db(DB_NAME_GABBY.encode(), dupsort=False)
        self.db_idx = self.db_env.open_db(DB_NAME_IDX.encode(), dupsort=False)
        self.db_scope = self.db_env.open_db(DB_NAME_SCOPE.encode(), dupsort=False)
        self.tle_pack_fmt = "ffffifffffi"

    def load_tlefile(self, path, store_tles=False):
        self._load_tlefile_impl(path, store_tles)
        self._annotate_gabby()
        self._annotate_index()

    def _load_tlefile_impl(self, path, store_tles):
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
        """

        logging.info("  Generating Time Lookup Table")
        # To make the timestamp computation go more quickly, we're
        # going to do it this way.  We DON'T keep track of leap
        # seconds, only leap days.
        offsets = np.zeros([100,366])
        for year in range(100):
            for day in range(366):
                o = 2000 if year < 57 else 1900
                try:
                    d = datetime.datetime(year=year+o,
                                          month=1,
                                          day=1,
                                          tzinfo=datetime.timezone.utc)
                    dt = datetime.timedelta(days=(day))
                    d += dt
                    offsets[year][day] = d.timestamp()
                except ValueError:
                    continue

        MINUTES_PER_DAY = 24*60
        SECONDS_PER_DAY = MINUTES_PER_DAY*60

        idx_bytes = struct.pack('h', 1)

        logging.info(f"  Importing data from {path}")
        txn = lmdb.Transaction(self.db_env, write=True)
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

                # Compute the timestamp (signed integer, seconds since epoch)
                year = int(line_1[18:20])
                day_part = float(line_1[20:32])
                day = int(day_part)
                seconds = int((day_part - day) * SECONDS_PER_DAY)
                ts = int(offsets[year][day-1] + seconds)

                key = ("%12.12d,%s"%(ts, des)).encode()
                rev_key = ("%s,%12.12d"%(des, ts)).encode()

                # Increment the processed count now, since we may skip
                # further action.
                N += 1

                if 0 == (N % 100000):
                    now = datetime.datetime.now().timestamp()
                    proc = now - start
                    logging.info(f"    Completed {N} TLEs in %.1fs: {int(N/proc)}" % proc)

                # If we've already procssed this entry, just move along
                if txn.get(key, db=self.db_gabby):
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
                        tle_bytes = struct.pack(self.tle_pack_fmt, *pack_vals)
                        txn.put(key, tle_bytes, db=self.db_tle,
                                overwrite=True)
                    except Exception as e:
                        # some data quality issues here
                        bad_fmt += 1
                        continue

                # Pack the bytes
                gabby_bytes = struct.pack('fff', apogee, perigee, period)
                txn.put(key, gabby_bytes,
                        db=self.db_gabby,
                        overwrite=True)

                txn.put(rev_key, gabby_bytes,
                        db=self.db_idx,
                        overwrite=True)

        logging.info(f"  Bad Data: {len(non_conforming)}")
        logging.info(f"  Skipped:  {skipped}")

        txn.commit()

    def _annotate_gabby(self):
        logging.info("Annotating Gabby Table (ts-des) at 1-day increments")
        txn = lmdb.Transaction(self.db_env, write=True)
        dt = datetime.timedelta(days=1)

        start = datetime.datetime.now().timestamp()
        N = 0

        reg = struct.pack('h', 1)

        cur = datetime.datetime(1957, 1, 1, 0, 0, 0,
                                tzinfo=datetime.timezone.utc)
        while cur.timestamp() < start:
            ts = int(cur.timestamp())
            cur += dt
            key = str(ts).encode()
            txn.put(key, reg,
                    overwrite=True)

            # Increment the processed count now, since we may skip
            # further action.
            N += 1

            if 0 == (N % 365):
                logging.info(f"  Put {N//365} years, latest: {ts}")

        txn.commit()

    def _annotate_index(self):
        logging.info("Annotating Fragment Index Table (des-ts)")
        txn = lmdb.Transaction(self.db_env, write=True)

        N = 0

        # Cursor to walk the DB
        cursor = txn.cursor(db=self.db_idx)

        # First key to initialize things:
        cursor.first()

        cur_des = None
        start_ts = None
        last_ts = None

        all_sats = {}
        start = {}
        end = {}

        for key, _ in cursor:
            des, ts, = key.decode().split(',')
            ts = int(ts)

            if des == cur_des: last_ts = ts

            else:
                if cur_des:
                    a = first_ts
                    b = last_ts
                    reg = struct.pack('ii', a, b)
                    txn.put(cur_des.encode(), reg,
                            db=self.db_scope,
                            overwrite=True)
                    start[cur_des] = a
                    end[cur_des] = b

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
            a = first_ts
            b = last_ts
            reg = struct.pack('ii', a, b)
            txn.put(cur_des.encode(), reg,
                    db=self.db_scope,
                    overwrite=True)
            start[cur_des] = a
            end[cur_des] = b

        txn.commit()

    # def _parse_date(self, s):
    #     return datetime.datetime.strptime(s, self.date_fmt)

    # def _parse_date(self, s):
    #     return datetime.datetime.strptime(s, self.date_fmt)

    # def _X(self):
    #     return np.arange(self.start, self.end+self.T_plot, self.T_plot)

    # def sample_data(self, des=None):
    #     """Samples the data and stores it.

    #     Because the server is unreliable, we need to save state as we
    #     go.  We're unlikely to do this more than a few thousand times,
    #     so the disk access shouldn't be a problem.
    #     """

    #     # If we're after a very specific target, we won't use like in
    #     # the comparison.
    #     use_like = False

    #     # Assume it's the primary
    #     if des is None:
    #         des = self.des
    #         use_like = True
    #     logging.info(f"Sampling Data for {des}")

    #     step = self.T_samp

    #     retval = self.load_raw_data(des)

    #     logging.info(f"  Integer Range: {self.start} -> {self.end} x {step}")
    #     logging.info(f"  Date Range: {self.start_d} -> {self.end_d}")

    #     for ts in range(self.start, self.end+1, step):
    #         if ts in retval: continue
    #         d = datetime.datetime.utcfromtimestamp(ts*SECONDS_IN_DAY)
    #         ds = d.strftime(self.date_fmt)
    #         logging.info(f"  Fetching data for date: {ts} ({ds})")
    #         retval[ts] = self._sample_day(ts, des, use_like=use_like)
    #         self.dump_raw_data(retval)

    #     return retval

    # def download_comparators(self):
    #     """
    #     """

    #     for comp in self.tgt['compartors'].strip().split(','):
    #         cur = self.load_raw_data(comp, prefix='comp')
    #         if len(cur): continue
    #         meta, data = gabby.get_data_for_range(comp,
    #                                               self.start, self.start+3,
    #                                               None,
    #                                               use_like=False)
    #         for datum in data:
    #             epoch = self._parse_date(datum['EPOCH'])
    #             edate = int(epoch.timestamp()/SECONDS_IN_DAY)
    #             if edate not in cur: cur[edate] = []
    #             cur[edate].append((datum['TLE_LINE1'], datum['TLE_LINE2']))

    #         self.dump_raw_data(comp, cur)

    # def dump_raw_data(self, des, data):
    #     with open(self.raw_data_path(des), 'w') as fd:
    #         try:
    #             dumps = json.dumps(data, indent=4, sort_keys=True)
    #         except Exception as e:
    #             pprint.pprint(data)
    #             raise e
    #         fd.write(dumps)

    # def load_raw_data(self, des, prefix='raw'):
    #     path = self.raw_data_path(des)
    #     if os.path.exists(path):
    #         logging.info(f"  Found data on disk at {path}")
    #         with open(path) as fd:
    #             raw = fd.read()
    #         try:
    #             tmp = json.loads(raw)
    #             retval = {}
    #             for k in tmp:
    #                 retval[int(k)] = tmp[k]
    #         except json.decoder.JSONDecodeError as e:
    #             logging.error(f"  Failed to read JSON: {raw}")
    #             return {}

    #         return retval
    #     return {}

    # def raw_data_path(self, des, prefix='raw'):
    #     """des != self.des
    #     that way we can pass in a comparator
    #     """
    #     return os.path.join(
    #         self.output_dir,
    #         self.des,
    #         f"{prefix}-{des}.json")

    # def xform_data_path(self):
    #     des = self.tgt['intldes']
    #     return os.path.join(
    #         self.output_dir,
    #         des,
    #         f"transformed-{des}.pickle")

    # def _sample_day(self, date, des, use_like=True):
    #     """Captures a single sample for each fragment on the given date
    #     """
    #     meta, data = self.get_data_for_range(des,
    #                                          date,
    #                                          date+1,
    #                                          None,
    #                                          use_like=use_like)
    #     retval = {}
    #     # FIXME: Do our own sort here.  First by fragment, then by
    #     #        date, and take the earliest sample...or
    #     #        latest...doesn't matter as long as we're consistent.
    #     for datum in data:
    #         # parse the epoch and make sure the date matches
    #         epoch = self._parse_date(datum['EPOCH'])
    #         edate = int(epoch.timestamp()/SECONDS_IN_DAY)
    #         if edate != date: continue

    #         # Doesn't matter which we get, so just clobber the old
    #         # one, who cares.
    #         tdes = datum['INTLDES']
    #         retval[tdes] = (datum['TLE_LINE1'], datum['TLE_LINE2'])

    #     return retval

    # def get_data_for_range(self, des, start, end, limit, use_like=True):

    #     logging.info(f"  Fetching data for time-range {start} -> {end}")
    #     assert(end > start)

    #     # looks like we need to do time ranges in days relative to
    #     # today...ugh...
    #     now = int(datetime.datetime.now().timestamp()/SECONDS_IN_DAY)
    #     start_d = now - start + 1
    #     end_d = now - end + 1

    #     gt = '%3E'
    #     lt = '%3C'

    #     time_restriction = f"{gt}now-{start_d},{lt}now-{end_d}"

    #     url = [
    #         "https://www.space-track.org",
    #         "basicspacedata", "query",
    #         "class", "tle",
    #         "EPOCH", time_restriction,
    #         ]
    #     if use_like: url += ["INTLDES", f"~~{des}~~",]
    #     else: url += ["INTLDES", f"{des}",]
    #     url += ["orderby", "EPOCH%20asc",]
    #     if limit: url += ["limit", f"{limit}",]
    #     url += [
    #         "format", "json",
    #         "metadata", "true",
    #         "emptyresult", "show"
    #         ]
    #     url = '/'.join(url)

    #     logging.info(f"  Loading url: {url}")

    #     return self._download(url)

    # def _download(self, url, path=None):
    #     logging.info(f"  Downloading URL {url}")
    #     cookie =  (f"Cookie: spacetrack_csrf_cookie={self.csrf};"
    #                + f" chocolatechip={self.chip}")
    #     cmd = [
    #         'curl',
    #         url,
    #         '-H', cookie,
    #         ]
    #     p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    #     raw = p.stdout.read()
    #     if not len(raw):
    #         return None, None
    #     if path:
    #         logging.info(f"  Saving to {path}")
    #         with open(path, 'wb') as fd:
    #             fd.write(raw)
    #     p.wait()
    #     try:
    #         retval = json.loads(raw)
    #     except json.decoder.JSONDecodeError as e:
    #         logging.error(f"  Failed to read JSON: {raw}")
    #         pprint.pprint(raw)
    #         raise e

    #     try:
    #         return retval['request_metadata'], retval['data']
    #     except TypeError as e:
    #         pprint.pprint(retval)
    #         raise e

    # def _process_piece(self, tles):
    #     retval = {}
    #     for tle_s in tles:
    #         retval[otime] = self._get_params(tle)

    #     return retval

    # def _get_params(self, tle):
    #     orb = tle.to_orbit()
    #     apogee = (orb.r_a - orb.attractor.R).to('km').value
    #     perigee = (orb.r_p - orb.attractor.R).to('km').value
    #     period = orb.period.to('min').value

    #     return (apogee, perigee, period,)

