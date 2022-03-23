#!/usr/bin/env python

import gc
import hashlib
import lmdb
import logging
import math
import matplotlib.pyplot as plt
import numpy as np

from .defs import *


class CloudDescriptor(object):
    """Holds values associated with APT/TLE with a timestamp index.

    There's a LOT of confusion that could happen WRT the
    size/shape/meaning/organization of data, so I'm standardizing the
    expression of tAPTN values into this wrapper object...it's also
    fewer things that have to be passed around.

    t=[[t0, t1, ..., tn, 0, ..., 0],
       [t0, t1, ..., tn, 0, ..., 0],
       ...
       [t0, t1, ..., tn, 0, ..., 0]],
    A=[[A0, A1, ..., An, 0, ..., 0],
       [A0, A1, ..., An, 0, ..., 0],
       ...
       [A0, A1, ..., An, 0, ..., 0]],
    P...
    T...,
    N = [N0, N1, ..., NL]

    """

    def __init__(self, fragments=None,
                 t=None,

                 # APT
                 A=None, P=None, T=None,

                 # TLE
                 n=None,
                 ndot=None,
                 nddot=None,
                 bstar=None,
                 tle_num=None,
                 inc=None,
                 raan=None,
                 ecc=None,
                 argp=None,
                 mean_anomaly=None,
                 rev_num=None,
                 N=None,
                 ):


        """Builds the descriptor.

        t: np.ndarray(L, M) int
        A/P/T/n/...: np.ndarray(L, M) float32
        N: np.ndarray(L) int

        L: Number of fragments
        M: (usually) Power of 2 >= max(Nx)
        """
        self.fragments = fragments
        self.t = t

        # APT
        self.A = A
        self.P = P
        self.T = T

        # TLE
        self.n            = n
        self.ndot         = ndot
        self.nddot        = nddot
        self.bstar        = bstar       
        self.tle_num      = tle_num
        self.inc          = inc
        self.raan         = raan
        self.ecc          = ecc
        self.argp         = argp
        self.mean_anomaly = mean_anomaly
        self.rev_num      = rev_num

        self.N = N
        self.M = t.shape[-1]
        self.L = len(t)
        assert(self.L == len(t) == len(A) == len(P) == len(T) == len(fragments))

    @property
    def pruned_A(self): return self.prune(self.A)

    @property
    def pruned_P(self): return self.prune(self.P)

    def prune(self, member):
        retval = []
        for i in range(self.L):
            retval.append(member[i][:self.N[i]])
        return retval

    def plot(self, path, idx, title=None,
             plt_energy=False,
             plt_sum=False,
             plt_mean=False):
        fig = plt.figure(figsize=(12, 8))

        if not title:
            title = f"Descriptor index {idx}"
        fig.suptitle(title, y=0.97, fontsize=25)

        ax_AP = fig.add_subplot(1, 1, 1)

        X = [ts_to_dt(t) for t in self.t[idx][:self.N[idx]]]

        lbls = []

        lbls += ax_AP.plot(X,
                           self.A[idx][:self.N[idx]],
                           label='A',
                           color='firebrick')

        lbls += ax_AP.plot(X,
                           self.P[idx][:self.N[idx]],
                           label='P',
                           color='dodgerblue')

        if plt_sum:
            lbls += ax_AP.plot(X,
                               (self.A[idx][:self.N[idx]] +
                                self.P[idx][:self.N[idx]]),
                               label='Sum',
                               color='purple')

        if plt_mean:
            lbls += ax_AP.plot(X,
                               (self.A[idx][:self.N[idx]] +
                                self.P[idx][:self.N[idx]])/2,
                               label='Mean',
                               color='green')

        # ax_T.plot(self.t[idx][:self.N[idx]],
        #           self.T[idx][:self.N[idx]],
        #           label='T',
        #           color='black')

        if plt_energy:
            ax_E = ax_AP.twinx()

            Re = 6371
            mu = 3.986004418e5
            A = self.A[idx][:self.N[idx]]
            P = self.P[idx][:self.N[idx]]
            Ra = P + Re
            a = (2*Re + A + P)/2

            E = mu/2 * (2/Ra - 1/a) - mu/Ra
            lbls += ax_E.plot(X,
                              E,
                              label='Energy',
                              color='black')

        ax_AP.legend(lbls, [l.get_label() for l in lbls], loc=1)

        fig.savefig(path)
        fig.clf()
        plt.close(fig)
        gc.collect()


class GabbyDB(object):
    """Database interface for the Gabby data

    We also stash some convenience methods in here for fetching common
    data.
    """

    def __init__(self, path=None, global_cache=None):
        """Does what every __init__ method does.

        path: path to the DB of satellites/fragments
        """

        logging.info(f"Loading DB at {path}")

        self.path = path
        self.global_cache = global_cache

        self.env = lmdb.Environment(path,
                                    max_dbs=N_DBS,
                                    map_size=DB_MAX_LEN)
        self.db_tle = self.env.open_db(DB_NAME_TLE.encode())
        self.db_apt = self.env.open_db(DB_NAME_APT.encode())
        self.db_scope = self.env.open_db(DB_NAME_SCOPE.encode())

    def find_daughter_fragments(self, base, txn=None):
        """Finds all fragments stemming from the given base.

        base: [<str>]
        returns [<str>, ...]

        This one is useful for finding daughter fragments after a
        collision.
        """

        assert(isinstance(base, list))

        retval = []
        commit, txn = self._txn(txn)
        cursor = txn.cursor(db=self.db_scope)
        cursor.first()

        for sat in base:
            prefix = srch = sat.encode()
            cursor.set_range(srch)
            for k, v in cursor:
                if not k.startswith(prefix): break
                retval.append(k.decode())

        if commit: txn.commit()
        return retval

    def _cache_name(self, fragments, apt=False):
        m = hashlib.sha256()
        m.update((','.join(sorted(fragments))).encode())
        prefix = ''
        prefix = 'apt-' if apt else 'tle-'
        return prefix + m.hexdigest()

    def load_tle(self, fragments, txn=None, cache=True):
        """Loads the TLE values from the DB.

        Returns a CloudDescriptor object
        """
        return self._load_data(fragments, txn=txn, cache=cache, apt=False)

    def load_apt(self, fragments, txn=None, cache=True):
        """Loads the APT values from the DB.

        Returns a CloudDescriptor object
        """
        assert(len(fragments))
        return self._load_data(fragments, txn=txn, cache=cache, apt=True)

    def apt_cache_names(self):
        return dict([(l, f"apt-raw-{l}") for l in 'dtAPT'])

    def tle_cache_names(self, field_off):
        fields = ['n',
                 'ndot',
                 'nddot',
                 'bstar',
                 'tle_num',
                 'inc',
                 'raan',
                 'ecc',
                 'argp',
                 'mean_anomaly',
                 'rev_num',
                 'N',]
        return f"tle-raw-d", f"tle-raw-t", f"tle-raw-{fields[field_off]}"

    def cache_apt(self, txn=None, force=False):
        # Initialize our main cursor
        commit, txn = self._txn(txn)
        cursor = txn.cursor(db=self.db_apt)

        cursor.first()

        logging.info(f"Loading/Caching APT in numpy arrays")

        names = self.apt_cache_names()
        if names['d'] in self.global_cache:
            if force:
                logging.info(f"  Overwriting cached values")
            else:
                logging.info(f"  APT already in cache")
                return [self.global_cache[n] for n in names]

        N = 0
        M = 1024
        d = np.array(['' for i in range(M)], dtype='<U8')
        t = np.zeros(M, dtype=np.int)
        A = np.zeros(M, dtype=np.float32)
        P = np.zeros(M, dtype=np.float32)
        T = np.zeros(M, dtype=np.float32)

        for k, v in cursor:
            d[N], t[N] = parse_key(k)
            A[N], P[N], T[N] = unpack_apt(v)
            assert(A[N] >= P[N])
            N += 1
            if 0 == N % 1e6:
                logging.info(f"Retreived {N} records")
            if N >= M:
                assert(N == M)
                M <<= 1
                d.resize(M)
                t.resize(M)
                A.resize(M)
                P.resize(M)
                T.resize(M)

        self.global_cache[names['d']] = d
        self.global_cache[names['t']] = t
        self.global_cache[names['A']] = A
        self.global_cache[names['P']] = P
        self.global_cache[names['T']] = T
        return [d, t, A, P, T,]

    def cache_tle_field(self, field_off, txn=None, force=False):

        # Initialize our main cursor
        commit, txn = self._txn(txn)
        cursor = txn.cursor(db=self.db_tle)

        cursor.first()

        logging.info(f"Loading/Caching TLE in numpy arrays")

        d_name, t_name, v_name = self.tle_cache_names(field_off)
        if v_name in self.global_cache:
            if force:
                logging.info(f"  Overwriting cached values")
            else:
                logging.info(f"  TLE field already in cache")
                return [self.global_cache[d_name],
                        self.global_cache[t_name],
                        self.global_cache[v_name],]

        N = 0
        M = 1024
        d = np.array(['' for i in range(M)], dtype='<U8')
        t = np.zeros(M, dtype=np.uint32)
        V = np.zeros(M, dtype=np.float32)

        name = self.tle_cache_names(field_off)

        for k, v in cursor:
            d[N], t[N] = parse_key(k)
            V[N] = unpack_tle(v)[field_off]
            N += 1
            if 0 == N % 1e6:
                logging.info(f"Retreived {N} records")
            if N >= M:
                assert(N == M)
                M <<= 1
                d.resize(M)
                t.resize(M)
                V.resize(M)

        self.global_cache[d_name] = d
        self.global_cache[t_name] = t
        self.global_cache[v_name] = V

        return d, t, V

    def _load_single_fragment(self, des, cursor, n_fields, apt=True):
        # Number of observations for this fragment
        n = 0

        # Seek to the beginning of the fragment in the table
        prefix = f"{des},".encode()
        cursor.set_range(prefix)
        off = len(prefix)

        # Stash the current round here
        M = 1024
        t = np.zeros(M, dtype=np.uint32)

        cur = [np.zeros(M, dtype=np.float32) for i in range(n_fields)]

        # Avoid the branch in our busy loop... cuz why not...
        i = 0
        if n_fields == 3:
            for k, v in cursor:
                if not k.startswith(prefix): break
                (cur[0][i],
                 cur[1][i],
                 cur[2][i],) = struct.unpack(APT_STRUCT_FMT, v)
                t[i] = int(k[off:])
                i += 1

                # We may need to expand our arrays
                if i >= M:
                    M *= 2
                    t.resize(M)
                    for j in range(n_fields): cur[j].resize(M)

        else:
            for k, v in cursor:
                if not k.startswith(prefix): break
                (cur[0][i],
                 cur[1][i],
                 cur[2][i],
                 cur[3][i],
                 cur[4][i],
                 cur[5][i],
                 cur[6][i],
                 cur[7][i],
                 cur[8][i],
                 cur[9][i],
                 cur[10][i],) = struct.unpack(TLE_STRUCT_FMT, v)
                t[i] = int(k[off:])
                i += 1

                # We may need to expand our arrays
                if i >= M:
                    M *= 2
                    t.resize(M)
                    for j in range(n_fields): cur[j].resize(M)

        return t, cur, i

    def _load_data(self, fragments,
                   txn=None,
                   cache=True,
                   apt=False):

        # numpy dimensions
        N = 0
        L = len(fragments)

        assert(0 < L)

        logging.info(f"Loading data for {L} fragments")

        # Get a few utility definitions out of the way for apt vs tle
        cache_name = self._cache_name(fragments, apt=apt)

        if self.global_cache and cache:
            if cache_name in self.global_cache:
                logging.info(f"  Loading data from cache")
                return self.global_cache[cache_name]

        # Initialize our main cursor
        commit, txn = self._txn(txn)
        db = self.db_apt if apt else self.db_tle
        cursor = txn.cursor(db=db)

        # Number of fields being recorded
        n_fields = APT_OFF_CAP if apt else TLE_OFF_CAP

        # Timestamps of each individual value: L arrays, each of N_i
        t_s = []

        # Values: n_fields arrays, each containing L arrays, of size N_i
        v_s = [[] for i in range(n_fields)]

        # Number of entries per fragment: L values
        N_s = np.zeros(L, dtype=np.uint32)

        # Load the individual fragments
        for i in range(L):
            des = fragments[i]
            t, v, n = self._load_single_fragment(des, cursor, n_fields, apt=apt)
            t_s.append(t)
            del t # decrement refcount for numpy resize
            N_s[i] = n
            for j in range(n_fields): v_s[j].append(v[j])
            del v # decrement refcount for numpy resize

            if i and 0 == i % 1000:
                logging.info(f"  Loaded {i} fragments")

        # Find our global N rounded to the nearest power of 2
        N = np.max(N_s)
        N = 1 << math.ceil(math.log(N, 2))

        # Resize all of the arrays to the newly-found N and concatenate
        for i in range(L):
            t_s[i].resize(N)
            for j in range(n_fields): v_s[j][i].resize(N)

        # Concatenate our final results
        t = np.concatenate(t_s).reshape((L, N))
        V = []
        for i in range(n_fields):
            V.append(np.concatenate(v_s[i]).reshape((L, N)))

        if commit: txn.commit()

        if apt:
            retval = CloudDescriptor(fragments=fragments,
                                     t=t, N=N_s,
                                     A=V[APT_OFF_APOGEE],
                                     P=V[APT_OFF_PERIGEE],
                                     T=V[APT_OFF_PERIOD])
        else:
            retval = CloudDescriptor(fragments=fragments,
                                     t=t, N=N_s,
                                     n=V[TLE_OFF_MEAN_MOTION],
                                     ndot=V[TLE_OFF_NDOT],
                                     nddot=V[TLE_OFF_NDDOT],
                                     bstar=V[TLE_OFF_BSTAR],
                                     tle_num=V[TLE_OFF_TLE_NUM],
                                     inc=V[TLE_OFF_INC],
                                     raan=V[TLE_OFF_RAAN],
                                     ecc=V[TLE_OFF_ECC],
                                     argp=V[TLE_OFF_ARGP],
                                     mean_anomaly=V[TLE_OFF_MEAN_ANOMALY],
                                     rev_num=V[TLE_OFF_REV_NUM],)

        if self.global_cache and cache:
            logging.info("  Saving results to cache")
            self.global_cache[cache_name] = retval

        return retval

    def txn(self, write=False):
        """Builds a new transaction on the db's environment.
        """
        return self._txn(None, write=write)[-1]

    def _txn(self, txn, write=False):
        """Internal convenience function to build a transaction as necessary.

        returns <do-commit>, txn
        """
        if txn: return False, txn
        return True, lmdb.Transaction(self.env, write=write)

    def load_scope(self, base, txn=None):
        """Loads the start and ending observation times for all daughters.

        base: [<str>, ...]
        returns: {<str>: start-ts}, {<str>: end-ts}
        """
        logging.info(f"Finding the scope of fragments")

        commit, txn = self._txn(txn)

        scope_start = {}
        scope_end = {}
        scope_cursor = txn.cursor(db=self.db_scope)
        scope_cursor.first()
        for des, scope in scope_cursor:
            des = des.decode()
            for prefix in base:
                if des.startswith(prefix):
                    start, end = unpack_scope(scope)
                    scope_start[des] = start
                    scope_end[des] = end

        if commit: txn.commit()

        return scope_start, scope_end

    def get_latest_apt(self, txn, des):
        """Returns the latest APT for the given designator
        """
        scope = txn.get(des.encode(), db=self.db_scope)
        start, end = unpack_scope(scope)

        key = fmt_key(end, des)
        tmp = txn.get(key, db=self.db_apt)
        a, p, t, = unpack_apt(tmp)
        return (a, p, t,)

    def lifetime_stats(self,
                       cutoff,
                       priors=None,
                       n_apogee=5,
                       n_perigee=5,
                       n_time=5,
                       weight_bins=True,
                       min_life=1.0):
        """Finds historical lifetime stats as a function of A/P.

        Examines all previous debris fragments associated with
        in-space collisions (ASAT and natural alike) and finds the
        probability histogram associated with decay time as a function
        of both apogee and perigee.  It stops at the cutoff date.

        Returns: hist

        n_apogee: number of apogee bins in the histogram
        n_perigee: number of perigee bins in the histogram
        n_time: number of time bins in the histogram
        weight_bins: use uniformly-weighted bins for apogee and perigee
        min_life: only consider fragments with a minimum lifetime (days)
        """

        if priors: prior_des = priors
        else: prior_des = self._prior_des()

        # Only need a read-only transaction for this
        txn = lmdb.Transaction(self.db_env, write=False)
        try:

            start_reg = {}
            end_reg = {}
            a_reg = {}
            p_reg = {}
            t_reg = {}

            cursor = txn.cursor(db=self.db_scope)

            cutoff_ts = cutoff.timestamp()

            # Load all the values into memory
            for satellite in prior_des:
                cursor.set_range(satellite.encode())
                for k, v in cursor:
                    frag = k.decode()
                    if satellite not in frag: break

                    start, end, = unpack_scope(v)

                    # We only consider fully-decayed fragments
                    if end > cutoff_ts: continue

                    life = end - start

                    # We ignore single-observation fragments
                    if life < min_life*24*3600: continue

                    start_reg[frag] = start
                    end_reg[frag] = end

                    apt_bytes = txn.get(fmt_key(start, frag), db=self.db_apt)
                    a, p, t, = unpack_apt(apt_bytes)

                    a_reg[frag] = a
                    p_reg[frag] = p
                    t_reg[frag] = life

            # Relinquish our handle and clear our pointer
            txn.commit()
            txn = None

            N = len(a_reg)
            assert(len(start_reg) == len(end_reg) == len(p_reg) == len(t_reg))
            logging.info(f"  Found {N} compliant fragments")

            # Find our ranges
            min_a = min([a_reg[k] for k in a_reg])
            max_a = max([a_reg[k] for k in a_reg])
            min_p = min([p_reg[k] for k in p_reg])
            max_p = max([p_reg[k] for k in p_reg])
            min_t = min([t_reg[k] for k in t_reg])
            max_t = max([t_reg[k] for k in t_reg])

            # Define our bins
            if weight_bins:
                A = sorted([a_reg[x] for x in a_reg])
                P = sorted([p_reg[x] for x in p_reg])
                T = sorted([t_reg[x] for x in t_reg])

                a_bins = np.zeros(n_apogee)
                p_bins = np.zeros(n_perigee)
                t_bins = np.zeros(n_time)

                a_idx = [int(x) for x in np.linspace(0, N, n_apogee+1)][:-1]
                p_idx = [int(x) for x in np.linspace(0, N, n_perigee+1)][:-1]
                t_idx = [int(x) for x in np.linspace(0, N, n_time+1)][:-1]

                for i in range(n_apogee): a_bins[i] = A[a_idx[i]]
                for i in range(n_perigee): p_bins[i] = P[p_idx[i]]
                for i in range(n_time): t_bins[i] = T[t_idx[i]]

            else:
                a_bins = np.linspace(min_a, max_a, n_apogee)
                p_bins = np.linspace(min_p, max_p, n_perigee)
                t_bins = np.linspace(min_t, max_t, n_time)

            logging.info(f"  Apogee:   {int(min_a)}-{int(max_a)}")
            bins_s = str([int(x) for x in a_bins])
            logging.info(f"  {bins_s}\n")
            logging.info(f"  Perigee:  {int(min_p)}-{int(max_p)}")
            bins_s = str([int(x) for x in p_bins])
            logging.info(f"  {bins_s}\n")
            logging.info(f"  Lifetime: {int(min_t)}-{int(max_t)}")
            bins_s = str([datetime.datetime.utcfromtimestamp(x)
                          for x in t_bins])

            # Assign counts to the bins
            decay_hist = np.zeros((n_apogee, n_perigee, n_time,))
            for frag in a_reg:
                a = a_reg[frag]
                p = p_reg[frag]
                t = t_reg[frag]

                i = np.digitize(a, a_bins)-1
                j = np.digitize(p, p_bins)-1
                k = np.digitize(t, t_bins)-1

                decay_hist[i][j][k] += 1

            # Normalize the distribution
            for i in range(n_apogee):
                for j in range(n_perigee):
                    s = sum(decay_hist[i][j])
                    if not s: continue
                    decay_hist[i][j] /= s

            return decay_hist

        finally:
            if txn: txn.commit()

