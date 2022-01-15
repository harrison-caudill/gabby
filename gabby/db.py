#!/usr/bin/env python

import lmdb
import logging
import numpy as np

from .defs import *

class GabbyDB(object):
    """Database interface for the Gabby data

    We also stash some convenience methods in here for fetching common
    data.
    """

    def __init__(self, path=None):
        """Does what every __init__ method does.

        path: path to the DB of satellites/fragments
        """

        self.path = path
        self.env = lmdb.Environment(path,
                                    max_dbs=N_DBS,
                                    map_size=DB_MAX_LEN)
        self.db_tle = self.env.open_db(DB_NAME_TLE.encode())
        self.db_apt = self.env.open_db(DB_NAME_APT.encode())
        self.db_scope = self.env.open_db(DB_NAME_SCOPE.encode())

    def find_daughter_fragments(self, base, txn=None):
        """Finds all fragments stemming from the given base.

        base: <str>
        returns [<str>, ...]

        This one is useful for finding daughter fragments after a
        collision.
        """
        retval = []
        commit, txn = self._txn(txn)
        cursor = txn.cursor(db=db_scope)
        for sat in base:
            prefix = sat.encode()
            cursor.set_range(prefix)
            for k, v in cursor:
                if not k.startswith(prefix): break
                retval.append(k.decode().split(',')[0])
        if commit: txn.commit()
        return retval

    def load_apt(self, fragments, txn=None):
        """Loads the APT values from the DB.
        
        Returns a tuple of np.arrays:
        
        There are L rows (one for each fragment in fragments) and a
        total of N columns where N is the maximum number of
        observations of any given fragment.  The array (of length L) N
        indicates the number of observations of that fragment.  Unless
        one of the fragments has exactly a power of two number of
        observations, then the length of the array will always be
        larger.  We dynamically expand the arrays as we go by powers
        of two, then just record the number of actual observations.
        
        (t=[[t0, t1, ..., tn, 0, ..., 0],
            [t0, t1, ..., tn, 0, ..., 0],
            ...
            [t0, t1, ..., tn, 0, ..., 0]],
         A=[[A0, A1, ..., An, 0, ..., 0],
            [A0, A1, ..., An, 0, ..., 0],
            ...
            [A0, A1, ..., An, 0, ..., 0]],
         P...
         T...,
         N = [N0, N1, ..., NL])
        """

        # Initialize our main cursor
        commit, txn = self._txn(txn)
        cursor = txn.cursor(db=self.db_apt)

        # numpy dimensions
        L = len(fragments)
        N = 1024

        # Keep track of the number of TLEs we find per fragment
        n_apt = np.zeros(L, dtype=np.int)

        # Use a regular python array, initially
        As = []
        Ps = []
        Ts = []
        ts = []

        logging.info(f"Loading APT for {L} fragments")

        for i in range(L):
            des = fragments[i]

            # Number of observations for this fragment
            n = 0

            # Seek to the beginning of the fragment in the table
            prefix = f"{des},".encode()
            cursor.set_range(prefix)
            off = len(prefix)

            # Stash the current round here
            M = N
            A = np.zeros(M, dtype=np.float32)
            P = np.zeros(M, dtype=np.float32)
            T = np.zeros(M, dtype=np.float32)
            t = np.zeros(M, dtype=np.int)

            # Loop through the DB
            j = 0
            for k, v in cursor:
                if not k.startswith(prefix): break
                A[j], P[j], T[j] = struct.unpack(APT_STRUCT_FMT, v)
                t[j] = int(k[off:])
                j += 1

                # We may need to expand our arrays
                if j >= M:
                    M *= 2
                    t.resize(M)
                    A.resize(M)
                    P.resize(M)
                    T.resize(M)

            # Update our global max
            N = max(N, j)

            # Store our local results
            n_apt[i] = j
            As.append(A)
            Ps.append(P)
            Ts.append(T)
            ts.append(t)

            # Numpy resize borks if there are other python references, so
            # we have to clear these.  The arrays above will still hold a
            # reference.  If we didn't do this here, later calls to resize
            # would fail.
            del A
            del P
            del T
            del t

            if 0 == i % 1000:
                logging.info(f"  Finished loading {i} fragments")

        # Resize all of the arrays to the newly-found N and concatenate
        for i in range(L):
            ts[i].resize(N)
            As[i].resize(N)
            Ps[i].resize(N)
            Ts[i].resize(N)

        # Concatenate our final results
        A = np.concatenate(As).reshape((L, N))
        P = np.concatenate(Ps).reshape((L, N))
        T = np.concatenate(Ts).reshape((L, N))
        t = np.concatenate(ts).reshape((L, N))

        retval = (t, A, P, T, n_apt)

        if commit: txn.commit()

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
        typ = "read/write" if write else "read only"
        logging.info(f"  Building new {typ} transaction")
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
        cursor = txn.cursor(db=self.db_scope)

        cursor.set_range(des.encode())
        key, scope = cursor.item()
        start, end = unpack_scope(scope)

        key = fmt_key(end, des)
        tmp = txn.get(key, db=self.db_apt)
        if not tmp:
            key = fmt_key(start, des)
            tmp = txn.get(key, db=self.db_apt)
        a, p, t, = unpack_apt(tmp)
        del cursor
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

            # for i in range(n_apogee):
            #     a_start = a_bins[i]
            #     logging.info(f"Decay Histogram (Apogee: {int(a_bins[i])})")
            #     logging.info(f"  Total: {sum(decay_hist[i])}")
            #     logging.info(f"  Total: {sum(sum(decay_hist[i]))}")
            #     bin_s = pprint.pformat(decay_hist[i]).replace('\n', '\n  ')
            #     logging.info(f"  {bin_s}")

            return decay_hist

        finally:
            if txn: txn.commit()

