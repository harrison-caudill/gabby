import astropy
import datetime
import lmdb
import logging
import numpy as np
import os
import pickle
import pprint
import struct
import sys

from .defs import *


def mkdir_p(path):
    """Emulates the functionality of mkdir -p
    """
    if not len(path): return
    if not os.path.isdir(path):
        logging.debug('Creating Directory: %s' % path)
        os.makedirs(path)

def setup_logging(log_file='output/gab.log', log_level='info'):
    """Gets the logging infrastructure up and running.

    log_file: either relative or absolute path to the log file.
    log_level: <str> representing the log level
    """

    # Let's make sure it has a directory and that we can write to that
    # file.
    log_dir = os.path.dirname(log_file)
    if not len(log_dir): log_dir = '.'
    if not os.path.exists(log_dir):
        chirp.mkdir_p(os.path.dirname(log_file))
    if not os.path.isdir(log_dir):
        msg = ("Specified logging directory is " +
               "not a directory: %s" % log_dir)
        logging.critical(msg)
        raise ConfigurationException(msg)
    try:
        fd = open(log_file, 'w')
        fd.close()
    except:
        msg = "Failed to open log file for writing: %s" % log_file
        logging.critical(msg)
        raise ConfigurationException(msg)

    if log_level.lower() not in ['critical',
                                 'error',
                                 'warning',
                                 'info',
                                 'debug']:
        msg = "Invalid log level: %s" % log_level
        logging.critical(msg)
        raise ConfigurationException(msg)
    log_level = log_level.upper()

    # Set up the logging facility
    logging.basicConfig(level=log_level, filename=log_file, filemode='w')
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    logging.getLogger().addHandler(ch)


def load_scope(txn, db_scope, target_des):
    logging.info(f"  Finding the scope of all fragments")
    scope_start = {}
    scope_end = {}
    scope_cursor = txn.cursor(db=db_scope)
    scope_cursor.first()
    for des, scope in scope_cursor:
        des = des.decode()
        for prefix in target_des:
            if des.startswith(prefix):
                start, end = unpack_scope(scope)
                scope_start[des] = start
                scope_end[des] = end
    return scope_start, scope_end


def load_dbs(obj, full_path, frag_path):
    full_env = lmdb.Environment(full_path,
                                max_dbs=N_DBS,
                                map_size=DB_MAX_LEN)
    obj.full_env = full_env
    obj.full_tle = obj.full_env.open_db(DB_NAME_TLE.encode())
    obj.full_apt = obj.full_env.open_db(DB_NAME_APT.encode())
    obj.full_scope = obj.full_env.open_db(DB_NAME_SCOPE.encode())

    if frag_path:
        frag_env = lmdb.Environment(frag_path,
                                    max_dbs=N_DBS,
                                    map_size=DB_MAX_LEN)
        obj.frag_env = frag_env
        obj.frag_tle = obj.frag_env.open_db(DB_NAME_TLE.encode())
        obj.frag_apt = obj.frag_env.open_db(DB_NAME_APT.encode())
        obj.frag_scope = obj.frag_env.open_db(DB_NAME_SCOPE.encode())
    else:
        obj.frag_env = None
        obj.frag_tle = None
        obj.frag_apt = None
        obj.frag_scope = None

def find_daughter_fragments(base, txn, db_scope):
    retval = []
    cursor = txn.cursor(db=db_scope)
    for sat in base:
        prefix = sat.encode()
        cursor.set_range(prefix)
        for k, v in cursor:
            if not k.startswith(prefix): break
            retval.append(k.decode().split(',')[0])
    return retval


def load_apt(fragments, txn, db_apt, cache_dir=None):
    """Loads the APT values from the DB.

    Returns a tuple of np.arrays:

    There are L rows (one for each fragment in fragments) and a total
    of N columns where N is the maximum number of observations of any
    given fragment.  The array (of length L) N indicates the number of
    observations of that fragment.

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

    if cache_dir:
        cache_data_path = os.path.join(cache_dir, "apt_data.np")
        if os.path.exists(cache_data_path):
            with open(cache_data_path, 'rb') as fd:
                logging.info(f"Loading APT from: {cache_data_path}")
                t = np.load(fd)
                A = np.load(fd)
                P = np.load(fd)
                T = np.load(fd)
                n_apt = np.load(fd)
                return t, A, P, T, n_apt

    # Initialize our main cursor
    cursor = txn.cursor(db=db_apt)

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

    # Cache the values
    if cache_dir:
        cache_data_path = os.path.join(cache_dir, "apt_data.np")
        logging.info(f"Saving APT data to cache: {cache_data_path}")
        with open(cache_data_path, 'wb') as fd:
            np.save(fd, t)
            np.save(fd, A)
            np.save(fd, P)
            np.save(fd, T)
            np.save(fd, n_apt)

            meta = {
                'fragments': fragments,
                }

        cache_meta_path = os.path.join(cache_dir, "apt_meta.pickle")
        with open(cache_meta_path, 'wb') as fd:
            pickle.dump(meta, fd)

    return retval

def time_command(f, msg):
    def retval(*args, **kwargs):
        logging.info("%s"%msg)
        start = datetime.datetime.now().timestamp()
        f(*args, **kwargs)
        end = datetime.datetime.now().timestamp()
        ms = int((end - start)*1000)
        logging.info("  Call completed in {ms}ms")

def parse_date(timestr):
    """Uses the package's preferred format to parse a date into UTC.
    """

    # Ugly, but it works
    timestr += '-+0000'
    fmt = DATE_FMT + '-%z'
    return datetime.datetime.strptime(timestr, fmt)

def plot_apt(frag, tapt, path):
    fig = plt.figure(figsize=(12, 8))
    fig.set_dpi(300)
    if frag: fig.suptitle(f"Decay profile: {frag}")
    else: fig.suptitle(f"Decay profile")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Time Since Launch (days)')
    ax.set_ylabel('Orbital Altitude (km)')

    t, A, P, T = tapt
    t -= t[0]
    plt_A = ax.plot(t, A, color='firebrick', label='Apogee')
    plt_P = ax.plot(t, P, color='dodgerblue', label='Perigee')
    ax.legend(loc=1)

    ax = ax.twinx()
    plt_T = ax.plot(t, T/60.0, color='black', label='Period')
    ax.set_ylabel('Period (minutes)')
    ax.legend(loc=2)

    plts = plt_A + plt_P + plt_T
    labs = [p.get_label() for p in plts]
    ax.legend(plts, labs, loc=1)

    fig.savefig(path)

def keplerian_period(A, P):
        Re = (astropy.constants.R_earth/1000.0).value
        RA = A+Re
        RP = P+Re
        e = (RA-RP) / (RA+RP)

        # These are all in meters
        a = 1000*(RA+RP)/2
        mu = astropy.constants.GM_earth.value
        T = 2 * np.pi * (a**3/mu)**.5 / 60.0

        return T
