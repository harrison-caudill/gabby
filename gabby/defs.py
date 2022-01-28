import datetime
import struct

# The keys of the DB are lexographically sorted byte strings.  While a
# negative sign does produce the desired results, it still leaves the
# problem that subsequent numbers which are later in time will be
# smaller in value if the number is negative.  Since the traditional
# timekeeping epoch is from 1970, and the first satellite was launched
# in 1957, a substantial number of TLEs end up with negative numbers
# for their timestamps which can result in reversal of values.  To
# account for this issue, we actually use our own internal epoch which
# is offset by exactly 13 years.  Anytime a time is created, we
# reference it to this date (UTC).  The integer value will be:
# int((<date> - <internal-epoch>).seconds)
EPOCH_YEAR = 1957
EPOCH_MONTH = 1
EPOCH_DAY = 1
EPOCH = datetime.datetime(EPOCH_YEAR, EPOCH_MONTH, EPOCH_DAY,
                          tzinfo=datetime.timezone.utc)

# The primary DB we use is the APT database which holds the most
# commonly-used items: Apogee(A), Perigee(P), and Period(T)
DB_NAME_APT = 'apt'

# The APT DB holds 3 float32 values: A, P, T
APT_STRUCT_FMT = "fff"

# The TLE database holds the remaining values from the original TLEs
DB_NAME_TLE = 'tle'

# TLE Format in the DB in order of appearance in the struct packing
# Value        fmt Description/units
# n             f  Mean Motion (revs/day)
# ndot          f  dn/dt
# nddot         f  d^2n/dt^2
# bstar         f  B* drag term
# tle_num       i  Observation number
# inc           f  Inclination (deg)
# raan          f  Right Ascension (deg)
# ecc           f  Eccentricity (dimensionless, 0-1)
# argp          f  Argument of Perigee
# mean_anomaly  f  Mean anomaly (phase in deg)
# rev_num       i  Number of orbits from epoch at time of observation
TLE_STRUCT_FMT = "ffffifffffi"

# The scope DB holds the times at which a spacecraft enters/exits
# scope in the DB.
DB_NAME_SCOPE = 'scope'

# Timestamp at which the fragment enters scope, then exits
SCOPE_STRUCT_FMT = 'ii'

# For whatever reason, lmdb needs to know the maximum number of DB's
# you can open, and you have to tell it the max number of rows, so we
# might as well have the constants easily accessible.
N_DBS = 3
DB_MAX_LEN = int(1*1000*1000*1000*1000)

# Every time we reference a date/time string, it will be in this
# format and will be assumed to be in UTC.
DATE_FMT = '%Y-%m-%d %H:%M:%S'

# This one is just handy to have around
SECONDS_IN_DAY = 24*60*60.0


def parse_date_ts(timestr):
    """Parses a time string and returns the internal integer timestamp.
    """
    return dt_to_ts(parse_date_d(timestr))

def parse_date_d(timestr):
    """Parses a time string and returns the datetime.
    """

    # repeated here to ensure explicit assignment of timezone
    fmt = DATE_FMT + '-%z'
    return datetime.datetime.strptime(timestr + '-+0000', fmt)

def fmt_key(ts=None, des=None):
    """Formats a timestamp and designator into a key for the DB.
    """
    assert(len(des))
    retval = ("%s,%12.12d"%(des, ts))
    return retval.encode()

def parse_key(key):
    """Deconstructs a DB key into the timestamp and designator.
    """
    if not isinstance(key, str): key = key.decode()
    des, ts, = key.split(',')
    return (des, int(ts),)

def unpack_apt(val):
    """Unpacks a binary string for an apt-table entry.
    """
    return struct.unpack(APT_STRUCT_FMT, val)

def pack_apt(A=None, P=None, T=None):
    """Packs values for an apt-table entry.
    """
    return struct.pack(APT_STRUCT_FMT, A,P,T)

def unpack_scope(val):
    """Unpacks a binary string for a scope-table entry.
    """
    return struct.unpack(SCOPE_STRUCT_FMT, val)

def pack_scope(start=None, end=None):
    """Packs values for a scope-table entry.
    """
    return struct.pack(SCOPE_STRUCT_FMT, start, end)

def unpack_tle(val):
    """Unpacks a binary string for a tle-table entry.
    """
    unpacked = struct.unpack(TLE_STRUCT_FMT, val)

    (n, ndot, nddot, bstar, tle_num, inc,
     raan, ecc, argp, mean_anomaly, rev_num) = unpacked

    return (n, ndot, nddot, bstar, tle_num, inc,
            raan, ecc, argp, mean_anomaly, rev_num)

def pack_tle(n=None,
             ndot=None,
             nddot=None,
             bstar=None,
             tle_num=None,
             inc=None,
             raan=None,
             ecc=None,
             argp=None,
             mean_anomaly=None,
             rev_num=None):
    """Packs values for a tle-table entry.
    """
    return struct.pack(TLE_STRUCT_FMT,
                       n, ndot, nddot, bstar, tle_num, inc,
                       raan, ecc, argp, mean_anomaly, rev_num)

def dt_to_ts(dt):
    """Converts a datetime object to an internal timestamp.

    The datetime object cannot be after the system epoch.
    """
    assert(dt >= EPOCH)
    return int(round((dt - EPOCH).total_seconds(), 0))

def ts_to_dt(ts):
    """Converts a timestamp to a datetime object.

    The ts CANNOT be negative.
    """
    assert(0 <= ts)
    return EPOCH + datetime.timedelta(seconds=ts)
