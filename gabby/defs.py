
SECONDS_IN_DAY = 24*60*60.0

DB_NAME_TLE = 'tle'
DB_NAME_APT = 'apt'
DB_NAME_SCOPE = 'scope'

DB_NAMES = [DB_NAME_TLE,
            DB_NAME_APT,
            DB_NAME_SCOPE]

DB_MAX_LEN = 1*1000*1000*1000*1000

def fmt_key(ts, des):
    return ("%s,%12.12d"%(des, ts)).encode()    

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

APT_STRUCT_FMT = "fff"
