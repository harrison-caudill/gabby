
SECONDS_IN_DAY = 24*60*60.0

DB_NAME_TLE = 'tle'
DB_NAME_IDX = 'idx'
DB_NAME_GABBY = 'gabby'
DB_NAME_SCOPE = 'designator_scope'

DB_NAMES = [DB_NAME_TLE,
            DB_NAME_IDX,
            DB_NAME_GABBY,
            DB_NAME_SCOPE]

DB_MAX_LEN = 1*1000*1000*1000*1000

def fmt_key(ts, des):
    return ("%12.12d,%s"%(ts, des)).encode()

def fmt_rev(ts, des):
    return ("%s,%12.12d"%(des, ts)).encode()
