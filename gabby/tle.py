
import datetime
import math

class TLE(object):
    """Internal representation of a TLE.

    TLEs are a bit of a mess and not terribly well supported.  There
    is the tletools package, but it will only do TLE-text ->
    TLE-object -> poliastro Orbit, not the reverse.  Consequently, if
    we want to export TLEs, we still need a home-grown method.
    """

    def __init__(self,
                 cat_number=None,
                 classification='U',
                 launch_year=2017,
                 launch_number=1,
                 launch_piece='A',
                 epoch=None,
                 n=15,
                 ndot=-1.0e-5,
                 nddot=0.0,
                 bstar=1e-5,
                 tle_num=1,
                 inc=51.6,
                 raan=0.0,
                 ecc=1.0e-6, # Doesn't like 0
                 argp=0.0,
                 mean_anomaly=0.0,
                 rev_num=1,
                 name=None):
        """TLE Object for exporting and manipulating.

        cat_number: <int>, Catalog Number
        classification: <str>, 'U', 'C', or 'S'

        International Designator:
        launch_year: <int> all 4 numbers please
        launch_number: <int> which launch of that year?
        launch_piece: <str> A-Z (presumably???)

        epoch: <datetime.datetime>
        n: <float> mean motion (revolutions per day)
        ndot: <float> (d/dt) mean motion (aka Ballistic Coefficient)
        nddot: <float> (d^2/dt^2) mean motion
        bstar: <float> B* drag term (aka Radiation Pressure Coefficient)
        tle_num: <int> Version number for this object's TLE
        inc: <float> Orbital inclination (deg)
        raan: <float> Right Ascention of the Ascending Node (deg)
        ecc: <float> eccentricity of the orbit
        argp: <float> Argument of the periapsis
        mean_anomaly: <float> Mean Anomaly (deg)
        rev_num: <int> 1-offset revolution number at epoch (usually 1)


        Classification can be Unclassified (U), Classified (C), or
        Secret (S).

        Who knows what's going on with the piece-of-launch
        portion...just use 'A'
        """

        if (not isinstance(cat_number, int)
            or cat_number <=0
            or cat_number >= 10**5):
            raise ValueError("Invalid Satellite Number: %s" % cat_number)

        if epoch is None:
            epoch = datetime.datetime.now()

        if name is None:
            name = 'test-%d' % cat_number

        self.cat_number = cat_number
        self.classification = classification
        assert('U' == classification)

        self.intldes = f"%s%3.3d%s" % (
            ('%4.4d'%(launch_year))[-2:], launch_number, launch_piece)

        self.launch_year = launch_year
        self.launch_number = launch_number
        self.launch_piece = launch_piece
        self.epoch = epoch
        self.n = n
        self.ndot = ndot
        self.nddot = nddot
        self.bstar = bstar
        self.tle_num = tle_num
        self.inclination_deg = inc
        self.right_ascention_ascending_deg = raan
        self.eccentricity = ecc
        self.argument_of_perigee_deg = argp
        self.mean_anomaly_deg = mean_anomaly
        self.rev_num = rev_num
        self.name = name

    @classmethod
    def from_poli(cls,
                  poli=None,
                  cat_number=None,
                  classification=None,
                  launch_year=None,
                  launch_number=None,
                  launch_piece=None,
                  bstar=None,
                  tle_num=None,
                  rev_num=None,
                  name=None):
        # FIXME: implement as needed
        assert(False)

    @classmethod
    def from_str(cls, s):
        """Parses a TLE from a string using tletools.

        We don't actually parse the raw string, instead we use
        tletools to parse it, then grab the numbers from there.
        tletools does not encode, however, so we have to do that
        ourselves.
        """
        tle = tletools.TLE.from_lines(*s.strip().splitlines())

        # separate out the international designator
        launch_year = int(str(tle.int_desig)[:2])
        launch_num = int(str(tle.int_desig)[2:5])
        launch_piece = str(tle.int_desig)[-1]

        # Compute the epoch datetime object
        yr = tle.epoch_year + 1900
        if yr < 1957: yr += 100
        day = int(math.floor(tle.epoch_day))
        rem = tle.epoch_day - day
        hrs = int(rem * 24)
        rem -= hrs/24.0
        mins = int(rem * 60.0)
        rem -= mins/60.0
        secs = int(rem * 60.0)
        epoch_s = '%d %3.3d %2.2d %2.2d %2.2d' % (yr, day, hrs, mins, secs)
        epoch = datetime.datetime.strptime(epoch_s, '%Y %j %H %M %S')

        return TLE(cat_number=tle.norad,
                   classification=tle.classification,
                   launch_year=launch_year,
                   launch_number=launch_num,
                   launch_piece=launch_piece,
                   epoch=epoch,
                   n=tle.n,
                   ndot=tle.dn_o2,
                   nddot=tle.ddn_o6,
                   bstar=tle.bstar,
                   tle_num=tle.set_num,
                   inc=tle.inc,
                   raan=tle.raan,
                   ecc=tle.ecc,
                   argp=tle.argp,
                   mean_anomaly=tle.M,
                   rev_num=tle.rev_num,
                   name=tle.name)

    def _ndot_stringify(self, n, l=12):
        n_int = int(n)
        n_frac = ('%14.14f' % (abs(n) % 1))[2:]
        n_sign = ' ' if n >= 0 else '-'
        if 0 == n_int:
            n_str = ('%s.%s' % (n_sign, n_frac))[:10]
        else:
            n_str = ('%s%d.%s' % (n_sign, n_int, n_frac))[:10]
        return n_str[:l]

    def _nddot_stringify(self, n, l=8):
        if 0 == n:
            return " 00000-0"

        exp = int(-1*math.log(abs(n), 10))
        n *= 10**(exp + 10)
        exp *= -1
        coeff = str(n)[:l-2]
        return coeff + ('%d' % exp)

    def _checksum(self, s):
        tot = 0
        for c in s:
            if c.isdigit():
                tot += int(c)
            elif '-' == c:
                tot += 1
        return tot % 10

    def _3_digify(self, n, l):
        a = '%3d' % n
        b = ('%*.*f' % (l+1, l+1, (n % 1)))[2:]
        return ('%s.%s' % (a, b))[:l]

    def __str__(self):

        e = self.epoch
        epoch_day = e.timetuple().tm_yday
        day_usec = (0.0 # force float
                    + e.hour * 60 * 60 * 1e6
                    + e.minute * 60 * 1e6
                    + e.second * 1e6
                    + e.microsecond)
        day_fraction = day_usec / (1.0e6 * 60 * 60 * 24)

        line_1 = ''.join([

            '1', # Line Number 1
            ' ',

            ('%5.5d'%self.cat_number)[:5], # satellite number
            self.classification,
            ' ',

            '%2.2d' % (self.launch_year % 100),
            '%-3.3d' % self.launch_number,
            '%-3.3s' % self.launch_piece,
            ' ',

            str(self.epoch.year % 100),
            '%3.3d.%8.8d' % (epoch_day, day_fraction * 1e8),
            ' ',

            self._ndot_stringify(self.ndot / 2.0, l=12),
            ' ',

            self._nddot_stringify(self.nddot / 6.0, l=8),
            ' ',

            self._nddot_stringify(self.bstar, l=8),
            ' ',

            '0',
            ' ',

            '%4d' % self.tle_num,
            ])
        line_1 = line_1 + str(self._checksum(line_1))

        line_2 = ''.join([

            '2',
            ' ',

            ('%5.5d'%self.cat_number)[:5], # satellite number
            ' ',

            self._3_digify(self.inclination_deg, 8),
            ' ',

            self._3_digify(self.right_ascention_ascending_deg, 8),
            ' ',

            ('%10.10f' % self.eccentricity)[2:9],
            ' ',

            self._3_digify(self.argument_of_perigee_deg, 8),
            ' ',

            self._3_digify(self.mean_anomaly_deg, 8),
            ' ',

            ('%11.11f' % self.n)[:11],
            ('%10d' % self.rev_num)[-5:]
            ])
        line_2 = line_2 + str(self._checksum(line_2))

        return '\n'.join(['%-24s' % self.name, line_1, line_2])
