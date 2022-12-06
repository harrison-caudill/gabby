import multiprocessing
import numpy as np
from sgp4.api import Satrec


class Simulator(object):
    """Runs SGP4 propagations to determine decay rates.

    It has been shown that computing MoralDecay by looking at the
    frequency histogram of historical observations provides a pretty
    reasonable approximation of historical events (i.e. there isn't
    that much variability once you look at apogee and perigee of the
    piece of debris).  This simulator's purpose is to use the SGP4
    propagation algorithm to determine decay rates without benefit of
    historical data.

    Decay rates are, generally, a function of three variables:
     * B*
     * Apogee
     * Perigee

     Since identical values of these three parameters yields an
     identical decay rate, we can produce a MoralDecay table by
     varying these four terms.  We do, in fact, have exactly the right
     structure already for three of them.  We have A' and P' as a
     funciton of Apogee bucket, then Perigee bucket, then a frequency
     histogram of values one might observe.  That frequency histogram
     of observed decay rates translates quite nicely to be the
     frequency histogram of B* values observed for collision
     fragments.

     Note about B*: B* has two meanings.  There's the definition, then
     there's how Space-Track.org uses it.  They incorporate
     variability of atmospheric density due to solar activity into
     that one value.  It's not a crazy thing to do, but it is
     definitely improper to call it a B* value.

     One thing to remember is that the TLEs provided by Space-Track
     will have apogee/perigee values that wander with the lunar cycle.
     If you run a propagator that accounts for the Earth's oblateness,
     then you find that the apogee and perigee vary by about the same
     amount as the difference in the Earth's radius.  You'll see
     minima and maxima when the argument of perigee crosses the poles.
     Presumably, their TLEs are accounting for polar crossings of
     argument of perigee.  Interestingly, the SGP4 propagator does not
     seem to account for the drift of the argument of perigee.  The
     test directory of this repo has 'oblate.py' which will build an
     image that pretty clearly illustrates the drift of argument of
     perigee.

     For the moment, this code-base will assume that no such drift
     occurs.  We understand that 15km can have a pretty pronounced
     effect on decay rates, but hope that the osciallatory nature of
     that drift can more-or-less even out over time making the
     long-term predictive results reasonably accurate.


     FIXME: We can do one major performance enhancement which is to
     only make observations around the time of apogee/perigee
     throughout the orbit instead of doing uniform sampling throughout
     one full orbit.  Analytically targeting that range would
     drastically reduce the required simulation time.  However....who
     cares...this codebase isn't exactly designed for performance,
     just designed to survive.

     One issue of simulating the decay rates is that if you have to
     wait a full orbit (up to 110 minutes) then you can have one full
     orbit of further decay before you record the decay amount.  This
     issue is especially pronounced at lower orbital altitudes.  More
     generally, though, if you pick 1 day as the time-step for the
     MoralDecay histogram, then 1 orbit can be circa 10% of the total
     time.

     There are a couple of potential solutions to this issue:

     1) We alter the initial orbital definition with a J2 perturbed
        keplerian orbit so that, after the simulation, the fragment
        ends at an apogee and again so it ends on a perigee.

     2) We take a linear approximation and scale the resulting value
        by the observed fraction of the orbit that corresponds to an
        apogee->apogee or perigee->perigee observation.

     Option 2 is way simpler so...yeah....that's the one we're gonna
     do...  Assuming that we're using 1 day and up to 110 minutes as
     the boundaries then we're talking about a linear extrapolation
     for up to 15% of the total time period.

     Nomenclature in the code:
     Ai/Pi: Index into the Apogee/Perigee bins
    """

    def __init__(self,
                 A_min=100,
                 A_max=4000,
                 N_A_bins=100,
                 P_min=100,
                 P_max=4000,
                 N_P_bins=100,
                 bstar_hist=None,
                 bstar_vals=None):
        """Builds a simulator.

        Because the MoralDecay object has a histogram of decay rates
        in each bucket, and decay rates are a function of bucket and
        bstar, we need to take in the bstar histogram.

        N_A_bins: <int> number of apogee bins
        N_P_bins: <int> number of perigee bins
        bstar_hist: sum(np[<float>]) = 1
        bstar_values: np[<float>]
        """
        assert(False)

    def _prep_simulations(self):
        """Prepares the variables for the multiprocessing simulation.

        """

        # Generate the arrays of julian dates for the propagation
        jds = np.zeros(N, dtype=int)
        frs = np.zeros(N, dtype=np.float32)
        for j in range(N):
            cur = ts[j].tolist()
            jd, fr = jday(cur.year,
                          cur.month,
                          cur.day,
                          cur.hour,
                          cur.minute,
                          cur.second)
            jds[j] = jd
            frs[j] = fr

        # Generate the arrays of julian dates for the propagation
        # throughout a single orbit.  In principle, we could speed
        # this up by having something like a lookup table of jds and
        # frs as a function of orbital period so that we don't do more
        # observations than are necessary when we observe the new
        # apogee/perigee, but who cares
        jds = np.zeros(N, dtype=int)
        frs = np.zeros(N, dtype=np.float32)
        for j in range(N):
            cur = ts[j].tolist()
            jd, fr = jday(cur.year,
                          cur.month,
                          cur.day,
                          cur.hour,
                          cur.minute,
                          cur.second)
            jds[j] = jd
            frs[j] = fr

        
        sat = Satrec.twoline2rv(*str(tle).split('\n')[1:])
        e, r, v = sat.sgp4_array(jds, frs)

        e, r, v = sat.sgp4_array(jds, frs)


    @classmethod
    def __run_sims_impl(As, Ps, Bs, epoch=None):
        """Runs a single simulation for group of Apogee/Perigee/Bstar/....

        This is a class method so that it can be readily employed by
        the multiprocessing library as the callback.

        As: np[<float>] Apogee values
        Ps: np[<float>] Perigee values
        Bs: np[<float>] BStar values
        """

        N = len(As)
        assert(len(As) == len(Ps) == len(Bs))

        for i in range(N):
            A = As[i]
            P = Ps[i]
            B = Bs[i]

            # Constants
            Re = 6371
            mu = 3.986004418e5 # km^3 / s^2
            one_day = datetime.timedelta(days=1).total_seconds()

            # Basic elliptical parameters
            a = (A + P + 2.0*Re) / 2.0
            c = a - P - Re
            b = (a**2 - c**2)**.5
            ecc = c / a
            Mu = 3.986 * 10**14
            Ts = 2 * np.pi * ((a*1000)**3 / Mu)**.5 # orbital period in seconds
            n = one_day / Ts # orbits per day

            # For simplicity, we'll do a single fine-grained
            # simulation for the entire day.  That's ~1500
            # observations per bin.

            start = epoch
            end = start + datetime.timedelta(seconds=1.1*T)
            dt = datetime.timedelta(minutes=1)
            ts = np.arange(start, end, dt)

            tle = gabby.TLE(n=n,
                            cat_number=i+1,
                            inc=inc,
                            raan=raan,
                            ecc=ecc,
                            argp=argp,
                            epoch=epoch)
            sat = Satrec.twoline2rv(*str(tle).split('\n')[1:])

            

        assert(False)

    def _run_sims(self, Ai, Pi):
        """Runs a single simulation for a given Apogee/Perigee Bucket.

        Returns np[A'], np[P']

        The returned arrays of values have indexes that correspond to
        the bstar values defined in the object proper.

        This one is little more than a convenience wrapper for the
        independent routine that can be used 
        """

        A = self.Apogee_values[Ai]
        P = self.Perigee_values[Pi]
        return self.__run_sim(self.Bstar_values,
                              self.Bstar_frequency,
                              A, P)

    def run(self, n_threads=1):
        """Runs the simulation with an optional number of threads.

        returns MoralDecay
        """

        Ap = np.linspace(Ap_min, Ap_max, N_A_bins)
        Pp = np.linspace(Pp_min, Pp_max, N_P_bins)

        moral_decay = np.zeros((2, # Apogee / Perigee
                                n_A_bins,
                                n_P_bins,
                                n_D_bins),
                               dtype=np.float32)

        for i in range(N_A_bins):
            for j in range(N_P_bins):
                # FIXME: Let's do midpoints here instead of range starts

                # Numpy float arrays of length N_D_bins
                Ad, Pd = self._run_sim(Ap[i], Pp[j])
                moral_decay[0][i][j] = Ad
                moral_decay[1][i][j] = Pd
                Ad_min = min(Ad_min, min(Ad))
                Pd_min = min(Pd_min, min(Pd))

        return MoralDecay(moral_decay,
                          Ap_min, Ap_max, dAp, Ad_min, Ad_max, dAd,
                          Pp_min, Pp_max, dPp, Pd_min, Pd_max, dPd)




>>> s = '1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991'
>>> t = '2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482'
>>> satellite = Satrec.twoline2rv(s, t)
>>>
>>> jd, fr = 2458827, 0.362605
>>> e, r, v = satellite.sgp4(jd, fr)
>>> e
0
>>> print(r)  # True Equator Mean Equinox position (km)
(-6102.44..., -986.33..., -2820.31...)
>>> print(v)  # True Equator Mean Equinox velocity (km/s)
(-1.45..., -5.52..., 5.10...)


    def decay_rates(self, apt, resampled, deriv, drag=None):
        """Bins the decay rate distributions.

        retval: [A'=0,P'=1][A-bin][P-bin][D-bin] = d(A/P)/dt

        dt is defined in the call to derivatives() defaulting to 1 day.
        """

        logging.info(f"Reticulating Splines") # Tee hee
        if drag: drag.normalize_decay_rates(deriv)

        deriv.A = np.where(deriv.A > 0, 0, deriv.A)
        deriv.P = np.where(deriv.P > 0, 0, deriv.P)

        assert(np.all(resampled.N == deriv.N))
        assert(np.all(resampled.t == deriv.t))

        sec = self.cfg['stats']
        n_A_bins = sec.getint('n-apogee-bins')
        n_D_bins = sec.getint('n-deriv-bins')

        kwargs = {
            'min_val': sec.getint('min-apogee'),
            'max_val': sec.getint('max-apogee'),
            'low_clip': sec.getfloat('apogee-deriv-low-prune'),
            'high_clip': sec.getfloat('apogee-deriv-high-prune'),
            'n_p_bins': n_A_bins,
            'n_D_bins': n_D_bins,
            'key': 'A',
            }
        dig_A = self.__concat_and_digitize(resampled, deriv, **kwargs)
        (Ap_min, Ap_max, dAp, Ap,
         Ad_min, Ad_max, dAd, Ad,) = dig_A

        n_P_bins = sec.getint('n-perigee-bins')
        kwargs = {
            'min_val': sec.getint('min-perigee'),
            'max_val': sec.getint('max-perigee'),
            'low_clip': sec.getfloat('perigee-deriv-low-prune'),
            'high_clip': sec.getfloat('perigee-deriv-high-prune'),
            'n_p_bins': n_P_bins,
            'n_D_bins': n_D_bins,
            'key': 'P',
            }
        dig_P = self.__concat_and_digitize(resampled, deriv, **kwargs)
        (Pp_min, Pp_max, dPp, Pp,
         Pd_min, Pd_max, dPd, Pd,) = dig_P

        logging.info(f"Quantifying Moral Decay")
        index, univ_A, univ_P, = self.__universalize(Ap, Ad, Pp, Pd,
                                                     n_A_bins,
                                                     n_P_bins,
                                                     n_D_bins)
        moral_decay = self.__bin_universalized_array(index,
                                                     n_A_bins,
                                                     n_P_bins,
                                                     n_D_bins,)
        
        # logging.info(f"Slowly Quantifying Moral Decay")
        # start = datetime.datetime.now().timestamp()

        # (Ap_min, Ap_max, dAp, Ap,
        #  Ad_min, Ad_max, dAd, Ad,
        #  Pp_min, Pp_max, dPp, Pp,
        #  Pd_min, Pd_max, dPd, Pd,
        #  moral_decay) = self._slow_decay_rates(apt, resampled, deriv)
        # end = datetime.datetime.now().timestamp()
        # logging.info(f"    Slow Version took: {int((end-start)*1000)}ms")

        # FIXME: Any normalization steps for things like B* compared
        # to mean would happen at this stage.

        return MoralDecay(moral_decay,
                          Ap_min, Ap_max, dAp, Ad_min, Ad_max, dAd,
                          Pp_min, Pp_max, dPp, Pd_min, Pd_max, dPd)
