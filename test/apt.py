#!/usr/bin/env python3

import argparse
import datetime
import functools
import logging
import matplotlib.pyplot as plt
import numpy as np
import pprint
from sgp4.api import Satrec
from sgp4.api import jday
import sys


"""See the help message for more information.
"""


# Earth constants -- km, kg, s
Re_max = 6378 # km
Re_avg = 6371 # km
Me = 5.972e24 # kg
J2e =  1.7553e10 # km^5/s^2

#https://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1964PASJ...16..263K&defaultprint=YES&filetype=.pdf
J4e = -1.65597e-6

G = 6.6743e-20 # km^3 / kg / s
mu = 3.986004418e5 # km^3 / s^2
one_day = datetime.timedelta(days=1).total_seconds()


def snarf_tles(tlefile):
    """Reads and parses a file of TLEs
    """

    logging.info(f"  Loading TLEs from {tlefile}")

    # Read the raw TLEs.  The result will be an array of dimensions
    # Lx2 where L is the number of TLEs and then two lines for each.
    if not tlefile: tlefile = f"{des}.txt"
    with open(tlefile) as fd: raw = fd.read()
    tles = raw.strip().split('\n')
    tles = [tles[i:i+2] for i in range(0, len(tles), 2)]
    L = len(tles)

    # We can't assume that the TLEs are sorted temporally, so we'll do
    # that now.
    def __my_cmp__(a, b):
        A = float(a[0][18:32])
        B = float(b[0][18:32])
        if A == B: return 0
        return -1 if A < B else 1

    # TLEs are now sorted by the epoch
    tles = sorted(tles, key=functools.cmp_to_key(__my_cmp__))

    # Raw TLE Outputs.  These values are pulled from the TLE and
    # nothing else.  No computations, augmentations, or
    # interpretations happen.
    E_t = []                               # Datetime of epoch from TLE
    raan_t = np.zeros(L, dtype=np.float32) # Right ascension from TLE
    argp_t = np.zeros(L, dtype=np.float32) # Argument of Perigee from TLE
    raan_t = np.zeros(L, dtype=np.float32) # Right ascension from keplerian
    argp_t = np.zeros(L, dtype=np.float32) # Argument of Perigee from keplerian
    ecc_t = np.zeros(L, dtype=np.float32)  # eccentricity
    n_t = np.zeros(L, dtype=np.float32)    # mean motion
    inc_t = np.zeros(L, dtype=np.float32)  # inclination                       

    for i in range(L):
        tle = tles[i]

        # Find the basic orbital paramters in the TLE
        ecc_t[i] = float('.'+tle[1][26:33])
        n_t[i] = float(tle[1][52:63])
        raan_t[i] = float(tle[1][17:25])
        argp_t[i] = float(tle[1][34:42])
        inc_t[i] = float(tle[1][8:16])
        year = int(tle[0][18:20])
        year = year + 1900 if year > 57 else year + 2000
        days = float(tle[0][20:32])
        epoch = (datetime.datetime(year, 1, 1)+datetime.timedelta(days=days))
        E_t.append(epoch)

    logging.info(f"  Successfully parsed {L} TLEs")
    return (L, tles, E_t, raan_t, argp_t, raan_t, argp_t, ecc_t, n_t,
            inc_t,)


def propagate_one_orbit(tle, epoch, dt_min):
    """Uses SGP4 to propagate the TLE by exactly one orbit (ish).
    """

    # python-sgp4 representation of a satellite
    sat = Satrec.twoline2rv(*tle)

    # The mean motion in the sgp4 package appears to be a mean of
    # mean-motion as days/orbit rather than orbits/day (so inverse of
    # some averaged value from propagation of the TLE I guess???)
    T = one_day*sat.nm

    # Find the date-ranges of the simulation
    start = epoch
    end = start + datetime.timedelta(seconds=1.1*T)
    dt = datetime.timedelta(minutes=dt_min)
    ts = np.arange(start, end, dt)
    N = len(ts)

    # Generate the arrays of julian dates for the propagation
    jds = np.zeros(N, dtype=np.int)
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

    # r is an array of N xyz coordinates
    e, r, v = sat.sgp4_array(jds, frs)

    return ts, N, e, r, v


def moon_2008():
    # Full Moons
    full = datetime.datetime(2008, 1, 22, 13, 35, 0)
    T_moon = datetime.timedelta(days=29.5)
    E_moon = []
    while(full < datetime.datetime(2009, 1, 1)):
        E_moon.append(full)
        full += T_moon
    return E_moon


def cmd_sim_apt(des, tlefile=None, dt_min=1, output=None):
    """Propagates a single fragment and compares to observations.

    Runs an SGP4 propagation on a series of fragment observations and
    compares the Apogee and Perigee of the simulation vs the assumed
    values from keplerian orbits.

    # The command-line for gathering the TLEs from the space-track.org
    # tle files (preserved for convenience):
    grep -A 1 ${des} tle2008.txt \
        | grep -v -- -- \
        | sed -e 's/\\//' \
        > ~/dev/gabby/test/${des}.txt
    """

    if not output: output = f"sim_apt-{des}.png"

    logging.info(f"=== Executing APT Simulation ===")
    logging.info(f"  Designator: {des}")
    logging.info(f"  Source:     {tlefile}")
    logging.info(f"  Time step:  {dt_min}")
    logging.info(f"  Output:     {output}")
    logging.info(f"")




    (L,       # Number of TLEs parsed
     tles,    # TLEs: [[<line-1>,<line-2>], ...]
     E_t,     # Datetime of epoch from TLE        
     raan_t,  # Right ascension from TLE          
     argp_t,  # Argument of Perigee from TLE      
     raan_t,  # Right ascension from keplerian    
     argp_t,  # Argument of Perigee from keplerian
     ecc_t,   # eccentricity                      
     n_t,     # mean motion                       
     inc_t,   # inclination                       
     ) = snarf_tles(tlefile)

    # Output Values from the TLE/Keplerian
    Ap_k = np.zeros(L, dtype=np.float32)   # Apogee values from keplerian
    Pp_k = np.zeros(L, dtype=np.float32)   # Perigee values from keplerian
    Au_k = np.zeros(L, dtype=np.float32)   # Energy at apogee
    Pu_k = np.zeros(L, dtype=np.float32)   # Energy at perigee
    dadt_k = np.zeros(L, dtype=np.float32) # dArgP/dt in deg/day
    drdt_k = np.zeros(L, dtype=np.float32) # dRAAN/dt in deg/day
    raan_k = np.zeros(L, dtype=np.float32) # Right ascension integrated
    argp_k = np.zeros(L, dtype=np.float32) # Argument of Perigee integrated
    a_k = np.zeros(L, dtype=np.float32)    # semi-major axis
    b_k = np.zeros(L, dtype=np.float32)    # semi-minor axis
    c_k = np.zeros(L, dtype=np.float32)    # center to focus
    ecc_k = np.zeros(L, dtype=np.float32)  # eccentricity
    inc_k = np.zeros(L, dtype=np.float32)  # inclination

    # Output Values from SGP4
    Ap_s = np.zeros(L, dtype=np.float32)   # Apogee values from SGP4
    Pp_s = np.zeros(L, dtype=np.float32)   # Perigee values from SGP4
    Au_s = np.zeros(L, dtype=np.float32)   # Energy at apogee
    Pu_s = np.zeros(L, dtype=np.float32)   # Energy at perigee
    dadt_s = np.zeros(L, dtype=np.float32) # dArgP/dt in deg/day
    drdt_s = np.zeros(L, dtype=np.float32) # dRAAN/dt in deg/day
    raan_s = np.zeros(L, dtype=np.float32) # Right ascension from 
    argp_s = np.zeros(L, dtype=np.float32) # Argument of Perigee from keplerian
    a_s = np.zeros(L, dtype=np.float32)    # semi-major axis
    b_s = np.zeros(L, dtype=np.float32)    # semi-minor axis
    c_s = np.zeros(L, dtype=np.float32)    # center to focus
    ecc_s = np.zeros(L, dtype=np.float32)  # eccentricity
    n_s = np.zeros(L, dtype=np.float32)    # mean motion
    inc_s = np.zeros(L, dtype=np.float32)  # inclination

    dadt_factor = 1.95
    drdt_factor = 2.05**-1

    def __dadt__(a, inc, ecc):
        # See Fundamentals of Astrodynamics (the Dover one) page 128
        c1 = 1.03237e14
        retval = c1*a**-3.5 * (4-5*np.sin(inc)**2)*(1-ecc**2)**-2

        # FIXME: For some reason, we seem to be off by a factor here...
        retval *= dadt_factor

        return retval

    def __drdt__(a, inc, ecc):
        # See Fundamentals of Astrodynamics (the Dover one) page 127.
        c1 = -2.06474e14
        c2 = -3.5
        retval = c1*a**c2 * np.cos(inc) * (1-ecc**2)**-2

        # FIXME: For some reason, we seem to be off by a factor here...
        retval *= drdt_factor

        # Note that it's the Nodal *Regression* rate defined on that
        # page, which means that for it to be the dr/dt, we need to
        # make it negative.
        retval *= -1

        return retval

    logging.info(f"  Computing Keplerian values for {L} observations")
    for i in range(L):

        # Apogee, Perigee, and basic elliptical values
        b_k[i] = 42241.122 * n_t[i]**(-2.0/3)
        Ap_k[i] = b_k[i]*(1+ecc_t[i])-Re_avg
        Pp_k[i] = b_k[i]*(1-ecc_t[i])-Re_avg
        a_k[i] = (Ap_k[i] + Pp_k[i] + 2*Re_avg)/2
        c_k[i] = (Ap_k[i] - Pp_k[i])/2
        ecc_k[i] = ecc_t[i] # Don't see a good way to compute this one
        inc_k[i] = inc_t[i] # Don't see a good way to compute this one
        assert(abs(ecc_t[i] - (c_k[i]/a_k[i])) < 1e-4)

        # Perturbative effects.
        drdt_k[i] = __drdt__(a_k[i], inc_t[i], ecc_t[i])
        dadt_k[i] = __dadt__(a_k[i], inc_t[i], ecc_t[i])

        # Specific mechanical energy relative to escape
        v_A = (mu * (2*(Ap_k[i]+Re_avg)**-1 - a_k[i]**-1))**.5
        v_P = (mu * (2*(Pp_k[i]+Re_avg)**-1 - a_k[i]**-1))**.5
        Au_k[i] = v_A**2/2 - mu/(Ap_k[i]+Re_avg)
        Pu_k[i] = v_P**2/2 - mu/(Pp_k[i]+Re_avg)


    logging.info(f"  Simulating one orbit for {L} observations")
    for i in range(L):

        # From where do you get t(he) NERV!? From SGP4, that's where!
        # Ba dum, CHING!!!
        # Thank you, thank you, I'll be pelted with tomatos now.
        t, N, e, r, v, = propagate_one_orbit(tles[i], E_t[i], dt_min)

        # These are the output values of interest from the
        # propagation.
        Rs = np.zeros(int(N))
        Vs = np.zeros(int(N))
        Us = np.zeros(int(N))

        # Magnitudes of the radius/velocity vectors (simple 2-norm)
        R = np.sum(r**2, axis=1)**.5
        Alt = R - Re_avg
        V = np.sum(v**2, axis=1)**.5

        # https://en.wikipedia.org/wiki/Geopotential_model eq 10(.5)
        z = np.array([r[j][-1] for j in range(N)])
        J2_term = J2e * R**-5 * 0.5 * (3*z**2 - R**2)
        J4_term = J4e
        U = (V**2/2        # Kinetic Energy
             + -1.0 * mu/R # Standard Gravitational Potential Energy
             + J2_term)    # Modification for J2

        # Index into the arrays of observations where we can find the
        # apogee and perigee.
        Aidx = np.argmax(R)
        Pidx = np.argmin(R)

        # Apogee/Perigee above mean Earth radius
        Ap_s[i] = Alt[Aidx]
        Pp_s[i] = Alt[Pidx]

        # Specific mechanical energy relative to escape
        Au_s[i] = U[Aidx]
        Pu_s[i] = U[Pidx]

        # Elliptical parameters
        a_s[i] = (Ap_s[i] + Pp_s[i] + 2*Re_avg)/2
        c_s[i] = (Ap_s[i] - Pp_s[i])/2
        b_s[i] = (a_s[i]**2 - c_s[i]**2)**.5
        ecc_s[i] = c_s[i]/a_s[i]
        inc_s[i] = inc_t[i] # Don't see a good way to compute this one

        # Perturbative effects.
        drdt_s[i] = __drdt__(a_s[i], inc_s[i], ecc_s[i])
        dadt_s[i] = __dadt__(a_s[i], inc_s[i], ecc_s[i])


    logging.info(f"  Numerically integrating ArgP & RAAN")
    for i in range(L):
        if 0 == i:
            raan_s[i] = raan_k[i] = raan_t[i]
            argp_s[i] = argp_k[i] = argp_t[i]
        else:
            dt = (E_t[i].timestamp()-E_t[i-1].timestamp())/one_day
            raan_k[i] = raan_k[i-1] + drdt_k[i] * dt
            argp_k[i] = argp_k[i-1] + dadt_k[i] * dt
            raan_s[i] = raan_s[i-1] + drdt_s[i] * dt
            argp_s[i] = argp_s[i-1] + dadt_s[i] * dt

    def __crossings__(arr, val):
        """Finds the locations where <arr> crosses <val>.

        It'll watch for a crossing one way, then do a 180-deg flip and
        run it again.  Because of the modulo nature of degrees, a true
        crossing will survive this flip, whereas a false crossing
        (from say going above 90-degrees at 359 to below 90-degrees at
        0) will not survive this flip.
        """
        arr = np.copy(arr)
        L = len(arr)

        x = np.where(arr >= val, 0, 1)
        y = np.diff(x)
        z = np.where(y == 0, 0, 1)

        # Real crossings survive a 180-degree flip
        survivors = z
        arr += 180
        arr %= 360
        val += 180
        val %= 360

        x = np.where(arr >= val, 0, 1)
        y = np.diff(x)
        z = np.where(y == 0, 0, 1)

        z *= survivors

        idx = np.linspace(1, L-1, L-1)
        retval = np.nonzero(z * idx)[0].astype(np.int)
        return retval

    def __interesting_crossings__(arr):
        """Finds the interesting crossings for the given array.

        These crossings are used to illustrate what is happening in
        various plots (energy/position vs what is happening in the
        argp).
        """
        vals = [90, 270]
        retval = np.sort(np.concatenate([__crossings__(arr, v) for v in vals]))
        return retval

    logging.info(f"  Done simulating, building plot")

    E_moon = moon_2008()

    # Done with the propagation, let's plot the results
    fig = plt.figure(figsize=(12, 8), dpi=600)

    # Legend labels
    lbls = []

    crossings = __interesting_crossings__(argp_t)

    # Position axis
    ax_p = fig.add_subplot(2, 1, 1)
    ax_p.set_xlabel("Observation Date")
    ax_p.set_ylabel("Orbital Altitude (km above mean)")
    CA = 'firebrick'
    CP = 'dodgerblue'
    lbls += ax_p.plot(E_t, Ap_s, '--', color=CA, label='Apogee (SGP4)')
    lbls += ax_p.plot(E_t, Ap_k, '-', color=CA, label='Apogee (Keplerian)')
    lbls += ax_p.plot(E_t, Pp_s, '--', color=CP, label='Perigee (SGP4)')
    lbls += ax_p.plot(E_t, Pp_k, '-', color=CP, label='Perigee (Keplerian)')
    lbls += ax_p.plot([E_t[i] for i in crossings],
                      [Ap_s[i] for i in crossings],
                      'x', color='black', label='Polar Perigee Crossing')
    ax_p.plot([E_t[i] for i in crossings],
              [Pp_s[i] for i in crossings],
              'x', color='black', label='Equatorial Perigee')

    # lbls += ax_p.plot(E_moon,
    #                   [Pp_s[np.searchsorted(E_t, e)] for e in E_moon],
    #                   '.', color='black', label='Full Moon')
    # ax_p.plot(E_moon,
    #           [Ap_s[np.searchsorted(E_t, e)] for e in E_moon],
    #           '.', color='black', label='Full Moon')

    ax_p.legend(lbls, [l.get_label() for l in lbls],
                bbox_to_anchor=(.75, .35, .2, 4),
                loc='lower left',
                ncol=1,
                mode="expand",
                borderaxespad=0)

    # Energy Axis
    ax_e = fig.add_subplot(2, 2, 3)
    ax_e.plot(E_t, Au_s, '--', color=CA, label='Energy at Apogee (SGP4)')
    ax_e.plot(E_t, Au_k, '-', color=CA, label='Energy at Apogee (TLE)')
    ax_e.plot(E_t, Pu_s, '--', color=CP, label='Energy at Perigee (SGP4)')
    ax_e.plot(E_t, Pu_k, '-', color=CP, label='Energy at Perigee (TLE)')
    ax_e.legend(bbox_to_anchor=(0, 1.05, 1, 4),
                loc='lower left',
                ncol=2,
                mode="expand",
                borderaxespad=0)
    ax_e.plot([E_t[i] for i in crossings],
              [Au_s[i] for i in crossings],
              'x', color='black', label='Crossing')
    # ax_e.plot(E_moon,
    #           [Au_s[np.searchsorted(E_t, e)] for e in E_moon],
    #           '.', color='black', label='Full Moon')
    ax_e.plot([E_t[i] for i in crossings],
              [Pu_s[i] for i in crossings],
              'x', color='black', label='Equatorial Perigee')
    # ax_e.plot(E_moon,
    #           [Pu_s[np.searchsorted(E_t, e)] for e in E_moon],
    #           '.', color='black', label='Full Moon')

    ax_e.set_xlabel("Observation Date")
    ax_e.set_ylabel("Specific Mechanical Energy (MJ/kg)")

    # Angular axis
    k = np.pi/180
    ax_a = fig.add_subplot(2, 2, 4, projection='polar')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
    ax_a.text(-.25, 1, "FIXME: Off by factor of ~2",
              transform=ax_a.transAxes, bbox=props)

    angular_t = ((np.array([t.timestamp() for t in E_t])
                  - E_t[0].timestamp())
                 / one_day)
    ro = ax_a.plot(k*raan_t, angular_t, '-', color=CA, label='RAAN (obs)')
    ao = ax_a.plot(k*argp_t, angular_t, '-', color=CP, label='ArgP (obs)')

    rcs = ax_a.plot(k*raan_s, angular_t, '--', color=CA, label='RAAN (int)')
    acs = ax_a.plot(k*argp_s, angular_t, '--', color=CP, label='ArgP (int)')

    ax_a.plot([k*argp_t[i] for i in crossings],
              [angular_t[i] for i in crossings],
              'x', color='black', label='Equatorial Perigee')


    lbls = [] + ro + rcs
    rlegend = ax_a.legend(handles=lbls,
                          bbox_to_anchor=(.75, -.1, .5, 4),
                          loc='lower left',
                          ncol=1,
                          mode="expand",
                          borderaxespad=0)

    ax_a.add_artist(rlegend)
    lbls = [] + ao + acs
    ax_a.legend(handles=lbls,
                bbox_to_anchor=(-.25, -.1, .5, 4),
                loc='lower left',
                ncol=1,
                mode="expand",
                borderaxespad=0)

    fig.suptitle(f"Keplerian Analytical vs SGP4 Propagation for {des} (inc=%1.1f$^\circ$)" % inc_t[0])

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(top=0.9)

    logging.info(f"  Saving figure to {output}")
    fig.savefig(output)
    logging.info(f"  Done...Please enjoy your data.")
    logging.info("")


def cmd_tle_var(des, tlefile=None, dt_min=1, output=None):
    """Examines a fragment's observations for covariance with argp.

    Since the SGP4 outputs are so tightly covariant with the argument
    of perigee, it is hypothesized that either

      a) Space-Track.org is not accounting for J2 when computing TLE values or
      b) They ARE accounting for J2 so as to make the naive approach meaningful

    This command will plot various figures of merit to look for that
    covariance.
    """

    if not output: output = f"tle_var-{des}.png"

    logging.info(f"=== Seek and Ye Shall Find Correlation ===")
    logging.info(f"  ...and Causation")
    logging.info(f"  ...that is absolutely how that works...")
    logging.info(f"  ...pretty sure anyway...")
    logging.info(f"  ")
    logging.info(f"  Designator: {des}")
    logging.info(f"  Source:     {tlefile}")
    logging.info(f"  Time step:  {dt_min}")
    logging.info(f"  Output:     {output}")
    logging.info(f"")

    (L,       # Number of TLEs parsed
     tles,    # TLEs: [[<line-1>,<line-2>], ...]
     E_t,     # Datetime of epoch from TLE        
     raan_t,  # Right ascension from TLE          
     argp_t,  # Argument of Perigee from TLE      
     raan_t,  # Right ascension from keplerian    
     argp_t,  # Argument of Perigee from keplerian
     ecc_t,   # eccentricity                      
     n_t,     # mean motion                       
     inc_t,   # inclination                       
     ) = snarf_tles(tlefile)

    ts = np.array([e.timestamp() for e in E_t])
    coef = np.polyfit(ts, 86400 / n_t, deg=1)
    ts -= ts[0]
    meaner = ts * coef[0] + coef[1]
    n = n_t - meaner

    dt = datetime.timedelta(days=5)
    E_moon = moon_2008()
    for i in range(len(E_moon)):
        E = E_moon[i]
        idx = np.searchsorted(E_t, E)
        raan = raan_t[idx]
        M = 5
        days = M * np.cos(raan_t[idx] * np.pi/180) + M
        days = 4
        days = M * raan/360.0
        dt = datetime.timedelta(days=days)
        E_moon[i] += dt

    fig = plt.figure(figsize=(12, 8), dpi=600)
    ax_argp = fig.add_subplot(1, 1, 1)
    ax_argp.plot(E_t, raan_t, label='ArgP')

    ax_ecc = ax_argp.twinx()
    ax_ecc.plot(E_t, n, label='Eccentricity')

    ax_ecc.plot(E_moon,
                [n[np.searchsorted(E_t, e)] for e in E_moon],
                '.', color='black', label='Full Moon')

    fig.savefig(output)


class ArgWrapper(object):
    """Wrapper class to make the argparser look like a dictionary.
    """

    def __init__(self, args):
        self.args = args

    def __getitem__(self, name):
        return getattr(self.args, name)


if __name__ == '__main__':
    desc = """Investigates the impacts of the oblate sphereoid on TLEs/Apogee/...

This is a self-contained test-script designed to illustrate specific
points.  As such, there are several specific tests with their own
options.  They're listed below.


=== Simulated APT values from Real TLEs (sim_apt) ===

Space-Track.org recommends finding the Apogee/Perigee of an orbit as follows:

    Re_max = 6378                  # Maximum radius of Earth
    n                              # orbits per day
    b                              # semi-minor axis

    b = 42241.122 * n**(-2.0/3)    # includes all the Earth constants
    apogee = b*(1+ecc)-Re_max      # keplerian
    perigee = b*(1-ecc)-Re_max     # keplerian
    period = (MINUTES_PER_DAY / n) # keplerian

I make one modification which is to use Re_avg instead.

Using the observations of a single spacecraft, we compute the APT
values using keplerian methods, and we also compute them using SGP4
propagation and observe the difference.



=== TLE Variance (tle_var) ===

With the results of <sim_apt> indicating that some values in the TLEs
are in a PLL with the Argument of Perigee, this command examines the
values in the TLEs themselves looking for covariance.

"""
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=formatter)

    parser.add_argument('cmd',
                        metavar='CMD',
                        action='store',
                        type=str,
                        choices=['sim_apt', 'tle_var'],
                        help='Compare simulations of APT to computations')

    parser.add_argument('--tle-file', '-t',
                        metavar='TLE_FILE',
                        action='store',
                        dest='tle_file',
                        required=False,
                        type=str,
                        default='%(des)s.txt',
                        help='Input file with TLEs (%(default)s)')

    parser.add_argument('--timestep-minutes',
                        metavar='TIMESTEP',
                        action='store',
                        dest='dt_min',
                        required=False,
                        type=float,
                        default=1.0,
                        help='Time-step when running SGP4 (%(default)s)')

    parser.add_argument('--designator', '-d',
                        metavar='INTL_DES',
                        action='store',
                        dest='des',
                        required=False,
                        type=str,
                        default='99025ABC',
                        help='International designator of fragment (%(default)s)')

    args = parser.parse_args()
    awrap = ArgWrapper(args)

    logging.root.level = logging.INFO
    np.set_printoptions(precision=2)

    if args.cmd == 'sim_apt':
        cmd_sim_apt(args.des, tlefile=args.tle_file%awrap, dt_min=args.dt_min)
    elif args.cmd == 'tle_var':
        cmd_tle_var(args.des, tlefile=args.tle_file%awrap, dt_min=args.dt_min)
    else:
        logging.critical(f"What did you do?!?!eleven??1?! {args.cmd}")

