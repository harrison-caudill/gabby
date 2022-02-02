#!/usr/bin/env python3

import datetime
import functools
import matplotlib.pyplot as plt
import numpy as np
import pprint
from sgp4.api import Satrec
from sgp4.api import jday
import sys

"""Investigates the impacts of the oblate sphereoid on TLEs/Apogee/...

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

Using the observations of a single spacecraft, we compute the APT
values using keplerian methods, and we also compute them using SGP4
propagation and observe the difference.

"""


def sim_apt(des, tlefile=None, dt_min=1):
    """Runs the test for an individual TLE.

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
    raan_t = np.zeros(L, dtype=np.float32) # Right ascension from TLE
    argp_t = np.zeros(L, dtype=np.float32) # Argument of Perigee from TLE

    # Output Values from SGP4
    Ap_s = np.zeros(L, dtype=np.float32)   # Apogee values from SGP4
    Pp_s = np.zeros(L, dtype=np.float32)   # Perigee values from SGP4
    Au_s = np.zeros(L, dtype=np.float32)   # Energy at apogee
    Pu_s = np.zeros(L, dtype=np.float32)   # Energy at perigee
    dadt_s = np.zeros(L, dtype=np.float32) # dArgP/dt in deg/day
    drdt_s = np.zeros(L, dtype=np.float32) # dRAAN/dt in deg/day
    raan_s = np.zeros(L, dtype=np.float32) # Right ascension from keplerian
    argp_s = np.zeros(L, dtype=np.float32) # Argument of Perigee from keplerian
    a_s = np.zeros(L, dtype=np.float32)    # semi-major axis
    b_s = np.zeros(L, dtype=np.float32)    # semi-minor axis
    c_s = np.zeros(L, dtype=np.float32)    # center to focus
    ecc_s = np.zeros(L, dtype=np.float32)  # eccentricity
    n_s = np.zeros(L, dtype=np.float32)    # mean motion
    inc_s = np.zeros(L, dtype=np.float32)  # inclination

    # Output Values from the TLE/Keplerian
    Ap_k = np.zeros(L, dtype=np.float32)   # Apogee values from keplerian
    Pp_k = np.zeros(L, dtype=np.float32)   # Perigee values from keplerian
    Au_k = np.zeros(L, dtype=np.float32)   # Energy at apogee
    Pu_k = np.zeros(L, dtype=np.float32)   # Energy at perigee
    raan_k = np.zeros(L, dtype=np.float32) # Right ascension from keplerian
    argp_k = np.zeros(L, dtype=np.float32) # Argument of Perigee from keplerian
    E_k = []                               # Datetime of epoch from TLE
    dadt_k = np.zeros(L, dtype=np.float32) # dArgP/dt in deg/day
    drdt_k = np.zeros(L, dtype=np.float32) # dRAAN/dt in deg/day
    a_k = np.zeros(L, dtype=np.float32)    # semi-major axis
    b_k = np.zeros(L, dtype=np.float32)    # semi-minor axis
    c_k = np.zeros(L, dtype=np.float32)    # center to focus
    ecc_k = np.zeros(L, dtype=np.float32)  # eccentricity
    n_k = np.zeros(L, dtype=np.float32)    # mean motion
    inc_k = np.zeros(L, dtype=np.float32)  # inclination

    # Earth constants
    Re_max = 6378
    Re_avg = 6371
    mu = 3.986004418e5 # km^3 / s^2
    one_day = datetime.timedelta(days=1).total_seconds()

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

    for i in range(L):
        tle = tles[i]

        # === Snarf/Compute the Keplerian version ===

        # Find the basic orbital paramters in the TLE
        ecc_k[i] = float('.'+tle[1][26:33])
        n_k[i] = float(tle[1][52:63])
        raan_t[i] = float(tle[1][17:25])
        argp_t[i] = float(tle[1][34:42])
        inc_k[i] = float(tle[1][8:16])
        year = int(tle[0][18:20])
        year = year + 1900 if year > 57 else year + 2000
        days = float(tle[0][20:32])
        epoch = (datetime.datetime(year, 1, 1)+datetime.timedelta(days=days))
        E_k.append(epoch)

        # Find the keplerian version using the space-track.org
        # recommended math.
        b_k[i] = 42241.122 * n_k[i]**(-2.0/3)
        Ap_k[i] = b_k[i]*(1+ecc_k[i])-Re_max
        Pp_k[i] = b_k[i]*(1-ecc_k[i])-Re_max
        a_k[i] = (Ap_k[i] + Pp_k[i] + 2*Re_max)/2
        c_k[i] = (Ap_k[i] - Pp_k[i])/2
        assert(abs(ecc_k[i] - (c_k[i]/a_k[i])) < 1e-4)
        drdt_k[i] = __drdt__(a_k[i], inc_k[i], ecc_k[i])
        dadt_k[i] = __dadt__(a_k[i], inc_k[i], ecc_k[i])
        v_A = (mu * (2*(Ap_k[i]+Re_max)**-1 - a_k[i]**-1))**.5
        v_P = (mu * (2*(Pp_k[i]+Re_max)**-1 - a_k[i]**-1))**.5
        Au_k[i] = v_A**2/2 - mu/(Ap_k[i]+Re_max)
        Pu_k[i] = v_P**2/2 - mu/(Pp_k[i]+Re_max)

        # === Simulate the SGP4 version ===

        # python-sgp4 representation of a satellite
        sat = Satrec.twoline2rv(*tle)

        # The mean motion in the sgp4 package appears to be a mean of
        # mean-motion as days/orbit rather than orbits/day (so inverse
        # of some averaged value from propagation of the TLE I
        # guess???)
        T = one_day*sat.nm

        # Find the date-ranges of the simulation
        start = epoch
        end = start + datetime.timedelta(seconds=1.1*T)
        dt = datetime.timedelta(minutes=dt_min)
        ts = np.arange(start, end, dt)

        # These are the output values of interest from the
        # propagation.
        N = len(ts)
        Rs = np.zeros(int(N))
        Vs = np.zeros(int(N))
        Us = np.zeros(int(N))

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

        # Magnitudes of the radius/velocity vectors (simple 2-norm)
        R = np.sum(r**2, axis=1)**.5
        Alt = R - Re_avg
        V = np.sum(v**2, axis=1)**.5
        U = V**2/2 - mu/R

        Aidx = np.argmax(R)
        Pidx = np.argmin(R)

        Ap_s[i] = Alt[Aidx]
        Pp_s[i] = Alt[Pidx]
        Au_s[i] = U[Aidx]
        Pu_s[i] = U[Pidx]

        a_s[i] = (Ap_s[i] + Pp_s[i] + 2*Re_avg)/2
        c_s[i] = (Ap_s[i] - Pp_s[i])/2
        b_s[i] = (a_s[i]**2 - c_s[i]**2)**.5
        ecc_s[i] = c_s[i]/a_s[i]
        inc_s[i] = inc_k[i] # Don't see a good way to compute this one
        drdt_s[i] = __drdt__(a_s[i], inc_s[i], ecc_s[i])
        dadt_s[i] = __dadt__(a_s[i], inc_s[i], ecc_s[i])

        if 0 == i:
            raan_s[i] = raan_k[i] = raan_t[i]
            argp_s[i] = argp_k[i] = argp_t[i]
        else:
            dt = (E_k[i].timestamp()-E_k[i-1].timestamp())/one_day
            raan_k[i] = raan_k[i-1] + drdt_k[i] * dt
            argp_k[i] = argp_k[i-1] + dadt_k[i] * dt
            raan_s[i] = raan_s[i-1] + drdt_s[i] * dt
            argp_s[i] = argp_s[i-1] + dadt_s[i] * dt

    # Done with the propagation, let's plot the results
    fig = plt.figure(figsize=(12, 8), dpi=600)

    # Legend labels
    lbls = []

    # Position axis
    ax_p = fig.add_subplot(2, 1, 1)
    ax_p.set_xlabel("Observation Date")
    ax_p.set_ylabel("Orbital Altitude (km above mean)")
    CA = 'firebrick'
    CP = 'dodgerblue'
    lbls += ax_p.plot(E_k, Ap_s, '--', color=CA, label='Apogee (SGP4)')
    lbls += ax_p.plot(E_k, Ap_k, '-', color=CA, label='Apogee (Keplerian)')
    lbls += ax_p.plot(E_k, Pp_s, '--', color=CP, label='Perigee (SGP4)')
    lbls += ax_p.plot(E_k, Pp_k, '-', color=CP, label='Perigee (Keplerian)')
    ax_p.legend(lbls, [l.get_label() for l in lbls],
                bbox_to_anchor=(.75, .35, .2, 4),
                loc='lower left',
                ncol=1,
                mode="expand",
                borderaxespad=0)

    # Energy Axis
    ax_e = fig.add_subplot(2, 2, 3)
    ax_e.plot(E_k, Au_s, '--', color=CA, label='Energy at Apogee (SGP4)')
    ax_e.plot(E_k, Au_k, '-', color=CA, label='Energy at Apogee (TLE)')
    ax_e.plot(E_k, Pu_s, '--', color=CP, label='Energy at Perigee (SGP4)')
    ax_e.plot(E_k, Pu_k, '-', color=CP, label='Energy at Perigee (TLE)')
    ax_e.legend(bbox_to_anchor=(0, 1, 1, 4),
                loc='lower left',
                ncol=2,
                mode="expand",
                borderaxespad=0)
    ax_e.set_xlabel("Observation Date")
    ax_e.set_ylabel("Specific Mechanical Energy (MJ/kg)")

    # Angular axis
    k = np.pi/180
    ax_a = fig.add_subplot(2, 2, 4, projection='polar')
    angular_t = ((np.array([t.timestamp() for t in E_k])
                  - E_k[0].timestamp())
                 / one_day)
    ro = ax_a.plot(k*raan_t, angular_t, '-', color=CA, label='RAAN (obs)')
    ao = ax_a.plot(k*argp_t, angular_t, '-', color=CP, label='ArgP (obs)')

    rcs = ax_a.plot(k*raan_s, angular_t, '--', color=CA, label='RAAN (SGP4)')
    acs = ax_a.plot(k*argp_s, angular_t, '--', color=CP, label='ArgP (SGP4)')

    # rck = ax_a.plot(k*raan_k, angular_t, linestyle=(0, (5, 1)), color=CA, label='RAAN (Keplerian)')
    # rck = ax_a.plot(k*argp_k, angular_t, linestyle=(0, (5, 1)), color=CP, label='ArgP (Keplerian)')

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

    fig.suptitle(f"{des} (inc=%1.1f$^\circ$)" % inc_k[0])

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(top=0.9)

    fig.savefig(f"{des}.png")

if __name__ == '__main__':
    np.set_printoptions(precision=2)
    sim_apt('99025ABC')
