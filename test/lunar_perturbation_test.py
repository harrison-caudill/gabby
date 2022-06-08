#!/usr/bin/env python

import datetime
import functools
import matplotlib.pyplot as plt
import numpy as np
import pprint
from sgp4.api import Satrec
from sgp4.api import jday
import sys

import gabby

"""Development file to find the right way to build orbits for MoralDecay.


The decay rate of a fragment is really affected by three numbers:

 * Apogee
 * Perigee
 * not-quite-B* Drag coefficient (the one that includes solar activity)

In a simplistic world, Apogee[1] = Apogee[0] + dA/dt * dt.  However,
there is one other major factor which impacts the actual Apogee (and
Perigee): Recession of the argument of perigee over time.  We assume
that these effects are small.


 * Figure out how to reliably generate an orbit which can be employed
   by SGP4 that, after a complete propagation, has the desired real
   apogee/perigee.

 * Make sure the A' and P' of a satellite is somewhat sane when using
   SGP4 as the investigative tool.


 * Should also consider the inclination...


 * Ideally, we have a function which is something like:
   f(A, P) -> TLE

"""


def vary_raan(A, P, inc, dt_min=1, N=360, ax=None):
    """Check the impact of RAAN on orbit A/P to watch for lunar impacts

    A: <float> nominal apogee
    P: <float> nominal perigee
    inc: <float> inclination
    N: <int> number of RAAN values to try
    """

    # The RAAN values to loop through
    draan = 360.0 / N
    raans = np.linspace(0, 360-draan, N)

    # Output Values from SGP4
    Ap_s = np.zeros(N, dtype=np.float32)   # Apogee values from SGP4
    Pp_s = np.zeros(N, dtype=np.float32)   # Perigee values from SGP4

    # Earth constants
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

    print(f"=== Examining Effects of RAAN/Inc on Orbit ===")
    print(f"  a: {a} km")
    print(f"  b: {b} km")
    print(f"  c: {c} km")
    print(f"  A: {A} km")
    print(f"  P: {P} km")
    print(f"  R: {Re} km")
    print(f"  e: {ecc}")
    print(f"  T: {Ts//60.0} minutes")
    print(f"  n: {n} orbits/day")
    print(f"  inc: {inc} deg")
    print("")

    # When else would we choose?
    epoch = datetime.datetime(1955, 11, 5, 0, 0, 0)

    # Loop through the raans and run a single sim
    for i in range(N):
        raan = raans[i]

        # for now, just use 0 as argument of perigee
        argp = 0

        tle = gabby.TLE(n=n,
                        cat_number=i+1,
                        inc=inc,
                        raan=raan,
                        ecc=ecc,
                        argp=argp,
                        epoch=epoch)

        # python-sgp4 representation of a satellite
        sat = Satrec.twoline2rv(*str(tle).split('\n')[1:])

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

        # r is an array of N xyz coordinates
        e, r, v = sat.sgp4_array(jds, frs)

        # Magnitudes of the radius/velocity vectors (simple 2-norm)
        R = np.sum(r**2, axis=1)**.5
        Alt = R - Re
        V = np.sum(v**2, axis=1)**.5
        U = V**2/2 - mu/R

        Aidx = np.argmax(R)
        Pidx = np.argmin(R)

        Ap_s[i] = Alt[Aidx]
        Pp_s[i] = Alt[Pidx]

    if ax:
        ax.set_xlabel("Right Ascension (deg)")
        ax.set_ylabel("Orbital Altitude (km above MSL)")
        CA = 'firebrick'
        CP = 'dodgerblue'
        ax.plot(raans, Ap_s, color=CA, label='Apogee')
        ax.plot(raans, Pp_s, color=CP, label='Perigee')
        ax.set_title(f"A={int(A)} P={int(P)} inc={int(inc)}")

    return raans, Ap_s, Pp_s


def plot_raan_inc(A=600, P=500):
    # Done with the propagation, let's plot the results
    fig = plt.figure(figsize=(12, 8), dpi=600)

    
    fig.suptitle(f"Impact of RAAN/inc on effective vs nominal Apogee/Perigee")

    vary_raan(A, P, 0, ax=fig.add_subplot(2, 2, 1))
    vary_raan(A, P, 30, ax=fig.add_subplot(2, 2, 2))
    vary_raan(A, P, 60, ax=fig.add_subplot(2, 2, 3))
    vary_raan(A, P, 90, ax=fig.add_subplot(2, 2, 4))

    fig.tight_layout(h_pad=2)

    fig.savefig(f"raan_inc.png")


if __name__ == '__main__':
    plot_raan_inc(A=700, P=700)
