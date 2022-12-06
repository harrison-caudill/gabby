#!/usr/bin/env python

import datetime
import functools
import math
import matplotlib.pyplot as plt
import numpy as np
import pprint
from sgp4.api import Satrec
from sgp4.api import jday
import sys

import scipy.signal

import gabby

"""Illustrates the relationship between decay rates and Bstar values

"""


def julian_dates(start, end, dt):
    """Generates the Julian days and Fractions for SGP4.

    Anytime you deal with anything at all astronomical in nature, you
    seem to have to start by questioning your concepts of up and down
    (i.e. your coordinate system) and of time.  SGP4 uses integer days
    somehow related to the Julian calendar and some epoch whose
    definition I don't really understand and also fractions within
    those days.

    This little helper routine will generate and return the two time
    arrays necessary for performing SGP4 propagations.

    start: datetime.datetime
    end: datetime.datetime
    dt: datetime.timedelta
    """

    ts = np.arange(start, end, dt)

    N = len(ts)

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

    return jds, frs


def run_sim(tle, jd, fr, dt_min=0.1, dt_days=1):
    """Runs a single round of simulation and returns the decay amount
    """

    # python-sgp4 representation of a satellite
    sat = Satrec.twoline2rv(*str(tle).split('\n')[1:])

    # These are the output values of interest from the
    # propagation.
    N = len(jd)
    Rs = np.zeros(int(N))

    # r is an array of N xyz coordinates
    e, r, v = sat.sgp4_array(jd, fr)

    # Magnitudes of the radius/velocity vectors (simple 2-norm)
    Re = 6371
    R = np.sum(r**2, axis=1)**.5 - Re

    # Find the first and last apogee/perigee
    N = len(R)
    T = tle.T
    dt = datetime.timedelta(minutes=dt_min)
    nT = math.ceil(1.1*T / dt)
    first = R[:nT]
    last = R[:nT]
    A0i = np.argmax(first)
    A1i = np.argmax(last) + N - nT
    P0i = np.argmin(first)
    P1i = np.argmin(last) + N - nT

    A0 = R[A0i]
    A1 = R[A1i]
    P0 = R[P0i]
    P1 = R[P1i]

    # Even a low earth orbit satellite has a 90 minute orbit.  If we
    # have to use up to 90 minutes for each observation, we'll want to
    # make sure we start at an apogee.  Even then, however, we still
    # lose up to 110 minutes out of 1440...not good enough...  We're
    # going to scale the result and assume local linearity over the
    # course of whatever time-step was specified.

    frac_dt = (A1i-A0i) / (datetime.timedelta(days=dt_days) / dt)
    dA = (A1-A0) / frac_dt
    frac_dt = (P1i-P0i) / (datetime.timedelta(days=dt_days) / dt)
    dP = (P1-P0) / frac_dt

    return dA, dP


def vary_argp(A, P, B, inc, epoch=None, dt_min=0.1, dt_days=1, N=360):

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

    # Find the date-ranges of the simulation.  We add one full orbital
    # period to bring our average peak->peak or trough->trough times
    # closer to 1 day than to 1 day - T
    if not epoch: epoch = datetime.datetime(1955, 11, 5, 0, 0, 0)
    T = datetime.timedelta(days=1) / n
    start = epoch
    end = start + datetime.timedelta(days=dt_days) + T
    dt = datetime.timedelta(minutes=dt_min)

    jd, fr = julian_dates(start, end, dt)

    # Output Values from SGP4
    dA = np.zeros(N, dtype=np.float32) # Apogee deltas
    dP = np.zeros(N, dtype=np.float32) # Perigee deltas

    # argp's to loop through
    dargp = 360.0 / N
    argp = np.linspace(0, 360-dargp, N)

    for i in range(N):
        raan = 0
        tle = gabby.TLE(n=n,
                        cat_number=i+1,
                        inc=inc,
                        raan=raan,
                        bstar=B,
                        ecc=ecc,
                        argp=argp[i],
                        epoch=epoch)
        dA[i], dP[i] = run_sim(tle, jd, fr, dt_min=dt_min, dt_days=dt_days)

    return np.mean(dA), np.mean(dP)


def vary_bstar(A, P, Bs, inc, dt_days=1, dt_min=1, ax=None):
    """Graphs the relationship between the bstar value and the decay rate.

    A: <float> nominal apogee
    P: <float> nominal perigee
    Bs: np[<float>] bstar values
    inc: <float> inclination
    """

    # Number of trials to run
    N = len(Bs)

    # Output Values from SGP4
    dA = np.zeros(N, dtype=np.float32) # Apogee deltas
    dP = np.zeros(N, dtype=np.float32) # Perigee deltas

    for i in range(N):
        print(f"BStar[%-3d]: {Bs[i]}" % i)
        dA[i], dP[i] = vary_argp(A, P, Bs[i], inc,
                                 dt_days=dt_days,
                                 dt_min=dt_min)

    if ax:
        CA = 'firebrick'
        CP = 'dodgerblue'
        ax.set_title(f"A/P Decay Rates (A={A} P={P} inc={inc})")
        ax.plot(Bs, dA, color=CA, label='Apogee')
        ax.plot(Bs, dP, color=CP, label='Perigee')
        ax.set_xlabel("B*")
        ax.set_ylabel("Daily Change in Orbital Altitude (km)")

    return dA, dP


def plot_bstars():

    fig = plt.figure(figsize=(12, 8), dpi=600)

    fig.suptitle(f"Impact of B* Values on Daily Decay Rates")

    B = np.linspace(-1e-6, -1e-4, 200)

    vary_bstar(700, 700, B, 90, ax=fig.add_subplot(2, 2, 1))

    vary_bstar(700, 700, B, 0, ax=fig.add_subplot(2, 2, 2))

    vary_bstar(800, 400, B, 90, ax=fig.add_subplot(2, 2, 3))

    vary_bstar(600, 400, B, 90, ax=fig.add_subplot(2, 2, 4))

    fig.tight_layout(h_pad=2)

    fig.savefig(f"bstar.png")

if __name__ == '__main__':
    plot_bstars()
