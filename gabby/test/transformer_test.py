#!/usr/bin/env python

import numpy as np
import pprint
import pytest

from .fixtures import *

from ..defs import *


class TestTransformer(object):

    def test_concatenate(self, jazzercise):

        N = [3, 1, 2]
        correct = np.array([1, 2, 3, 4, 7, 8], dtype=np.float32)

        tmp = np.array([[1, 2, 3],
                        [4, 0, 0],
                        [7, 8, 0],],
                       dtype=np.float32)
        res = jazzercise._concatenate(tmp, N)
        assert(0 == len(np.trim_zeros(res - correct)))

        tmp = np.array([[1, 2, 3],
                        [4, 0, 0],
                        [7, 8, 0]],
                       dtype=np.float32)
        res = jazzercise._concatenate(tmp, N)
        assert(0 == len(np.trim_zeros(res - correct)))

    def test_percentile(self, jazzercise):

        mu = .1
        sigma = .01
        N = 1024
        dist = np.random.normal(mu, sigma, N)

        frac = .1
        off = int(frac*N)

        a, b = jazzercise._percentile_values(dist, frac, frac)

        dist_sorted = np.sort(dist)
        A = dist_sorted[off]
        B = dist_sorted[N-off-1]

        assert(a == A and b == B)

    def test_clip_to_flanks(self, jazzercise):

        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
        c_min, c_max, step, res = jazzercise.clip_to_flanks(arr, 6,
                                                            low_clip=.2,
                                                            high_clip=.2)
        correct = np.array([2.1, 2.1, 3, 4, 5, 6, 7, 8, 8.9, 8.9],
                           dtype=np.float32)
        assert(0 == len(np.trim_zeros(res - correct)))

        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
        c_min, c_max, step, res = jazzercise.clip_to_flanks(arr, 5,
                                                            low_clip=.2,
                                                            high_clip=.2,
                                                            max_val=7)
        correct = np.array([2.1, 2.1, 3, 4, 5, 6, 7, 7.9, 7.9, 7.9],
                           dtype=np.float32)
        assert(0 == len(np.trim_zeros(res - correct)))

        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
        c_min, c_max, step, res = jazzercise.clip_to_flanks(arr, 4,
                                                            max_val=7,
                                                            min_val=4)
        correct = np.array([3.1, 3.1, 3.1, 4, 5, 6, 7, 7.9, 7.9, 7.9],
                           dtype=np.float32)
        assert(0 == len(np.trim_zeros(res - correct)))

    def test_flanking_digitize(self, jazzercise):
        arr = np.array([2.1, 2.1, 3, 4, 5, 6, 7, 8, 8.9, 8.9],
                       dtype=np.float32)

        res = jazzercise._flanking_digitize(arr, 3, 1)
        correct = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 7], dtype=np.int)
        assert(0 == len(np.trim_zeros(res - correct)))

    def test_derivative(self, cfg, linear_faker):

        # fewer keystrokes
        faker = linear_faker

        # Make sure we're loading the right data
        tranny = Jazz(cfg)
        fragments = ['99025']
        apt = faker.db.load_apt(fragments)
        apt_A = [[600, 475, 350, 225, 100]]
        apt_P = [[400, 325, 250, 175, 100]]
        assert(np.all(apt.A[0,:5] == apt_A))
        assert(np.all(apt.P[0,:5] == apt_P))

        # Make sure the derivatives work
        filtered, deriv = tranny.filtered_derivatives(apt,
                                                      min_life=1.0,
                                                      dt=SECONDS_IN_DAY)

        assert(np.all(np.diff(apt_A[0]) == deriv.A[0]))
        assert(np.all(np.diff(apt_P[0]) == deriv.P[0]))
        assert((1, 4) == deriv.A.shape == deriv.P.shape)
        assert((1, 4) == filtered.A.shape == filtered.P.shape)
        assert(np.all(filtered.t == deriv.t))

    def test_moral_decay(self, cfg, linear_faker):

        # fewer keystrokes
        faker = linear_faker

        # Compute the inputs
        apt = faker.db.load_apt(['99025'])
        tranny = Jazz(cfg)
        filtered, deriv = tranny.filtered_derivatives(apt,
                                                      min_life=1.0,
                                                      dt=SECONDS_IN_DAY)

        # Make the derivatives non-uniform so that we actually have
        # different values to use.
        deriv.A[0] += np.linspace(0, deriv.N[0]-1, deriv.N[0])
        deriv.P[0] += np.linspace(0, deriv.N[0]-1, deriv.N[0])

        ret = tranny.decay_rates(apt, filtered, deriv)
        expected = [[[[0, 1],
                      [0, 0]],

                     [[0, 0],
                      [1, 0]]],


                    [[[0, 1],
                      [0, 0]],

                     [[0, 0],
                      [1, 0]]]]
        assert(np.all(expected == ret.decay_hist))
