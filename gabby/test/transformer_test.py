#!/usr/bin/env python

import numpy as np
import pytest

from .fixtures import *


class TestTransformer(object):
    pass

    # FIXME: Re-enable after refactor
    # def test_concatenate(self, jazzercise):

    #     N = [3, 1, 2]
    #     correct = np.array([1, 2, 3, 4, 7, 8], dtype=np.float32)

    #     tmp = np.array([[[1, 2, 3], [0, 0, 0]],
    #                     [[4, 0, 0], [0, 0, 0]],
    #                     [[7, 8, 0], [0, 0, 0]]],
    #                    dtype=np.float32)
    #     res = jazzercise._Jazz__concatenate(tmp, N, subkeys=[0])
    #     assert(0 == len(np.trim_zeros(res - correct)))

    #     tmp = np.array([[[0, 0, 0], [1, 2, 3], ],
    #                     [[0, 0, 0], [4, 0, 0], ],
    #                     [[0, 0, 0], [7, 8, 0], ]],
    #                    dtype=np.float32)
    #     res = jazzercise._Jazz__concatenate(tmp, N, subkeys=[1])
    #     assert(0 == len(np.trim_zeros(res - correct)))

    # def test_percentile(self, jazzercise):

    #     mu = .1
    #     sigma = .01
    #     N = 1024
    #     dist = np.random.normal(mu, sigma, N)

    #     frac = .1
    #     off = int(frac*N)

    #     a, b = jazzercise._percentile_values(dist, frac, frac)

    #     dist_sorted = np.sort(dist)
    #     A = dist_sorted[off]
    #     B = dist_sorted[N-off-1]

    #     assert(a == A and b == B)

    # def test_clip_to_flanks(self, jazzercise):

    #     arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    #     c_min, c_max, step, res = jazzercise.clip_to_flanks(arr, 6,
    #                                                         low_clip=.2,
    #                                                         high_clip=.2)
    #     correct = np.array([2.1, 2.1, 3, 4, 5, 6, 7, 8, 8.9, 8.9],
    #                        dtype=np.float32)
    #     assert(0 == len(np.trim_zeros(res - correct)))

    #     arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    #     c_min, c_max, step, res = jazzercise.clip_to_flanks(arr, 5,
    #                                                         low_clip=.2,
    #                                                         high_clip=.2,
    #                                                         max_val=7)
    #     correct = np.array([2.1, 2.1, 3, 4, 5, 6, 7, 7.9, 7.9, 7.9],
    #                        dtype=np.float32)
    #     assert(0 == len(np.trim_zeros(res - correct)))

    #     arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    #     c_min, c_max, step, res = jazzercise.clip_to_flanks(arr, 4,
    #                                                         max_val=7,
    #                                                         min_val=4)
    #     correct = np.array([3.1, 3.1, 3.1, 4, 5, 6, 7, 7.9, 7.9, 7.9],
    #                        dtype=np.float32)
    #     assert(0 == len(np.trim_zeros(res - correct)))

    # def test_flanking_digitize(self, jazzercise):
    #     arr = np.array([2.1, 2.1, 3, 4, 5, 6, 7, 8, 8.9, 8.9],
    #                    dtype=np.float32)

    #     res = jazzercise._flanking_digitize(arr, 3, 1)
    #     correct = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 7], dtype=np.int)
    #     assert(0 == len(np.trim_zeros(res - correct)))

    # def test_derivative(self, jazzercise):
    #     # FIXME
    #     pass

    # def test_moral_decay(self, jazzercise):

    #     # Test the very basics
    #     # This should result in a single filled bin for the derivative
    #     # and 2 bins for A/P
    #     fragments = ['99025A', '99025B']
    #     N = 10
    #     A0 = P0 = 500

    #     # From the docstring:
    #     # positions: [frag-number][off][0] = time
    #     #            [frag-number][off][1] = apogee value
    #     #            [frag-number][off][2] = perigee value
    #     pos = np.zeros((2, N, 3), dtype=np.float32)

    #     times = np.linspace(0, N-1, N, dtype=np.int)*gabby.SECONDS_IN_DAY
    #     As = np.linspace(0, 1-N, N, dtype=np.float32) + A0
    #     Ps = np.linspace(0, 1-N, N, dtype=np.float32) + P0
    #     pos[0,:,0] = times
    #     pos[0,:,1] = As
    #     pos[0,:,2] = Ps

    #     N = 5
    #     times = np.linspace(0, N-1, N, dtype=np.int)*gabby.SECONDS_IN_DAY
    #     As = np.linspace(0, 1-N, N, dtype=np.float32) + A0
    #     Ps = np.linspace(0, 1-N, N, dtype=np.float32) + P0

    #     Ns = [10, 5,]

    #     pos[1,:,0] = times
    #     pos[1,:,1] = As
    #     pos[1,:,2] = Ps

    #     deriv = np.zeros((2, N-1, 3), dtype=np.float32)
    #     deriv[0] = pos[0,1:,:]
    #     deriv[0,:,0] = np.diff(pos[0,:,0])
    #     deriv[0,:,1] = np.diff(pos[0,:,1])
    #     deriv[0,:,2] = np.diff(pos[0,:,2])
    #     deriv[1,:,0] = np.diff(pos[1,:,0])
    #     deriv[1,:,1] = np.diff(pos[1,:,1])
    #     deriv[1,:,2] = np.diff(pos[1,:,2])

    #     Ns = np.zeros(2, dtype=np.int) + N
    #     import pprint
    #     pprint.pprint(Ns)

    #     moral_decay = jazzercise.decay_rates(pos, deriv, Ns)

    #     print(moral_decay)

    #     # FIXME: Make sure negative values work properly
