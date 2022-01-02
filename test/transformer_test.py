#!/usr/bin/env python

import numpy as np
import pytest

import gabby


@pytest.fixture
def jazzercise():
    jazz = gabby.Jazz(None, None, None, None, None)


class TestTransformer(object):

    def test_concatenate(self, jazzercise):

        N = [3, 1, 2]
        correct = np.array([1, 2, 3, 4, 7, 8], dtype=np.float32)

        tmp = np.array([[[1, 2, 3], [0, 0, 0]],
                        [[4, 0, 0], [0, 0, 0]],
                        [[7, 8, 0], [0, 0, 0]]],
                       dtype=np.float32)
        res = jazzercise._Jazz__concatenate(tmp, N, subkeys=[0])
        assert(0 == len(np.trim_zeros(res - correct)))

        tmp = np.array([[[0, 0, 0], [1, 2, 3], ],
                        [[0, 0, 0], [4, 0, 0], ],
                        [[0, 0, 0], [7, 8, 0], ]],
                       dtype=np.float32)
        res = jazzercise._Jazz__concatenate(tmp, N, subkeys=[1])
        assert(0 == len(np.trim_zeros(res - correct)))

    def test_percentile(self, jazzercise):

        mu = .1
        sigma = .01
        N = 1024
        dist = np.random.normal(mu, sigma, N)

        frac = .1
        off = int(frac*N)

        a, b = jazzercise._percentile_values(dist, frac)

        dist_sorted = np.sort(dist)
        A = dist_sorted[off]
        B = dist_sorted[N-off-1]

        assert(a == A and b == B)

    def test_clip_to_flanks(self, jazzercise):

        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
        c_min, c_max, step, res = jazzercise.clip_to_flanks(arr, 6)
        correct = np.array([2.1, 2.1, 3, 4, 5, 6, 7, 8, 8.9, 8.9],
                           dtype=np.float32)
        print(res)
        assert(0 == len(np.trim_zeros(res - correct)))

        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
        c_min, c_max, step, res = jazzercise.clip_to_flanks(arr, 5,
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

    def test_moral_decay(self, jazzercise):
        # FIXME: Make sure negative values work properly
        pass
