#!/usr/bin/env python

import datetime
import lmdb
import numpy as np
import pprint
import pytest
import sys

import gabby
from fixtures import *

class TestFaker(object):

    def test_fill_apt(self, single_faker):

        faker = single_faker

        L = 2
        frag = '99025'
        utc = datetime.timezone.utc
        t = [gabby.dt_to_ts(datetime.datetime(1965, 11, 6, tzinfo=utc)),
             gabby.dt_to_ts(datetime.datetime(1965, 11, 5, tzinfo=utc))]
        A = [649, 650]
        P = [399, 400]
        T = [gabby.keplerian_period(A[i], P[i]) for i in range(L)]

        faker._fill_apt(frag, zip(t, A, P, T))

        # The loaded values should be sorted
        txn = faker.db.txn()
        wat = faker.db.load_apt([frag])
        tr, Ar, Pr, Tr, Nr = wat

        A.sort(reverse=True)
        P.sort(reverse=True)
        T.sort(reverse=True)

        for i in range(L):
            assert(A[i] == Ar[0][i])
            assert(P[i] == Pr[0][i])
            assert(abs(T[i] -Tr[0][i]) < 1e-4)

