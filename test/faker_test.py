#!/usr/bin/env python

import datetime
import lmdb
import numpy as np
import pprint
import pytest
import sys

import gabby

class TestFaker(object):

    def test_fill_apt(self):
        faker = gabby.FakeDB(None, None)

        L = 2
        frag = '99025'
        utc = datetime.timezone.utc
        t = [gabby.dt_to_ts(datetime.datetime(1965, 11, 6, tzinfo=utc)),
             gabby.dt_to_ts(datetime.datetime(1965, 11, 5, tzinfo=utc))]
        A = [649, 650]
        P = [399, 400]
        T = [gabby.keplerian_period(A[i], P[i]) for i in range(L)]

        faker._fill_apt(frag, zip(t, A, P, T))

        for v in zip(t, A, P, T): print(v)

        # The loaded values should be sorted
        txn = lmdb.Transaction(faker.full_env, write=False)
        wat = gabby.load_apt([frag], txn, faker.full_apt)
        pprint.pprint(wat)
        tr, Ar, Pr, Tr, Nr = wat

        A.sort(reverse=True)
        P.sort(reverse=True)
        T.sort(reverse=True)

        for i in range(L):
            assert(A[i] == Ar[0][i])
            assert(P[i] == Pr[0][i])
            assert(abs(T[i] -Tr[0][i]) < 1e-4)

