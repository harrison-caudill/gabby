#!/usr/bin/env python

import datetime
import lmdb
import numpy as np
import pprint
import pytest
import sys

import gabby
from fixtures import *

from gabby_plotter import GabbyDataModel

class TestGabbyDataModel(object):

    def test_basic_params(self, cfg, single_faker):
        """Checks the basic interface-required values (tAPT, N, ...)
        """

        tgt = cfg['gabby-test-1']
        model = GabbyDataModel(tgt)

        # Verify basic config file ingestion here
        utc = datetime.timezone.utc
        assert(model.start_d == datetime.datetime(2006, 12, 1, tzinfo=utc))
        assert(model.end_d == datetime.datetime(2020, 10, 1, tzinfo=utc))
        assert(1 == model.dt.days)
        assert(0 == model.dt.seconds)

        model.fetch_from_db(single_faker.db)

        # Verify correct values from the db here
        
        assert(False)

        self.L = None
        self.N = None
        self.names = None

        # Integer timestamps of each sample (can also be derived from
        # the datetime and timedelta objects above).
        self.ts = None

        # APT values for the fragments, 0's for invalid/unused
        # numpy arrays of shape (N, L) and dtype np.float32
        self.As = None
        self.Ps = None
        self.Ts = None

        # Logical true/false for which samples are valid
        # numpy array of shape (N, L) and dtype np.int8
        self.valid = None

        # Number of fragments in scope at any given time sample.
        # Derivable from valid with np.sum(valid, axis=1)
        # numpy array of shape (N)
        self.Ns = None


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

        for v in zip(t, A, P, T): print(v)

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

