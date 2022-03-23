#!/usr/bin/env python

import copy
import datetime
import numpy as np
import pprint
import pytest
import tempfile

from .fixtures import *
from ..utils import *
from ..propagator import StatsPropagator
from ..gabby_data_model import GabbyDataModel


class TestPropagator(object):

    def test_basic(self, cfg, noop_decay):
        db_tgt = cfg['db-event-test-basic']
        db = FakeDB(cfg, db_tgt)

        des = ["99025A", "99025B", "99025C", "99025D", "99025E"]
        base = [0, 86400, 172800, 259200, 345600]

        t = [
            # on the boundaries, all frames
            base,

            # late start
            [172800, 259200, 345600],

            # late start
            [172800, 259200, 345600],

            # late start
            [172800, 259200],

            # late start
            [172800, 259200, 345600],
            ]

        A = [
            # omnipresent
            [500, 500, 500, 500, 500],

            # late start, all valid
            [500, 500, 500],

            # late start, proper decay
            [500, 300, 100],

            # late start, early decay
            [500, 500],

            # late start, proper decay, missing middle
            [500, -100, 100],
            ]

        P = copy.deepcopy(A)
        T = copy.deepcopy(A)

        db.build_manual(des, t, A, P, T)

        data = GabbyDataModel.from_db(db=db.db,
                                      des=des,
                                      start_d=ts_to_dt(base[0]),
                                      end_d=ts_to_dt(base[-1]),
                                      dt_d=datetime.timedelta(days=1))
        print("==============================================================")
        print("Prop Test")
        print("==============================================================")

        print(data.Vs)

        prop = StatsPropagator(noop_decay)

        incident_d = ts_to_dt(base[1])
        prop.propagate(data, incident_d,
                       drop_early=False,
                       fwd=True,
                       rev=True,
                       prop_after_obs=False,
                       n_threads=1,
                       decay_alt=200)
        print(data.Vs)
