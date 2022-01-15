#!/usr/bin/env python

import datetime
import lmdb
import numpy as np
import os
import pprint
import pytest
import sys

from .fixtures import *
from ..utils import *
from ..gabby_plotter import GabbyDataModel
from ..gabby_plotter import GabbyPlotter

class TestGabbyDataModel(object):

    def test_basic_params(self, cfg, double_faker):
        """Checks the basic interface-required values (tAPT, N, ...)
        """

        tgt = cfg['gabby-test-2']
        model = GabbyDataModel(tgt)

        # Verify basic config file ingestion here
        utc = datetime.timezone.utc
        assert(model.start_d == datetime.datetime(2006, 12, 1, tzinfo=utc))
        assert(model.end_d == datetime.datetime(2006, 12, 10, tzinfo=utc))
        assert(1 == model.dt.days)
        assert(0 == model.dt.seconds)

        model.fetch_from_db(double_faker.db)

        assert(model.L == 2)
        assert(model.N == 10)

        assert(sorted(model.names) == ["99025A", "99025B"])

        dt = datetime.timedelta(days=1).total_seconds()
        ts = np.arange(dt_to_ts(model.start_d),
                       dt_to_ts(model.end_d) + dt,
                       dt)

        assert(np.all(ts == model.ts))

        assert(model.ts.shape == (10,))
        assert(model.As.shape == (10, 2))
        assert(model.Ps.shape == (10, 2))
        assert(model.Ts.shape == (10, 2))

        assert(model.As[0][0] == 650)
        assert(model.As[-1][0] == 100)
        assert(model.As[0][1] == 600)
        assert(model.As[4][1] == 100)

        assert(model.Ps[0][0] == 400)
        assert(model.Ps[-1][0] == 100)
        assert(model.Ps[0][1] == 400)
        assert(model.Ps[4][1] == 100)

        # Logical true/false for which samples are valid
        # numpy array of shape (N, L) and dtype np.int8
        valid = np.ones((10, 2), dtype=np.int8)
        valid[5:,1] = np.zeros(5, dtype=np.int8)
        assert(np.all(model.valid == valid))

        Ns = np.ones(10) + np.concatenate([np.ones(5), np.zeros(5)])
        assert(np.all(model.Ns == Ns))

    def test_img_generation_human_review(self, cfg, double_faker):
        tmpdir = '/Users/kungfoo/tmp'
        p = GabbyPlotter(cfg=cfg,
                         tgt=cfg['gabby-test-2'],
                         img_dir=tmpdir,
                         output_dir=tmpdir,
                         cache_dir=tmpdir,
                         db=double_faker.db)

        # Uncomment for human inspection
        # p.plot()

        pass
