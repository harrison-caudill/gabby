#!/usr/bin/env python

import configparser
import os
import numpy as np
import pytest
import tempfile

from ..cache import GabbyCache
from ..faker import FakeDB
from ..undertaker import Undertaker
from ..transformer import Jazz
from ..moral_decay import MoralDecay


@pytest.fixture
def cache():
    td = tempfile.TemporaryDirectory()
    tmp = GabbyCache(td.name)
    setattr(tmp, '__tmpdir', td)
    return tmp


@pytest.fixture
def cfg():
    path = os.path.join(os.path.dirname(__file__), 'test.cfg')
    retval = configparser.ConfigParser(allow_no_value=True)
    retval.read(path)
    return retval


@pytest.fixture
def single_faker(cfg):
    return FakeDB(cfg, cfg['db-single'])

@pytest.fixture
def double_faker(cfg):
    retval = FakeDB(cfg, cfg['db-double'])
    retval.build_linear()
    retval.build_scope()
    return retval

@pytest.fixture
def linear_faker(cfg):
    retval = FakeDB(cfg, cfg['db-single-linear'])
    retval.build_linear()
    return retval

@pytest.fixture
def jazzercise(cfg):
    return Jazz(cfg['stats-test-1'])

@pytest.fixture
def undertaker(single_faker):
    return Undertaker(single_faker.db)

@pytest.fixture
def noop_decay():
    decay_hist = np.ones((2, 2, 2, 2))

    Ap_min = 0
    Ap_max = 10000
    dAp = (Ap_max - Ap_min) / 2

    Ad_min = 0
    Ad_max = 0
    dAd = (Ad_max - Ad_min) / 2

    Pp_min = 0
    Pp_max = 10000
    dPp = (Pp_max - Pp_min) / 2

    Pd_min = 0
    Pd_max = 0
    dPd = (Pd_max - Pd_min) / 2

    return MoralDecay(decay_hist,
                      Ap_min, Ap_max, dAp, Ad_min, Ad_max, dAd,
                      Pp_min, Pp_max, dPp, Pd_min, Pd_max, dPd)

@pytest.fixture
def static_decay():
    decay_hist = np.ones((2, 2, 2, 2))

    Ap_min = 0
    Ap_max = 10000
    dAp = (Ap_max - Ap_min) / 2

    Ad_min = 0
    Ad_max = 0
    dAd = (Ad_max - Ad_min) / 2

    Pp_min = 0
    Pp_max = 10000
    dPp = (Pp_max - Pp_min) / 2

    Pd_min = -50
    Pd_max = -100
    dPd = (Pd_max - Pd_min) / 2

    return MoralDecay(decay_hist,
                      Ap_min, Ap_max, dAp, Ad_min, Ad_max, dAd,
                      Pp_min, Pp_max, dPp, Pd_min, Pd_max, dPd)
