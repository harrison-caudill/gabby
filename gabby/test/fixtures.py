#!/usr/bin/env python

import configparser
import os
import pytest
import tempfile

from ..cache import GabbyCache
from ..faker import FakeDB
from ..undertaker import Undertaker
from ..transformer import Jazz


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
