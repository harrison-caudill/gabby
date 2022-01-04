#!/usr/bin/env python

import configparser
import os
import pytest
import tempfile

import gabby.cache


@pytest.fixture
def cache():
    td = tempfile.TemporaryDirectory()
    tmp = gabby.cache.GabbyCache(td.name)
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
    return gabby.FakeDB(cfg, cfg['db-single'])


@pytest.fixture
def jazzercise():
    jazz = gabby.Jazz(None, None, None, None, None)

@pytest.fixture
def undertaker(single_faker):
    return gabby.Undertaker(single_faker.db)
