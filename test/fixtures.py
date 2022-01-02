#!/usr/bin/env python

import configparser
import lmdb
import os
import tempfile

@pytest.fixture
def cfg():
    path = os.path.join(os.path.dirname(__file__), 'test.cfg')
    retval = configparser.ConfigParser(allow_no_value=True)
    retval.read(path)
    return retval
