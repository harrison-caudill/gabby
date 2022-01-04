#!/usr/bin/env python

import numpy as np
import os
import pprint
import pytest
import tempfile

from fixtures import *


class TestCache(object):

    def test_all_the_things(self, cache):
        assert(os.path.isdir(cache.path))

        # Hello array
        data = [
            np.zeros((2, 2,), dtype=np.float32),
            np.zeros(1, dtype=np.int) + 1
            ]
        meta = {'foo':'bar'}
        name = 'hello-array'
        cache.put(name, meta, data)
        rmeta, rdata = cache.get(name)
        assert(rmeta == meta)
        for i in range(len(data)):
            assert((rdata[i]==data[i]).all())
        assert(cache.is_cached(name))

        # Make sure clearing works
        cache.clear_entry(name)
        assert(not cache.is_cached(name))

        # Hello dict
        data = {
            'b': np.zeros((2, 2,), dtype=np.float32),
            'a': np.zeros(1, dtype=np.int) + 1
            }
        meta = {'foo':'bar'}
        name = 'hello-dict'
        cache.put(name, meta, data)
        rmeta, rdata = cache.get(name)
        assert(rmeta == meta)
        for k in data:
            assert((rdata[k]==data[k]).all())

        # Make sure overwrite works
        data = {
            'c': np.zeros((2, 2,), dtype=np.float32) + 2,
            'd': np.zeros(1, dtype=np.int) + 3
            }
        meta = {'baz':'quux'}
        cache.put(name, meta, data)
        rmeta, rdata = cache.get(name)
        assert(rmeta == meta)
        for k in data:
            assert((rdata[k]==data[k]).all())

        # Make sure not-overwrite works
        new_data = {
            'e': np.zeros((2, 2,), dtype=np.float32) + 4,
            'f': np.zeros(1, dtype=np.int) + 5
            }
        new_meta = {'what':'ever'}
        cache.put(name, new_meta, new_data, overwrite=False)
        rmeta, rdata = cache.get(name)
        assert(rmeta == meta)
        for k in data:
            assert((rdata[k]==data[k]).all())
