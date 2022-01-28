#!/usr/bin/env python

import numpy as np
import os
import pprint
import pytest
import tempfile

from .fixtures import *


class Foo(object):
    pass


class TestCache(object):

    def test_all_the_things(self, cache):
        assert(os.path.isdir(cache.path))

        # Hello array
        data = [
            np.zeros((2, 2,), dtype=np.float32),
            np.zeros(1, dtype=np.int) + 1
            ]

        foo = Foo()
        setattr(foo, 'A', data)
        setattr(foo, 'B', "whatever")

        name = 'hello-obj'
        cache.put(name, foo)

        assert(cache.is_cached(name))
        bar = cache.get(name)

        for i in range(len(data)):
            assert((foo.A[i]==bar.A[i]).all())
        assert(foo.B == bar.B)

        # Make sure clearing works
        cache.clear_entry(name)
        assert(not cache.is_cached(name))

        # Make sure overwrite works
        cache.put(name, foo)
        setattr(foo, 'C', np.zeros((2, 2,), dtype=np.float32) + 2)
        setattr(foo, 'D', np.zeros(1, dtype=np.int) + 3)
        setattr(foo, 'baz', 'quux')
        cache.put(name, foo, overwrite=True)
        bar = cache.get(name)
        assert(bar.baz == 'quux' == foo.baz)
        assert(np.all(foo.C == bar.C))
        assert(np.all(foo.D == bar.D))

        # Make sure not-overwrite works
        setattr(foo, 'E', np.zeros(1, dtype=np.int) + 4)
        cache.put(name, foo, overwrite=False)
        bar = cache.get(name)
        assert('E' not in vars(bar))
