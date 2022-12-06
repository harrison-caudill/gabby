#!/usr/bin/env python

import numpy as np
import pprint
import pytest

from .fixtures import *

from ..defs import *


class TestMoralDecay(object):

    def test_rates(self, static_decay):
        print()
        print(static_decay.boundaries_Ap)
