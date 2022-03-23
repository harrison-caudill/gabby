#!/usr/bin/env python

import datetime
import numpy as np
import pprint
import pytest
import tempfile

from .fixtures import *
from ..utils import *
from ..asat_event import ASATEvent


class TestEvent(object):

    def test_basic(self, cfg, static_decay):
        db_tgt = cfg['db-event-test-basic']
        db = FakeDB(cfg, db_tgt)
        db.build_manual()

        event = ASATEvent.from_db(["99025"],
                                  db.db,
                                  ts_to_dt(259200-1),
                                  static_decay,
                                  200,
                                  ts_to_dt(1))
        assert(event.names == ["99025A", "99025C", "99025D"])

        assert(event.As[0] == 450)
        assert(abs(event.As[1] - 850) < 5)
        assert(event.As[2] == 400)

        assert(event.Ps[0] == 425)
        assert(abs(event.Ps[1] - 850) < 5)
        assert(event.Ps[2] == 400)
