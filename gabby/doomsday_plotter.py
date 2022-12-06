
import json
import math
import numpy as np

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from .moral_decay import MoralDecay
from .utils import *
from .transformer import Jazz
from .asat_event import ASATEvent
from .gabby_data_model import GabbyDataModel
from .propagator import StatsPropagator


class DoomsdayPlotContext(object):

    def __init__(self):
        pass


class DoomsdayPlotter(object):

    def __init__(self,
                 cfg=None,
                 tgt=None,
                 global_cache=None,
                 tgt_cache=None,
                 img_dir=None,
                 output_dir=None,
                 db=None):

        # Preserve the inputs...why does python not make this easier?
        self.cfg = cfg
        self.tgt = tgt
        self.global_cache = global_cache
        self.tgt_cache = tgt_cache
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.db = db

        # Initialize from the config file
        self._init_from_config()

    def _init_from_config(self):

        # Initialize a random-number generator
        self.seed = 0xcafebabe
        if 'random-seed' in self.cfg['general']:
            self.seed = int(self.cfg['general']['random-seed'], 16)
        self.rs = RandomState(MT19937(SeedSequence(self.seed)))

        # Timestamps of the window
        self.start_d = parse_date_d(self.tgt['start-date'])
        self.end_d = parse_date_d(self.tgt['end-date'])
        self.dt = datetime.timedelta(days=self.tgt.getint('plot-period'))
        self.Xt = np.arange(self.start_d,
                            self.end_d+self.dt,
                            self.dt)
        self.N = len(self.Xt)

        self.moral_decay = Jazz.moral_decay_from_cfg(self.cfg,
                                                     self.db,
                                                     cache=self.global_cache)

    def _load_events(self, n_threads=1):
        events = []
        for name in json.loads(self.tgt['events']):
            evt = self.cfg[f"event-{name}"]
            des = json.loads(evt['intldes'])
            cache_name = ASATEvent.cache_name(des)

            if self.global_cache and cache_name in self.global_cache:
                logging.info(f"Found {name} in cache")
                events.append(self.global_cache[cache_name])
            else:
                event = self._load_event(self.cfg,
                                         self.tgt,
                                         evt,
                                         self.global_cache,
                                         self.db,
                                         n_threads)
                if self.global_cache:
                    self.global_cache[cache_name] = event

                events.append(event)
        return events

    def _load_event(self, cfg, tgt, evt, cache, db, n_threads):
        name = evt.name[len('event-'):]
        des = json.loads(evt['intldes'])
        alive_date = parse_date_d(evt['alive-date'])

        logging.info(f"Loading ASAT Event: {name}")
        dan = ASATEvent.from_db(des,
                                db,
                                alive_date,
                                self.moral_decay,
                                tgt.getint('decay-alt'),
                                parse_date_d(evt['incident']),
                                cache=cache,
                                n_threads=n_threads)
        return dan

    @classmethod
    def data_cache_name(cls, tgt):
        return f"doom-data-{cfg_hash(tgt)}"

    def run_sim(self, n_threads=1):

        cache_name = self.data_cache_name(self.tgt)
        if self.tgt_cache and cache_name in self.tgt_cache:
            logging.info(f"Found doomsday data in cache")
            return self.tgt_cache[cache_name]

        events = self._load_events(n_threads=n_threads)
        indexes = self._sample()

        dt_s = int(self.dt.total_seconds())
        ts = np.arange(dt_to_ts(self.start_d),
                       dt_to_ts(self.end_d) + dt_s,
                       dt_s)
        N = len(ts)
        assert(N == self.N)

        # determine the ordering of events
        q = []
        for idx in indexes: q.append((idx, self.rs.choice(events)))

        # Initialize the empty gabby model
        L = sum([e[-1].L for e in q])
        As = np.zeros((N, L), dtype=np.float32)
        Ps = np.zeros((N, L), dtype=np.float32)
        Ts = np.zeros((N, L), dtype=np.float32)
        Ns = np.zeros(N, dtype=int)
        Vs = np.zeros((N, L), dtype=np.int8)
        dt = self.dt

        # Add in the events
        l = 0 # Current index into the fragments of the model
        fragments = []
        for idx, evt in q:
            print(f"Adding at time index {idx} fragment index {l}")
            dl = evt.L
            As[idx,l:l+dl] = evt.As
            Ps[idx,l:l+dl] = evt.Ps
            Ts[idx,l:l+dl] = evt.Ts
            Vs[idx,l:l+dl] = 1
            l += dl

            # FIXME: Consider adding new namespaces for these
            fragments += evt.names

        assert(l == L)
        model = GabbyDataModel(fragments, ts, As, Ps, Ts, Ns, Vs, dt)

        prop = StatsPropagator(self.moral_decay)

        prop.propagate(model,
                       self.start_d, # FIXME: not used
                       drop_early=False,
                       fwd=True,
                       rev=False,
                       prop_after_obs=False,
                       n_threads=n_threads)

        if self.tgt_cache:
            logging.info(f"Adding doomsday data to cache")
            self.tgt_cache[cache_name] = model

        return model

    def _sample(self):
        samp_rate = self.tgt.getfloat('frequency')
        n = math.floor(samp_rate * (self.end_d - self.start_d).days / 365.0)
        if self.rs.choice([True, False]): n += 1
        retval = np.sort(self.rs.choice(range(self.N), n))
        return retval
