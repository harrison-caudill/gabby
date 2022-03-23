from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


class DoomsdayVector(object):
    """Has all the info about a single event.

    Multiple vectors can be combined.
    """

    def __init__(self):
        assert(False)


class DoomsdayPlotContext(object):

    def __init__(self):
        self.Xt = np.arange(self.data.start_d,
                            self.data.end_d+self.data.dt,
                            self.data.dt)


class DoomsdayPlotter(object):

    def __init__(self,
                 cfg=None,
                 tgt=None,
                 output_dir=None,
                 cache_dir=None,
                 data=None,
                 db=None):
        self.cfg = cfg
        self.tgt = tgt
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.data = data
        self.db = db

        self.cache = GabbyCache(cache_dir) if cache_dir else None

    def run_sim(self):
        self.rs = RandomState(MT19937(SeedSequence(seed)))
        self.Xt = np.arange(self.data.start_d,
                            self.data.end_d+self.data.dt,
                            self.data.dt)
        self.N = len(Xt)
        indexes = self._sample()

    def _sample(self):
        n = math.floor(samp_rate * (end - start).days / 365.0)
        if rs.choice([True, False])[0]: n += 1
        return rs.choice(range(self.N))
