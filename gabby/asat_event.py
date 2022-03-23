import hashlib
import numpy as np

from .utils import *
from .defs import *
from .gabby_data_model import GabbyDataModel
from .propagator import StatsPropagator

class ASATEvent(object):

    def __init__(self, names, As, Ps, Ts):
        """ASAT Events hold the inital status of the debris cloud

        names: international designators of the L fragments
        [APT]s: np.array's with A, P, and T values of size L
        """
        self.names = names
        self.As = As
        self.Ps = Ps
        self.Ts = Ts

    @classmethod
    def cache_name(cls, des):
        m = hashlib.sha256()
        m.update((','.join(sorted(des))).encode())
        digest = m.hexdigest()
        return f"event-{digest}"

    @classmethod
    def from_db(cls, des, db, alive_date, moral_decay, decay_alt, incident_d,
                cache=None,
                rest_in_plasma=400,
                n_threads=1):
        """Loads the data from an ASAT event from the DB.

        Fragments are released in batches, and there are two issues to
        consider:

        1. It typically takes some time to Find a fragment, so the
           first observation is rarely close to the event.

        2. Sometimes, multiple fragments seem to result in a single
           fragment designation (guessing the RADAR cross section of
           multiple fragments is attributed to a single designation).
           The apparent eresult is that sometime later fragments are
           added and the original fragment appears to decay with a
           potentially very high perigee.

        We can address point 1 by reverse-propagating all of the
        fragments to the event date.  That should give us a good
        approximation of the initial conditions.

        We can address poing 2 by ignoring any decayed fragments whose
        last-observed perigee is above 400km.

        In order to determine which fragments are alive and which are
        not, we need the alive_date.  Any fragment which has
        observations AFTER that date are considered to be still alive,
        whereas fragments whose last observation is BEFORE that date
        are said to have decayed.  That allows us to determine which
        fragments have decayed at a high perigee so we can determine
        which fragments to ignore.  This value is computable, but
        would require scanning the entire scope database and looking
        at the maximum value and even then it's not necessarily good
        (for example, we have more recent observations from the nudol
        incident than for the DB at large at the time of writing this
        docstring which means that if we looked at the largest numbers
        we would conclude that everything except the nudol debris had
        decayed and we would ignore most of the Fengyun debris which
        is still up there).  This value should be selected on a
        per-event basis based upon the recency of the data and the
        observational frequency.  I recommend starting with something
        like (<date-of-data-fetch> - 1 week).

        rest_in_plasma is the altitude (in kilometers) at which it is
        considered acceptable to decay.  Anything below that and we
        consider it to have been a false read (guessing it was two or
        more fragments whose collective RADAR cross section was
        mistaken as one big fragment).

        des: ['des1', 'des2', ..., 'desN']
        db: GabbyDB
        alive_date: zone-aware datetime
        rest_in_plasma: 400
        """

        # Check the cache
        cache_name = cls.cache_name(des)
        if cache and cache_name in cache:
            logging.info(f"  Found event in the local cache.")
            return cache[cache_name]

        # So we don't have to build a bunch of them
        txn = db.txn(write=False)

        # Find the timestamps of the first and last observations
        scope_start, scope_end = db.load_scope(des)

        # Find the full list of daughter fragments in sorted order
        fragments = sorted(list(scope_start.keys()))

        # Get an independent list of start/end sorted by fragment number
        L = len(fragments)
        start = np.array([scope_start[f] for f in fragments], dtype=int)
        end = np.array([scope_end[f] for f in fragments], dtype=int)

        # Find the decay altitude for each of the fragments under
        # consideration
        decay_alt = np.zeros(L, dtype=np.float32)
        still_alive = np.zeros(L, dtype=np.int8)
        early_to_bed = np.zeros(L, dtype=np.int8)
        alive_ts = dt_to_ts(alive_date)
        for i in range(L):
            if scope_end[fragments[i]] > alive_ts:
                still_alive[i] = 1
            else:
                decay_alt[i] = db.get_latest_apt(txn, fragments[i])[1]
                if decay_alt[i] > rest_in_plasma:
                    early_to_bed[i] = 1

        # The propagator uses the GabbyDataModel object, so let's
        # build one
        logging.info("  Building Data Model to back propagate to start")
        start_d = ts_to_dt(np.min(start))
        end_d = ts_to_dt(np.max(start) + 1)
        model = GabbyDataModel.from_db(db=db,
                                       des=des,
                                       start_d=start_d,
                                       end_d=end_d,
                                       dt_d=datetime.timedelta(days=1))
        assert(model.fragments == fragments)

        # Done with our DB work
        txn.commit()
        txn = None

        # Build the stats propagator
        stat = StatsPropagator(moral_decay, decay_alt)

        # Back propagate all of the fragments to the incident date so
        # we have the expected initial conditions.
        logging.info("  Back propagating to the start")
        stat.propagate(model, incident_d, drop_early=True,
                       fwd=False, rev=True, n_threads=n_threads)

        # Our new value of L is limited to the satellites that do not
        # suffer from premature plasmafication.
        L -= np.sum(early_to_bed)

        # Now that we've back-propagated to the beginning, we should
        # have valid observations for inital conditions of all the
        # fragments.
        rFragments = []
        rA = []
        rP = []
        rT = []
        for i in range(model.L):
            if early_to_bed[i]: continue
            rFragments.append(fragments[i])
            rA.append(model.As[0][i])
            rP.append(model.Ps[0][i])
            rT.append(model.Ts[0][i])

        retval = ASATEvent(rFragments,
                           np.array(rA),
                           np.array(rP),
                           np.array(rT))

        if cache:
            logging.info("  Caching results")
            cache[cache_name] = retval

        return retval
