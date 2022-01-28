


class GabbyData(object):
    """Data model for Gabby Plots.

    We break this out for a few reasons.  Image generation needs to be
    done with hard processes because matplotlib is not thread safe, so
    we need our data model to be serializable and not dependent upon
    matplotlib.  It's also nice to be able to easily unit-test things.

    
    """


    

class FragmentData(object):
    """The raw data for a debris cloud throughout a gabby animation.
    """

    def __init__(self,
                 names=None,
                 ts=None,
                 As=None,
                 Ps=None,
                 Ts=None,
                 valid=None,
                 Ns=None):
        """
        ts: timestamps [NxL]
        As: apogees [NxL]
        Ps: perigees [NxL]
        Ts: periods [NxL]
        valid: Whether the element is valid [NXL]
        Ns: number of valid fragments at any given offset
        """

        # tAPT values as NxL arrays.
        self.ts = ts
        self.As = As
        self.Ps = Ps
        self.Ts = Ts

        # Valid: inidcates with a 1 or a 0 whether or not the given
        # tAPT value is valid.
        self.valid = valid

        # Useful for the plot of total number of fragments.  This
        # array is equivalent to summing Ns[i] = sum(valid[i])
        self.Ns = Ns

        self.N = len(ts)
        assert(len(ts) == len(As) == len(Ps) == len(Ts))


class GabbyPlotContext(object):
    """Easily serializeable object with necessary data for plotting.

    We break this out into a separate and object to make it easier to
    use multiprocessing to process images.  For convenience, we place
    the logic to load the data in here.  It also makes unit tests
    easier as we don't have to entangle the logic for matplotlib with
    the logic for the data.
    """

    def load_from_db(self, db):
        """Loads the data from the database.
        """

        # First, find when pieces come into scope and when they go out
        self.scope_start, self.scope_end = db.load_scope(self.target_des)

        # The results will be sorted in this order
        fragments = sorted(list(self.scope_start.keys()))
        fragments = [f for f in fragments if f not in self.mask]
        self.fragments = fragments

        # Basic read-only transaction
        txn = db.txn()

        # Initialize our main cursor
        cursor = txn.cursor(db=db.db_apt)

        # Get our array dimensions
        L = len(fragments)
        N = int(math.ceil((self.end_d - self.start_d)/self.dt))+1

        self._load_comps(db, txn)

        logging.info(f"  Shape of data: {N}, {L}")

        # Get the integer timestamps to use for indexing
        start_ts = dt_to_ts(self.start_d)
        end_ts = dt_to_ts(self.end_d)
        dt_s = int(self.dt.total_seconds())
        timestamps = np.arange(start_ts, end_ts+dt_s, dt_s)
        assert(N == len(timestamps))

        # Before and after values for the timestamp and Apogee/Perigee
        before_ts = np.zeros((N, L), dtype=np.float32)
        tgt_ts = np.zeros((N, L), dtype=np.float32)
        after_ts = np.zeros((N, L), dtype=np.float32)

        before_A = np.zeros((N, L), dtype=np.float32)
        after_A = np.zeros((N, L), dtype=np.float32)

        before_P = np.zeros((N, L), dtype=np.float32)
        after_P = np.zeros((N, L), dtype=np.float32)

        before_T = np.zeros((N, L), dtype=np.float32)
        after_T = np.zeros((N, L), dtype=np.float32)

        # The big main loop
        for i in range(L):
            frag = fragments[i]
            logging.info(f"    Fetching data for {frag} ({i+1}/{L})")

            # Loop through the observations
            prefix = (frag+',').encode()
            cursor.set_range(prefix)

            # The starting value we linearly interpolate from
            prev_v = None
            prev_ts = 0

            # The next targeted timestamp
            next_ts = start_ts

            k, v, = cursor.item()
            j = 0
            while True:
                if not k.startswith(prefix): break
                ts = int(k[-12:])
                if ts > end_ts: break

                if ts < next_ts:
                    prev_ts = ts
                    prev_v = v
                    cursor.next()
                    k, v, = cursor.item()
                else:
                    after_ts[j][i] = ts
                    tgt_ts[j][i] = next_ts
                    if prev_v:
                        A, P, T = unpack_apt(prev_v)
                        before_A[j][i] = A
                        before_P[j][i] = P
                        before_T[j][i] = T
                        before_ts[j][i] = prev_ts
                        A, P, T = unpack_apt(v)
                        after_A[j][i] = A
                        after_P[j][i] = P
                        after_T[j][i] = T

                    next_ts += dt_s
                    j += 1

        logging.info(f"  Computing temporal offsets")
        dt = after_ts - before_ts
        off_before = (tgt_ts - before_ts)/dt
        off_after = (after_ts - tgt_ts)/dt

        logging.info(f"  Interpolating Values")
        t = tgt_ts
        A = before_A * off_before + after_A * off_after
        P = before_P * off_before + after_P * off_after
        T = before_T * off_before + after_T * off_after

        logging.info(f"  Computing validity and data lengths")
        valid = np.copy(A)
        valid = np.where(np.logical_not(np.isnan(valid)), 1, 0)
        valid = valid.astype(np.int8)
        Ns = np.sum(valid, axis=1, dtype=np.int64)

        logging.info(f"  Removing the nans")
        A = np.where(np.isnan(A), 0, A)
        P = np.where(np.isnan(A), 0, P)
        T = np.where(np.isnan(A), 0, T)

        # We're done with our read-only transaction now
        txn.commit()

        logging.info(f"  Registering the resulting data")
        self.data = FragmentData(names=fragments,
                                 ts=t,
                                 As=A,
                                 Ps=P,
                                 Ts=T,
                                 valid=valid,
                                 Ns=Ns)

    def _load_plot_parameters(self):
        # The list of base satellites who's daughters are to be
        # tracked.  It has to be a list so that we can handle
        # collisions and intentional detonations.
        self.target_des = json.loads(self.tgt['intldes'])

        # We're going to store the images in a separate directory for
        # cleanliness
        self.img_dir = os.path.join(self.output_dir, 'gabby-img')

        # Pull in the time boundaries from the config file
        self.start_d = parse_date_d(self.tgt['start-date'])
        self.end_d = parse_date_d(self.tgt['end-date'])

        # Mask off the original rocket body, as it's a distraction
        self.mask = json.loads(self.tgt['mask'])

        # Time step between images
        self.dt = datetime.timedelta(days=self.tgt.getint('plot-period'))

        # The datetime objects along the x-axis of the plot with the
        # number of fragments.
        self.Xt = np.arange(self.start_d, self.end_d+self.dt, self.dt)

        # The colors are meant to be interpreted directly by matplotlib
        self.apogee_color = self.tgt['apogee-color']
        self.perigee_color = self.tgt['perigee-color']

    def _load_comps(self, db, txn):

        # Loads the comparators
        static_comparators = json.loads(self.tgt['static-comparators'])
        logging.info(f"  Using static comparators: {static_comparators}")
        comparators = {}
        for comp in static_comparators:
            des, name, color, = comp
            cur = {'name': name,
                   'color': color,
                   'apt': tuple(db.get_latest_apt(txn, des)),
                   'is_static': True
                   }
            comparators[des] = cur
        self.comparators = comparators

    def plt_setup(self):
        # The legend is shared by all and customized.
        self.legend_contents = [
            matplotlib.lines.Line2D([0], [0],
                                    color='white',
                                    markerfacecolor=self.apogee_color,
                                    marker='.',
                                    markersize=12),
            matplotlib.lines.Line2D([0], [0],
                                    color='white',
                                    markerfacecolor=self.perigee_color,
                                    marker='.',
                                    markersize=12),
            ]
        self.legend_labels = [
            'Fragment Apogee',
            'Fragment Perigee',
            ]

        # Process the comparators
        comp_X = []
        comp_Y = []
        comp_C = []
        for des in self.comparators:
            cur = self.comparators[des]
            apt = cur['apt']
            color = cur['color']
            name = cur['name']
            A,P,T, = apt
            O = (A+P)/2
            legend_patch = matplotlib.lines.Line2D([0], [0],
                                                   color='white',
                                                   markerfacecolor=color,
                                                   marker='o',
                                                   markersize=12)
            self.legend_contents.append(legend_patch)
            self.legend_labels.append(name)
            comp_X.append(T)
            comp_Y.append(O)
            comp_C.append(color)

        self.comp_X = comp_X
        self.comp_Y = comp_Y
        self.comp_C = comp_C

    def __init__(self, tgt=None,
                 data=None, fragments=None,
                 output_dir=None, indexes=None):
        """
        tgt: configparser results for the plot section
        data: FragmentData
        fragments: [str, ...] list of fragment names
        start_d: starting datetime
        end_d: ending datetime
        output_dir: just what it says
        indexes: list of integer indexes for the images to build
        """

        self.tgt = tgt
        self.data = data
        self.fragments = fragments
        self.output_dir = output_dir
        self.indexes = indexes

        self._load_plot_parameters()

