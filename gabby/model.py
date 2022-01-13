
class DataModel(object):
    """Parent class for other DataModels

    Data models are used to fetch, transform, and plot data.  They're
    also the interface for propagators.
    """

    def __init__(self, *args, **kwargs):
        """The method of initialization is left to the subclasses.
        """

        # The total number of fragments
        # <int>
        self.L = None

        # The total number of frames in the gabby animation
        # <int>
        self.N = None

        # start/stop/step for the data
        # <datetime> & <timedelta>
        self.start_d = None
        self.end_d = None
        self.dt = None

        # International designators of the fragment
        # [<str>, ...]
        self.names = None

        # Integer timestamps of each sample (can also be derived from
        # the datetime and timedelta objects above).
        self.ts = None

        # APT values for the fragments, 0's for invalid/unused
        # numpy arrays of shape (N, L) and dtype np.float32
        self.As = None
        self.Ps = None
        self.Ts = None

        # Logical true/false for which samples are valid
        # numpy array of shape (N, L) and dtype np.int8
        self.valid = None

        # Number of fragments in scope at any given time sample.
        # Derivable from valid with np.sum(valid, axis=1)
        # numpy array of shape (N)
        self.Ns = None

        # For the moment, this is just documentation, I might get
        # around to using the abc module at some point.
        assert(False)

    def fetch_from_db(self, db, txn):
        """Fetches data from the DB.

        Any other data necessary for execution is assumed to have been
        passed in during object initialization.  All data resulting
        from this operation is assumed to be serializeable.  The db
        and txn should NOT be stored in the object.
        """

        # For the moment, this is just documentation, I might get
        # around to using the abc module at some point.
        assert(False)
