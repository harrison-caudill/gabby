#!/usr/bin/env python

import lmdb
import numpy as np
import os
import pickle
import pprint


class GabbyCacheEntry(object):

    def __init__(self, name, obj, keys):
        """
        name: cache entry name
        obj: user-supplied object (sans numpy arrays)
        keys: ordered list of keys of numpy objects we put into the cache
        """
        self.name = name
        self.obj = obj
        self.keys = keys

class GabbyCache(object):
    """On-disk caching system.

    pickle is hella slow compared to numpy serializing, so instead of
    using pickle to serialize everything, we use a combination of
    pickle and numpy.  Pickle works well for the metadata, but we'll
    want to use numpy for the data.  Since you may have multiple numpy
    arrays to save and dictionaries, we break out the caching
    capabilities.

    The data will be stored as follows:
    <cache-dir>/<name>/ent.pickle -- object(sans numpy) and cache metadata
    <cache-dir>/<name>/data.np -- numpy arrays found in that object
    """

    def __init__(self, path):
        """Does what every __init__ method does.

        path: The <cache-dir> from the class docstring.
        """
        self.path = path

    def _cache_path(self, name):
        """Constructs the directory where the <name> would be cached.
        """
        return os.path.join(self.path, name)

    def _ent_path(self, name):
        return os.path.join(self._cache_path(name), 'ent.pickle')

    def _data_path(self, name):
        return os.path.join(self._cache_path(name), 'data.np')

    def __setitem__(self, name, obj):
        return self.put(name, obj, overwrite=True)

    def put(self, name, obj, overwrite=True):
        """Adds the object to the cache.

        name: <str> the name to use when saving/referencing the data
        obj:  whatever picklable object you want
        overwrite: <bool>

        Defaults to overwriting any old cache values, but you can
        change that so it returns False and does nothing
        """

        if self.is_cached(name):
            if overwrite: self.clear_entry(name)
            else: return False

        os.mkdir(self._cache_path(name))


        with open(self._data_path(name), 'wb') as fd:
            keys = []
            mappings = {}
            if isinstance(obj, np.ndarray):
                np.save(fd, obj)
                keys.append(None) # None signifies a top-level array
            else:
                attrs = vars(obj)
                for k in attrs:
                    if isinstance(attrs[k], np.ndarray):
                        keys.append(k)
                        mappings[k] = getattr(obj, k)
                        np.save(fd, mappings[k])
                        setattr(obj, k, None)

                for k in mappings: setattr(obj, k, mappings[k])

            ent = GabbyCacheEntry(name, obj, keys)

        with open(self._ent_path(name), 'wb') as fd: pickle.dump(ent, fd)

        return True

    def __getitem__(self, name):
        return self.get(name)

    def get(self, name):
        """Retreives the cached object.

        returns obj
        """

        if not self.is_cached(name): return None

        ent = self._load_cache_entry(name)

        with open(self._data_path(name), 'rb') as fd:
            for key in ent.keys:
                if key is None:
                    # None means that it's a top-level numpy array
                    return np.load(fd)
                setattr(ent.obj, key, np.load(fd))

        return ent.obj

    def _load_cache_entry(self, name):
        with open(self._ent_path(name), 'rb') as fd: return pickle.load(fd)

    def clear_entry(self, name):
        """Deletes any cache entry for the name
        """
        if not self.is_cached(name): return

        ent = self._load_cache_entry(name)
        del_files = [self._ent_path(name), self._data_path(name)]
        for path in del_files:
            assert(os.path.isfile(path))
            os.unlink(path)
        os.rmdir(self._cache_path(name))

    def __contains__(self, name):
        return self.is_cached(name)

    def is_cached(self, name):
        """Determines whether or not the <name> is in the cache.
        """
        return os.path.isdir(self._cache_path(name))
