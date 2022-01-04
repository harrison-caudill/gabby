#!/usr/bin/env python

import lmdb
import numpy as np
import os
import pickle
import pprint


class GabbyCacheEntry(object):

    def __init__(self, name, meta, mappings, is_dict):
        self.name = name
        self.meta = meta
        self.mappings = mappings
        self.is_dict = is_dict

class GabbyCache(object):
    """On-disk caching system.

    pickle is hella slow compared to numpy serializing, so instead of
    using pickle to serialize everything, we use a combination of
    pickle and numpy.  Pickle works well for the metadata, but we'll
    want to use numpy for the data.  Since you may have multiple numpy
    arrays to save and dictionaries, we break out the caching
    capabilities.

    The data will be stored as follows:
    <cache-dir>/<name>/meta.pickle -- user-supplied metadata
    <cache-dir>/<name>/data.np -- numpy arrays
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

    def _meta_path(self, name):
        return os.path.join(self._cache_path(name), 'metadata.pickle')

    def _data_path(self, name):
        return os.path.join(self._cache_path(name), 'data.np')

    def put(self, name, metadata, data, overwrite=True):
        """Adds the metadata and list of numpy arrays to the cache.

        name: <str> the name to use when saving/referencing the data
        metadata: <dict> whatever picklable metadata you want
        data: Either an array or a dictionary of numpy arrays
        overwrite: <bool>

        The <data> entry can be either a dictionary or a list so
        that you can name them if you want.  We'll just use numbers as
        the implied names of the arrays when saving to disk, then
        reconstruct the array as needed during a cache get.

        Defaults to overwriting any old cache values, but you can
        change that so it silently does nothing.
        """

        if self.is_cached(name):
            if overwrite: self.clear_entry(name)
            else: return

        os.mkdir(self._cache_path(name))

        is_dict = isinstance(data, dict)

        if is_dict: mappings = sorted(data.keys())
        else: mappings = list(range(len(data)))

        ent = GabbyCacheEntry(name, metadata, mappings, is_dict)
        with open(self._meta_path(name), 'wb') as fd: pickle.dump(ent, fd)

        with open(self._data_path(name), 'wb') as fd:
            for key in mappings: np.save(fd, data[key])

    def get(self, name):
        """Retreives the cached meta/data.

        Whether you passed in an array or a dictionary for the data,
        it'll be reloaded in the same order.

        returns meta, data
        """

        if not self.is_cached(name): return None, None

        ent = self._load_cache_meta(name)
        with open(self._meta_path(name), 'rb') as fd: ent = pickle.load(fd)

        if ent.is_dict: data = {}
        else: data = [None for i in range(len(ent.mappings))]

        with open(self._data_path(name), 'rb') as fd:
            for key in ent.mappings: data[key] = np.load(fd)

        return ent.meta, data

    def _load_cache_meta(self, name):
        with open(self._meta_path(name), 'rb') as fd: return pickle.load(fd)

    def clear_entry(self, name):
        """Deletes any cache entry for the name
        """
        if not self.is_cached(name): return

        ent = self._load_cache_meta(name)
        del_files = [self._meta_path(name), self._data_path(name)]
        for path in del_files:
            assert(os.path.isfile(path))
            os.unlink(path)
        os.rmdir(self._cache_path(name))

    def is_cached(self, name):
        """Determines whether or not the <name> is in the cache.
        """
        return os.path.isdir(self._cache_path(name))
