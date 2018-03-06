import os
import json
import numpy as np
from warnings import warn
from .dataset import Dataset
from .attribute_manager import AttributeManager


class Base(object):
    default_dataset_chunk_size = 64
    default_dataset_dtype = np.dtype('float32')

    def __init__(self, path, is_zarr=True):
        self.path = path
        self.is_zarr = is_zarr
        self._attrs = AttributeManager(path, is_zarr)

    @property
    def attrs(self):
        return self._attrs

    # FIXME this is not what we wan't, because
    # a) we want to have the proper python key syntax
    # b) this will not list nested paths in file properly,
    # like 'volumes/raw'
    def keys(self):
        return os.listdir(self.path)

    def __contains__(self, key):
        return os.path.exists(os.path.join(self.path, key))

    # TODO open_dataset, open_group and close_group should also be implemented here

    def create_dataset(self, key, dtype=None, shape=None, chunks=None,
                       fill_value=0, compression='raw', data=None,
                       **compression_options):
        assert key not in self.keys(), "Dataset is already existing"

        if data is None:
            assert shape is not None, "Datasets must be given a shape"
            if chunks is None:
                chunks = tuple(min(s, self.default_dataset_chunk_size) for s in shape)
            if dtype is None:
                warn('Data type was not given, using ' + str(self.default_dataset_dtype))
                dtype = self.default_dataset_dtype
        else:
            data_chunks = getattr(data, 'chunks', None)
            data_dtype = getattr(data, 'dtype', None)
            data_shape = getattr(data, 'shape', None)

            if data_dtype is None or data_shape is None:
                data = np.asarray(data)
                data_dtype = data.dtype
                data_shape = data.shape

            if dtype is None:
                dtype = data_dtype
            else:
                assert dtype == data_dtype, "Given dtype ({}) conflicts with type of given data ({})".format(
                    dtype, data_dtype
                )  # could coerce instead

            if shape is None:
                shape = data_shape
            else:
                assert shape == data_shape, "Given shape ({}) conflicts with shape of given data ({})".format(
                    shape, data_shape
                )

            chunks = chunks or data_chunks
            if chunks is None:
                chunks = tuple(min(s, self.default_dataset_chunk_size) for s in shape)

        path = os.path.join(self.path, key)
        ds = Dataset.create_dataset(path, dtype, shape,
                                      chunks, self.is_zarr,
                                      compression, compression_options,
                                      fill_value)

        if data is None:
            return ds

        ds[:] = data
        return ds

    def is_group(self, key):
        path = os.path.join(self.path, key)
        if self.is_zarr:
            return os.path.exists(os.path.join(path, '.zgroup'))
        else:
            meta_path = os.path.join(path, 'attributes.json')
            if not os.path.exists(meta_path):
                return True
            with open(meta_path, 'r') as f:
                # attributes for n5 file can be empty which cannot be parsed by json
                try:
                    attributes = json.load(f)
                except ValueError:
                    attributes = {}
            # The dimensions key is only present in a dataset
            return 'dimensions' not in attributes
