import unittest
import sys
import warnings

import numpy as np
import os
from shutil import rmtree

try:
    import z5py
except ImportError:
    sys.path.append('..')
    import z5py


class ChunkedDataset(np.ndarray):
    pass


def chunked_dataset(arr, chunks):
    instance = arr.view(ChunkedDataset)
    instance.chunks = chunks
    return instance


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.shape = (100, 100, 100)
        self.ff_zarr = z5py.File('array.zr', True)
        self.ff_zarr.create_dataset(
            'test', dtype='float32', shape=self.shape, chunks=(10, 10, 10)
        )
        self.ff_n5 = z5py.File('array.n5', False)
        self.ff_n5.create_dataset(
            'test', dtype='float32', shape=self.shape, chunks=(10, 10, 10)
        )

        self.files = [self.ff_zarr, self.ff_n5]
        self.ds2_name = 'other'
        self.ds2_data = np.array([[inner * outer for inner in range(100)] for outer in range(100)], dtype='float32')
        self.default_chunk = z5py.base.Base.default_dataset_chunk_size
        self.default_dtype = z5py.base.Base.default_dataset_dtype

    def tearDown(self):
        if(os.path.exists('array.zr')):
            rmtree('array.zr')
        if(os.path.exists('array.n5')):
            rmtree('array.n5')

    def test_ds_open_empty_zarr(self):
        print("open empty zarr array")
        ds = self.ff_zarr['test']
        out = ds[:]
        self.assertEqual(out.shape, self.shape)
        self.assertTrue((out == 0).all())

    def test_ds_open_empty_n5(self):
        print("open empty n5 array")
        ds = self.ff_n5['test']
        out = ds[:]
        self.assertEqual(out.shape, self.shape)
        self.assertTrue((out == 0).all())

    def test_ds_zarr(self):
        dtypes = ('int8', 'int16', 'int32', 'int64',
                  'uint8', 'uint16', 'uint32', 'uint64',
                  'float32', 'float64')

        for dtype in dtypes:
            print("Running Zarr-Test for %s" % dtype)
            ds = self.ff_zarr.create_dataset(
                'data_%s' % dtype, dtype=dtype, shape=self.shape, chunks=(10, 10, 10)
            )
            in_array = 42 * np.ones(self.shape, dtype=dtype)
            ds[:] = in_array
            out_array = ds[:]
            self.assertEqual(out_array.shape, in_array.shape)
            self.assertTrue(np.allclose(out_array, in_array))

    def test_ds_n5(self):
        dtypes = ('int8', 'int16', 'int32', 'int64',
                  'uint8', 'uint16', 'uint32', 'uint64',
                  'float32', 'float64')

        for dtype in dtypes:
            print("Running N5-Test for %s" % dtype)
            ds = self.ff_n5.create_dataset(
                'data_%s' % dtype, dtype=dtype, shape=self.shape, chunks=(10, 10, 10)
            )
            in_array = 42 * np.ones(self.shape, dtype=dtype)
            ds[:] = in_array
            out_array = ds[:]
            self.assertEqual(out_array.shape, in_array.shape)
            self.assertTrue(np.allclose(out_array, in_array))

    def check_dtype_chunks_shape(self, f, data, exp_dtype, exp_chunks):
        ds2 = f.create_dataset(self.ds2_name, data=data)
        self.assertEqual(ds2.dtype, exp_dtype)
        self.assertEqual(ds2.chunks, exp_chunks)
        self.assertEqual(ds2.shape, np.asarray(data).shape)

    def test_create_with_list_of_lists_n5(self):
        data = self.ds2_data.tolist()
        self.check_dtype_chunks_shape(
            self.ff_n5, data, np.dtype('float64'),
            (self.default_chunk, self.default_chunk)
        )

    def test_create_with_list_of_lists_zarr(self):
        data = self.ds2_data.tolist()
        self.check_dtype_chunks_shape(
            self.ff_zarr, data, np.dtype('float64'),
            (self.default_chunk, self.default_chunk)
        )

    def test_create_with_ndarray_n5(self):
        self.check_dtype_chunks_shape(
            self.ff_n5, self.ds2_data, np.dtype('float32'),
            (self.default_chunk, self.default_chunk)
        )

    def test_create_with_ndarray_zarr(self):
        self.check_dtype_chunks_shape(
            self.ff_zarr, self.ds2_data, np.dtype('float32'),
            (self.default_chunk, self.default_chunk)
        )

    def test_create_with_chunked_n5(self):
        chunks = (10, 10)
        data = chunked_dataset(self.ds2_data, chunks)
        self.check_dtype_chunks_shape(
            self.ff_n5, data, np.dtype('float32'),
            chunks
        )

    def test_create_with_chunked_zarr(self):
        chunks = (10, 10)
        data = chunked_dataset(self.ds2_data, chunks)
        self.check_dtype_chunks_shape(
            self.ff_zarr, data, np.dtype('float32'),
            chunks
        )

    def check_default_values_and_warn(self, f):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertEqual(len(w), 0)
            ds2 = f.create_dataset(self.ds2_name, shape=(1000, 50))
            self.assertEqual(len(w), 1)
        self.assertEqual(ds2.chunks, (self.default_chunk, 50))
        self.assertEqual(ds2.dtype, self.default_dtype)

    def test_default_values_n5(self):
        self.check_default_values_and_warn(self.ff_n5)

    def test_default_values_zarr(self):
        self.check_default_values_and_warn(self.ff_zarr)

    @unittest.skipIf(sys.version_info.major < 3, "This fails in python 2")
    def test_ds_n5_array_to_format(self):
        dtypes = ('int8', 'int16', 'int32', 'int64',
                  'uint8', 'uint16', 'uint32', 'uint64',
                  'float32', 'float64')

        for dtype in dtypes:
            ds = self.ff_n5.create_dataset('data_%s' % dtype,
                                           dtype=dtype,
                                           shape=self.shape,
                                           chunks=(10, 10, 10))
            in_array = 42 * np.ones((10, 10, 10), dtype=dtype)
            ds[:10, :10, :10] = in_array

            path = os.path.join(os.path.dirname(ds.attrs.path), '0', '0', '0')
            with open(path, 'rb') as f:
                read_from_file = np.array([byte for byte in f.read()], dtype='int8')

            converted_data = ds.array_to_format(in_array)

            self.assertEqual(len(read_from_file), len(converted_data))
            self.assertTrue(np.allclose(read_from_file, converted_data))


if __name__ == '__main__':
    unittest.main()
