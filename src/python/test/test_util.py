import sys
import unittest
import numpy as np
import os
from shutil import rmtree

try:
    from concurrent import futures
except ImportError:
    futures = False

try:
    import z5py
except ImportError:
    sys.path.append('..')
    import z5py


class TestUtil(unittest.TestCase):
    tmp_dir = './tmp_dir'
    shape = (100, 100, 100)
    chunks = (10, 10, 10)

    def setUp(self):
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            rmtree(self.tmp_dir)

    @unittest.skipUnless(futures, "Needs 3rd party concurrent.futures in python 2")
    def test_rechunk_default(self):
        from z5py.util import rechunk
        in_path = os.path.join(self.tmp_dir, 'in.n5')
        out_path = os.path.join(self.tmp_dir, 'out.n5')

        # create input file
        in_file = z5py.File(in_path, use_zarr_format=False)
        ds_in = in_file.create_dataset('data', dtype='float32',
                                       shape=self.shape, chunks=self.chunks,
                                       compression='gzip')
        # write test data
        data = np.arange(ds_in.size).reshape(ds_in.shape).astype(ds_in.dtype)
        ds_in[:] = data

        # rechunk for different out blocks
        out_file = z5py.File(out_path, use_zarr_format=False)
        new_chunks = (20, 20, 20)

        # NOTE we can only choose out blocks that align with the chunks
        # because otherwise we run into issues due to not thread safe blocking
        for out_blocks in (None, (40, 40, 40), (60, 60, 60)):
            ds_str = 'none' if out_blocks is None else '_'.join(map(str, out_blocks))
            ds_name = 'data_%s' % ds_str
            rechunk(in_path, out_path, 'data', ds_name, new_chunks,
                    out_blocks=out_blocks,
                    n_threads=8)
            # make sure that new data agrees
            ds_out = out_file[ds_name]
            data_out = ds_out[:]
            self.assertEqual(data_out.shape, data.shape)
            self.assertEqual(ds_out.chunks, new_chunks)
            self.assertTrue(np.allclose(data, data_out))

    @unittest.skipUnless(futures, "Needs 3rd party concurrent.futures in python 2")
    def test_rechunk_custom(self):
        from z5py.util import rechunk
        in_path = os.path.join(self.tmp_dir, 'in.n5')
        out_path = os.path.join(self.tmp_dir, 'out.n5')

        # create input file
        in_file = z5py.File(in_path, use_zarr_format=False)
        ds_in = in_file.create_dataset('data', dtype='float32',
                                       shape=self.shape, chunks=self.chunks,
                                       compression='gzip')
        # write test data
        data = np.arange(ds_in.size).reshape(ds_in.shape).astype(ds_in.dtype)
        ds_in[:] = data
        # rechunk
        new_chunks = (20, 20, 20)
        for compression in ('raw', 'gzip'):
            for dtype in ('float64', 'int32', 'uint32'):
                ds_name = 'ds_%s_%s' % (compression, dtype)
                rechunk(in_path, out_path,
                        'data', ds_name, new_chunks,
                        n_threads=8,
                        compression=compression,
                        dtype=dtype)
                # make sure that new data agrees
                out_file = z5py.File(out_path, use_zarr_format=False)
                ds_out = out_file[ds_name]
                data_out = ds_out[:]
                self.assertEqual(data_out.shape, data.shape)
                self.assertEqual(ds_out.chunks, new_chunks)
                self.assertTrue(np.allclose(data, data_out))

    # TODO finish blocking tests
    def simple_blocking(self, shape, block_shape):
        blocking = []
        ndim = len(shape)
        for x in range(0, shape[0], block_shape[0]):
            blocking.append((x, min(x + block_shape[0], shape[0])))
            if ndim > 1:
                block = blocking.pop()
                for y in range(0, shape[1], block_shape[1]):
                    blocking.append(block + min(y + block_shape[1], shape[1]))

    def _test_blocking(self):
        from z5py.util import blocking
        n_reps = 10

        for dim in range(1, 6):
            for _ in range(n_reps):
                shape = tuple(np.random.randint(0, 1000) for ii in range(dim))
                block_shape = tuple(min(np.random.randint(0, 100), sh)
                                    for ii, sh in zip(range(dim), shape))
                blocking1 = [(block.start, block.stop)
                             for block in blocking(shape, block_shape)]
                blocking2 = self.simple_blocking(shape, block_shape)
                sorted(blocking1)
                sorted(blocking2)
                self.assertEqual(blocking1, blocking2)


if __name__ == '__main__':
    unittest.main()
