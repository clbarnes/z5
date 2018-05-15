import unittest
import numpy as np
import os
from shutil import rmtree
from six import add_metaclass
from abc import ABCMeta

import sys
try:
    import z5py
except ImportError:
    sys.path.append('..')
    import z5py


@add_metaclass(ABCMeta)
class GroupTestMixin(object):
    data_format = None
    
    def setUp(self):
        self.shape = (100, 100, 100)
        self.file_path = 'array.' + self.data_format
        self.file = z5py.File(self.file_path)
        g = self.file.create_group('test')
        g.create_dataset(
            'test', dtype='float32', shape=self.shape, chunks=(10, 10, 10)
        )

    def tearDown(self):
        if os.path.exists(self.file_path):
            rmtree(self.file_path)

    def test_open_empty_group(self):
        g = self.file['test']
        ds = g['test']
        out = ds[:]
        self.assertEqual(out.shape, self.shape)
        self.assertTrue((out == 0).all())

    def test_open_empty_dataset(self):
        ds = self.file['test/test']
        out = ds[:]
        self.assertEqual(out.shape, self.shape)
        self.assertTrue((out == 0).all())

    def test_group(self):
        g = self.file.create_group('group')
        ds = g.create_dataset(
            'data', dtype='float32', shape=self.shape, chunks=(10, 10, 10)
        )
        in_array = 42 * np.ones(self.shape, dtype='float32')
        ds[:] = in_array
        out_array = ds[:]
        self.assertEqual(out_array.shape, in_array.shape)
        self.assertTrue(np.allclose(out_array, in_array))


class TestN5Group(GroupTestMixin, unittest.TestCase):
    data_format = 'n5'


class TestZarrGroup(GroupTestMixin, unittest.TestCase):
    data_format = 'zarr'


if __name__ == '__main__':
    unittest.main()
