import unittest
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
class FileTestMixin(object):
    data_format = None
    other_format = None
    is_zarr = None
    DirectConstructor = None

    def setUp(self):
        self.file_path = "array." + self.data_format
        self.wrong_file_path = "array." + self.other_format
        self.assertFalse(os.path.exists(self.file_path))

    def tearDown(self):
        if os.path.exists(self.file_path):
            rmtree(self.file_path)

    def test_context_manager(self):
        with z5py.File(self.file_path, self.is_zarr) as f:
            self.assertIsInstance(f, z5py.File)
            self.assertEqual(self.is_zarr, f.is_zarr)

        self.assertTrue(os.path.isdir(self.file_path))

    def test_extension_detect(self):
        f = z5py.File(self.file_path, None)
        self.assertEqual(self.is_zarr, f.is_zarr)

    def test_direct_constructor(self):
        f = self.DirectConstructor(self.file_path)
        self.assertEqual(self.is_zarr, f.is_zarr)

    def test_wrong_ext_fails(self):
        with self.assertRaises(RuntimeError):
            f = z5py.File(self.wrong_file_path, use_zarr_format=False)


class TestN5File(FileTestMixin, unittest.TestCase):
    data_format = "n5"
    other_format = "zarr"
    is_zarr = False
    DirectConstructor = z5py.N5File


class TestZarrFile(FileTestMixin, unittest.TestCase):
    data_format = "zarr"
    other_format = "n5"
    is_zarr = True
    DirectConstructor = z5py.ZarrFile


class TestZrFile(TestZarrFile):
    data_format = "zr"
