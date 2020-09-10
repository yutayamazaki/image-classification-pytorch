import os
import unittest

from PIL import Image

import hashing


class ReadImageAsMD5Tests(unittest.TestCase):

    def setUp(self):
        self.save_path = 'fig.png'
        img: Image.Image = Image.new('RGB', (10, 10), color=0)
        img.save(self.save_path)

    def tearDown(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

    def test_simple(self):
        hashed: str = hashing.read_image_as_md5(self.save_path)

        self.assertIsInstance(hashed, str)
        self.assertEqual(len(hashed), 32)
        self.assertEqual(
            hashed,
            hashing.read_image_as_md5(self.save_path)
        )


class AverageHashTests(unittest.TestCase):

    def setUp(self):
        self.img: Image.Image = Image.new('RGB', (10, 10), color=0)

    def test_simple(self):
        hashed: str = hashing.average_hash(self.img)

        self.assertIsInstance(hashed, str)
        self.assertEqual(len(hashed), 64)
        self.assertEqual(hashed, hashing.average_hash(self.img))


class HammingDistanceTests(unittest.TestCase):

    def test_simple(self):
        a: str = '0101'
        b: str = '0110'
        distance: int = hashing.hamming_distance(a, b)
        self.assertEqual(distance, 2)
