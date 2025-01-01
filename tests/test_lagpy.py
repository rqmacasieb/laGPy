import unittest
from laGPy import laGP, Method
import numpy as np

class TestLaGPy(unittest.TestCase):

    def setUp(self):
        # Set up any necessary data or state before each test
        self.Xref = np.array([[3.0, 4.0]])
        self.X = np.random.rand(30, 2)
        self.Z = np.sin(self.X[:, 0]) + np.cos(self.X[:, 1]) + 0.1 * np.random.randn(30)
        self.start = 10
        self.end = 30
        self.d = 1.0
        self.g = 0.01

    def test_laGP_basic(self):
        # Test the basic functionality of the laGP function
        result = laGP(self.Xref, self.start, self.end, self.X, self.Z, self.d, self.g, method=Method.ALC)
        self.assertIn('mean', result)
        self.assertIn('s2', result)
        self.assertEqual(len(result['selected']), self.end)

    def test_laGP_invalid_start(self):
        # Test that laGP raises an error with invalid start
        with self.assertRaises(ValueError):
            laGP(self.Xref, 1, self.end, self.X, self.Z, self.d, self.g)

    # TODO: Add more tests

if __name__ == '__main__':
    unittest.main()