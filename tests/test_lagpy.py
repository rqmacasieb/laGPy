import unittest
from laGPy import laGP, Method, buildGP, loadGP, fullGP, newGP, updateGP
import numpy as np

class TestLaGPy(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(30, 2)
        self.Z = np.sin(self.X[:, 0]) + np.cos(self.X[:, 1]) + 0.1 * np.random.randn(30)
        self.Xref = np.array([[3.0, 4.0]])
        self.d = 1.0
        self.g = 0.01
        self.start = 10
        self.end = 20
        self.wdir = '.'
        self.fname = 'test_model.gp'

    def test_laGP_basic(self):
        result = laGP(self.Xref, self.X, self.Z, self.start, self.end, self.d, self.g, method=Method.ALC)
        self.assertIn('mean', result)
        self.assertIn('s2', result)
        self.assertEqual(len(result['selected']), self.end)

    def test_laGP_noprior(self):
        result = laGP(self.Xref, self.X, self.Z, self.start, self.end, method=Method.ALC)
        self.assertIn('mean', result)
        self.assertIn('s2', result)
        self.assertEqual(len(result['selected']), self.end)

    def test_laGP_invalid_start(self):
        with self.assertRaises(ValueError):
            laGP(self.Xref, self.X, self.Z, 1, self.end, self.d, self.g)

    def test_laGP_single_point(self):
        X_single = np.array([[0.5, 0.5]])
        Z_single = np.array([0.5])
        with self.assertRaises(ValueError):
            laGP(self.Xref, X_single, Z_single, 1, 1, self.d, self.g)

    def test_laGP_output_values(self):
        result = laGP(self.Xref, self.X, self.Z, self.start, self.end, self.d, self.g)
        self.assertTrue(np.all(result['mean'] >= -1) and np.all(result['mean'] <= 1))
        self.assertTrue(np.all(result['s2'] >= 0))

    def test_buildGP(self):
        gp = buildGP(self.X, self.Z, self.d, self.g, wdir=self.wdir, fname=self.fname, export=False)
        self.assertIsNotNone(gp)
        self.assertEqual(gp.X.shape, self.X.shape)
        self.assertEqual(gp.Z.shape, self.Z.shape)

    def test_loadGP(self):
        buildGP(self.X, self.Z, self.d, self.g, wdir=self.wdir, fname=self.fname, export=True)
        gp = loadGP(wdir=self.wdir, fname=self.fname)
        self.assertIsNotNone(gp)
        self.assertEqual(gp.X.shape, self.X.shape)
        self.assertEqual(gp.Z.shape, self.Z.shape)

    def test_fullGP(self):
        result = fullGP(self.Xref, self.X, self.Z, self.d, self.g, lite=True)
        self.assertIn('mean', result)
        self.assertIn('s2', result)
        self.assertIn('df', result)
        self.assertIn('llik', result)
        self.assertIn('d_posterior', result)
        self.assertIn('g_posterior', result)

    def test_newGP(self):
        gp = newGP(self.X, self.Z, self.d, self.g)
        self.assertIsNotNone(gp)
        self.assertEqual(gp.X.shape, self.X.shape)
        self.assertEqual(gp.Z.shape, self.Z.shape)
        self.assertEqual(gp.d, self.d)
        self.assertEqual(gp.g, self.g)

    def test_updateGP(self):
        gp = newGP(self.X, self.Z, self.d, self.g)
        
        new_X = np.random.rand(5, 2)
        new_Z = np.sin(new_X[:, 0]) + np.cos(new_X[:, 1]) + 0.1 * np.random.randn(5)
        
        gp.update(new_X, new_Z)

        self.assertEqual(gp.X.shape[0], self.X.shape[0] + new_X.shape[0])
        self.assertEqual(gp.Z.shape[0], self.Z.shape[0] + new_Z.shape[0])

if __name__ == '__main__':
    unittest.main()