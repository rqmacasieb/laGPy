import unittest
import pandas as pd
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

    def test_michalewicz(self):
        X = pd.read_csv('./tests/data/mic.dv_pop.csv').drop(columns=['real_name'])
        Y = pd.read_csv('./tests/data/mic.obs_pop.csv').drop(columns=['real_name'])['func']

        Xref = pd.read_csv('./tests/data/mic.0.dv_pop.csv')
        Yref = pd.read_csv('./tests/data/mic.0.obs_pop.csv')

        for mem in Xref['real_name']:
            sim = laGP(Xref = Xref[Xref['real_name'] == mem].drop(columns=['real_name']).values,
                       X = X.values,
                       Z = Y.values,
                       start = 10, 
                       end = 60)

            val_mean = Yref.loc[Yref['real_name'] == mem]['func'].item()
            assert abs(sim['mean'].item() - val_mean)**2 < 1e-10

            val_s2 = Yref.loc[Yref['real_name'] == mem]['func_s2'].item()
            assert abs(sim['s2'].item() - val_s2)**2 < 1e-10
    

    def test_gradient_calcs(self):
        def test_function_2d(x):
            x1, x2 = x[0], x[1]
            return np.sin(2 * np.pi * x1) * np.cos(np.pi * x2)

        def true_gradients(x):
            x1, x2 = x[0], x[1]
            df_dx1 = 2 * np.pi * np.cos(2 * np.pi * x1) * np.cos(np.pi * x2)
            df_dx2 = -np.pi * np.sin(2 * np.pi * x1) * np.sin(np.pi * x2)
            return np.array([df_dx1, df_dx2])

        def finite_difference_gradients(X, X_train, Z_train, h=1e-8):
            X_plus_x1 = X.copy()
            X_minus_x1 = X.copy()
            X_plus_x2 = X.copy()
            X_minus_x2 = X.copy()
            
            X_plus_x1[0, 0] += h
            X_minus_x1[0, 0] -= h
            X_plus_x2[0, 1] += h
            X_minus_x2[0, 1] -= h
            
            # Get GP predictions at perturbed points
            gp_plus_x1 = laGP(Xref=X_plus_x1, X=X_train, Z=Z_train, start=20, end=40, method="alc")
            gp_minus_x1 = laGP(Xref=X_minus_x1, X=X_train, Z=Z_train, start=20, end=40, method="alc")
            gp_plus_x2 = laGP(Xref=X_plus_x2, X=X_train, Z=Z_train, start=20, end=40, method="alc")
            gp_minus_x2 = laGP(Xref=X_minus_x2, X=X_train, Z=Z_train, start=20, end=40, method="alc")
            
            # Compute finite differences
            df_dx1 = (gp_plus_x1['mean'][0] - gp_minus_x1['mean'][0]) / (2 * h)
            df_dx2 = (gp_plus_x2['mean'][0] - gp_minus_x2['mean'][0]) / (2 * h)
            
            return np.array([df_dx1, df_dx2])

        np.random.seed(42)
        n_train = 500
        X_train = np.random.uniform(0, 1, (n_train, 2))
        Z_train = np.array([test_function_2d(x) for x in X_train]) 

        # Test point
        X_test = np.array([[0.3, 0.7]])
        
        results = laGP(Xref=X_test, X=X_train, Z=Z_train, 
                       start=20, end=40, method="alc", 
                       compute_gradients=True)

        assert 'dmean' in results

        gp_grads = results['dmean'][0]
        true_grads = true_gradients(X_test[0])
        fd_grads = finite_difference_gradients(X_test, X_train, Z_train)

        assert abs(gp_grads[0] - true_grads[0]) < 1e-2
        assert abs(gp_grads[1] - true_grads[1]) < 1e-3
        assert abs(gp_grads[0] - fd_grads[0]) < 1e-3
        assert abs(gp_grads[1] - fd_grads[1]) < 5e-4


if __name__ == '__main__':
    unittest.main()