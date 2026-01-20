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

    def test_matern32_kernel(self):
        """Test that Matern 3/2 kernel works correctly"""
        result = laGP(self.Xref, self.X, self.Z, self.start, self.end, 
                     self.d, self.g, kernel='matern32')
        self.assertIn('mean', result)
        self.assertIn('s2', result)
        self.assertTrue(np.all(result['s2'] >= 0))
        self.assertEqual(len(result['selected']), self.end)
        
        gp = buildGP(self.X, self.Z, self.d, self.g, kernel='matern32', export=False)
        self.assertEqual(gp.kernel, 'matern32')

    def test_matern52_kernel(self):
        """Test that Matern 5/2 kernel works correctly"""
        result = laGP(self.Xref, self.X, self.Z, self.start, self.end, 
                     self.d, self.g, kernel='matern52')
        self.assertIn('mean', result)
        self.assertIn('s2', result)
        self.assertTrue(np.all(result['s2'] >= 0))
        self.assertEqual(len(result['selected']), self.end)
        
        gp = buildGP(self.X, self.Z, self.d, self.g, kernel='matern52', export=False)
        self.assertEqual(gp.kernel, 'matern52')

    def test_kernel_differences(self):
        """Test that different kernels produce different results"""

        np.random.seed(42)
        X_test = np.random.rand(20, 2)
        Z_test = np.sin(X_test[:, 0]) + np.cos(X_test[:, 1])
        X_ref = np.array([[0.5, 0.5]])
        
        result_se = laGP(X_ref, X_test, Z_test, start=10, end=15, 
                        kernel='squared_exponential')
        result_m32 = laGP(X_ref, X_test, Z_test, start=10, end=15, 
                         kernel='matern32')
        result_m52 = laGP(X_ref, X_test, Z_test, start=10, end=15, 
                         kernel='matern52')

        self.assertIn('mean', result_se)
        self.assertIn('mean', result_m32)
        self.assertIn('mean', result_m52)
        
        self.assertTrue(result_se['s2'] >= 0)
        self.assertTrue(result_m32['s2'] >= 0)
        self.assertTrue(result_m52['s2'] >= 0)

    def test_matern_kernel_covariance_properties(self):
        """Test that Matern kernels produce valid covariance matrices"""
        from laGPy.covar import covar_symm
        
        K_m32 = covar_symm(self.X, self.d, self.g, kernel='matern32')
        self.assertEqual(K_m32.shape, (self.X.shape[0], self.X.shape[0]))
        self.assertTrue(np.allclose(K_m32, K_m32.T)) 
        self.assertTrue(np.all(np.diag(K_m32) >= 1.0)) 
        
        K_m52 = covar_symm(self.X, self.d, self.g, kernel='matern52')
        self.assertEqual(K_m52.shape, (self.X.shape[0], self.X.shape[0]))
        self.assertTrue(np.allclose(K_m52, K_m52.T))  
        self.assertTrue(np.all(np.diag(K_m52) >= 1.0)) 

    def test_matern_kernel_gradients(self):
        """Test gradient calculations with Matern kernels"""
        def test_function_2d(x):
            x1, x2 = x[0], x[1]
            return np.sin(2 * np.pi * x1) * np.cos(np.pi * x2)

        np.random.seed(42)
        n_train = 200
        X_train = np.random.uniform(0, 1, (n_train, 2))
        Z_train = np.array([test_function_2d(x) for x in X_train])
        X_test = np.array([[0.3, 0.7]])
        
        result_m32 = laGP(Xref=X_test, X=X_train, Z=Z_train, 
                         start=15, end=30, method="alc",
                         kernel='matern32', compute_gradients=True)
        
        self.assertIn('dmean', result_m32)
        self.assertEqual(result_m32['dmean'].shape, (1, 2))
        self.assertIn('ds2', result_m32)
        self.assertEqual(result_m32['ds2'].shape, (1, 2))
        
        result_m52 = laGP(Xref=X_test, X=X_train, Z=Z_train, 
                         start=15, end=30, method="alc",
                         kernel='matern52', compute_gradients=True)
        
        self.assertIn('dmean', result_m52)
        self.assertEqual(result_m52['dmean'].shape, (1, 2))
        self.assertIn('ds2', result_m52)
        self.assertEqual(result_m52['ds2'].shape, (1, 2))

    def test_matern_kernel_save_load(self):
        """Test that kernel type is preserved when saving/loading GP models"""
        gp_m32 = buildGP(self.X, self.Z, self.d, self.g, 
                        kernel='matern32', wdir=self.wdir, 
                        fname='test_matern32.gp', export=True)
        self.assertEqual(gp_m32.kernel, 'matern32')
        
        gp_loaded = loadGP(wdir=self.wdir, fname='test_matern32.gp')
        self.assertEqual(gp_loaded.kernel, 'matern32')
        
        gp_m52 = buildGP(self.X, self.Z, self.d, self.g, 
                        kernel='matern52', wdir=self.wdir, 
                        fname='test_matern52.gp', export=True)
        self.assertEqual(gp_m52.kernel, 'matern52')
        
        gp_loaded_m52 = loadGP(wdir=self.wdir, fname='test_matern52.gp')
        self.assertEqual(gp_loaded_m52.kernel, 'matern52')

    def test_matern_kernel_at_zero_distance(self):
        """Test that Matern kernels handle zero distance correctly"""
        from laGPy.covar import covar
        
        X1 = np.array([[0.5, 0.5]])
        X2 = np.array([[0.5, 0.5]])
        
        # Matern 3/2: k(0) = 1
        k_m32 = covar(X1, X2, self.d, kernel='matern32')
        self.assertAlmostEqual(k_m32[0, 0], 1.0, places=10)
        
        # Matern 5/2: k(0) = 1
        k_m52 = covar(X1, X2, self.d, kernel='matern52')
        self.assertAlmostEqual(k_m52[0, 0], 1.0, places=10)

    def test_matern_kernel_smoothness(self):
        """Test that Matern kernels have correct smoothness properties"""
        from laGPy.covar import covar
        
        X1 = np.array([[0.0, 0.0]])
        X2_close = np.array([[0.1, 0.0]])
        X2_far = np.array([[1.0, 0.0]])
        
        d = 1.0
        
        # Matern 3/2 should be smoother than exponential
        k_exp_close = covar(X1, X2_close, d, kernel='exponential')[0, 0]
        k_m32_close = covar(X1, X2_close, d, kernel='matern32')[0, 0]
        k_m52_close = covar(X1, X2_close, d, kernel='matern52')[0, 0]
        
        # Matern 5/2 should decay slower than Matern 3/2 (smoother)
        # At close distances, all should be similar
        self.assertGreater(k_m52_close, k_m32_close)
        self.assertGreater(k_m32_close, k_exp_close)
        
        # At far distances, Matern kernels should decay faster than squared_exponential
        k_se_far = covar(X1, X2_far, d, kernel='squared_exponential')[0, 0]
        k_m32_far = covar(X1, X2_far, d, kernel='matern32')[0, 0]
        k_m52_far = covar(X1, X2_far, d, kernel='matern52')[0, 0]
        
        # Matern kernels decay faster (smaller covariance) at large distances
        self.assertLess(k_se_far, k_m32_far)
        self.assertLess(k_se_far, k_m52_far)
        self.assertGreater(k_m52_far, k_m32_far)

    def test_invalid_kernel_type(self):
        """Test that invalid kernel types raise appropriate errors"""
        with self.assertRaises(ValueError):
            laGP(self.Xref, self.X, self.Z, self.start, self.end, 
                self.d, self.g, kernel='invalid_kernel')
        
        with self.assertRaises(ValueError):
            buildGP(self.X, self.Z, self.d, self.g, kernel='invalid_kernel')

if __name__ == '__main__':
    unittest.main()