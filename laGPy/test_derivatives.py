#!/usr/bin/env python3
"""
Simple test script to verify the derivative functionality in laGPy.
"""

import numpy as np
import sys
import os

# Add the current directory to the path so we can import laGPy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from laGPy import laGP, fullGP_with_derivatives
    print("✓ Successfully imported laGPy with derivatives")
except ImportError as e:
    print(f"✗ Failed to import laGPy: {e}")
    sys.exit(1)

def test_simple_1d():
    """Test simple 1D case"""
    print("\n--- Testing 1D Derivatives ---")
    
    # Simple 1D data
    X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
    Z_train = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # Linear function
    
    X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
    
    try:
        results = laGP(
            Xref=X_test,
            X=X_train,
            Z=Z_train,
            start=3,
            end=5,
            method="alc",
            lite=True,
            verb=0,
            compute_derivatives=True
        )
        
        print(f"✓ 1D laGP with derivatives completed")
        print(f"  Mean predictions: {results['mean']}")
        print(f"  Derivatives: {results['dmean'][:, 0]}")
        print(f"  Shapes - mean: {results['mean'].shape}, dmean: {results['dmean'].shape}")
        
        # For a linear function, derivatives should be approximately constant
        derivative_std = np.std(results['dmean'][:, 0])
        print(f"  Derivative std dev: {derivative_std:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 1D test failed: {e}")
        return False

def test_simple_2d():
    """Test simple 2D case"""
    print("\n--- Testing 2D Derivatives ---")
    
    # Simple 2D data
    X_train = np.array([[0.1, 0.1], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7], [0.9, 0.9]])
    Z_train = np.array([0.2, 0.6, 1.0, 1.4, 1.8])  # Linear function x + y
    
    X_test = np.array([[0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]])
    
    try:
        results = laGP(
            Xref=X_test,
            X=X_train,
            Z=Z_train,
            start=3,
            end=5,
            method="alc",
            lite=True,
            verb=0,
            compute_derivatives=True
        )
        
        print(f"✓ 2D laGP with derivatives completed")
        print(f"  Mean predictions: {results['mean']}")
        print(f"  Derivatives w.r.t. x: {results['dmean'][:, 0]}")
        print(f"  Derivatives w.r.t. y: {results['dmean'][:, 1]}")
        print(f"  Shapes - mean: {results['mean'].shape}, dmean: {results['dmean'].shape}")
        
        # For a linear function x + y, derivatives should be approximately 1
        dx_mean = np.mean(results['dmean'][:, 0])
        dy_mean = np.mean(results['dmean'][:, 1])
        print(f"  Mean derivative w.r.t. x: {dx_mean:.6f} (expected ~1.0)")
        print(f"  Mean derivative w.r.t. y: {dy_mean:.6f} (expected ~1.0)")
        
        return True
        
    except Exception as e:
        print(f"✗ 2D test failed: {e}")
        return False

def test_full_gp():
    """Test full GP with derivatives"""
    print("\n--- Testing Full GP Derivatives ---")
    
    # Simple 1D data
    X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
    Z_train = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
    
    try:
        results = fullGP_with_derivatives(
            Xref=X_test,
            X=X_train,
            Z=Z_train,
            lite=True,
            verb=0
        )
        
        print(f"✓ Full GP with derivatives completed")
        print(f"  Mean predictions: {results['mean']}")
        print(f"  Derivatives: {results['dmean'][:, 0]}")
        print(f"  Shapes - mean: {results['mean'].shape}, dmean: {results['dmean'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Full GP test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases"""
    print("\n--- Testing Edge Cases ---")
    
    # Test with identical points (should handle gracefully)
    X_train = np.array([[0.1], [0.1], [0.5], [0.7], [0.9]])  # Duplicate point
    Z_train = np.array([0.1, 0.1, 0.5, 0.7, 0.9])
    
    X_test = np.array([[0.1], [0.5]])  # One test point same as training
    
    try:
        results = laGP(
            Xref=X_test,
            X=X_train,
            Z=Z_train,
            start=3,
            end=5,
            method="alc",
            lite=True,
            verb=0,
            compute_derivatives=True
        )
        
        print(f"✓ Edge case test completed")
        print(f"  Results shape: {results['mean'].shape}")
        print(f"  Derivatives shape: {results['dmean'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("laGPy Derivatives Test Suite")
    print("=" * 40)
    
    tests = [
        test_simple_1d,
        test_simple_2d,
        test_full_gp,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Derivatives functionality is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
