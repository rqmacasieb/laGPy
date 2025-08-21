#!/usr/bin/env python3
"""
Demonstration of the integrated derivative functionality in laGPy.

This script shows how to use the new compute_derivatives parameter
in the existing laGP function instead of calling separate functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from laGPy.laGPy import laGP, fullGP_with_derivatives

def test_function(x):
    """Test function: f(x) = sin(2πx) + 0.1*noise"""
    return np.sin(2 * np.pi * x) + 0.1 * np.random.randn()

def demo_integrated_approach():
    """Demonstrate the integrated derivative approach"""
    print("=== Integrated Derivatives Demo ===")
    print("Using laGP(compute_derivatives=True) instead of separate functions")
    
    # Generate training data
    np.random.seed(42)
    n_train = 50
    X_train = np.random.uniform(0, 1, n_train).reshape(-1, 1)
    Z_train = test_function(X_train.flatten())
    
    # Generate test points
    n_test = 100
    X_test = [[0.1]]
    
    print("\n1. Standard laGP (no derivatives):")
    results_standard = laGP(
        Xref=X_test,
        X=X_train,
        Z=Z_train,
        start=20,
        end=40,
        method="alc",
        lite=True,
        verb=0
    )
    print(f"   Results keys: {list(results_standard.keys())}")
    print(f"   Derivatives included: {'dmean' in results_standard}")
    
    print("\n2. laGP with derivatives enabled:")
    results_with_derivatives = laGP(
        Xref=X_test,
        X=X_train,
        Z=Z_train,
        start=20,
        end=40,
        method="alc",
        lite=True,
        verb=0,
        compute_derivatives=True  # New parameter!
    )
    print(f"   Results keys: {list(results_with_derivatives.keys())}")
    print(f"   Derivatives included: {'dmean' in results_with_derivatives}")
    print(f"   Derivative shape: {results_with_derivatives['dmean'].shape}")
    
    print("\n3. Full GP with derivatives (still separate function):")
    results_full = fullGP_with_derivatives(
        Xref=X_test,
        X=X_train,
        Z=Z_train,
        lite=True,
        verb=0
    )
    print(f"   Results keys: {list(results_full.keys())}")
    print(f"   Derivatives included: {'dmean' in results_full}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot function and predictions
    plt.subplot(1, 3, 1)
    plt.scatter(X_train, Z_train, alpha=0.6, label='Training data', s=20)
    plt.plot(X_test, results_standard['mean'], 'r-', linewidth=2, label='Standard laGP')
    plt.plot(X_test, results_with_derivatives['mean'], 'b--', linewidth=2, label='laGP + derivatives')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Predictions Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot derivatives
    plt.subplot(1, 3, 2)
    plt.plot(X_test, results_with_derivatives['dmean'][:, 0], 'b-', linewidth=2, label='∂μ/∂x')
    plt.xlabel('x')
    plt.ylabel('df/dx')
    plt.title('Analytical Derivatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot variance
    plt.subplot(1, 3, 3)
    plt.plot(X_test, results_standard['s2'], 'r-', linewidth=2, label='Standard laGP')
    plt.plot(X_test, results_with_derivatives['s2'], 'b--', linewidth=2, label='laGP + derivatives')
    plt.xlabel('x')
    plt.ylabel('σ²')
    plt.title('Prediction Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Demo completed successfully!")
    print("\nKey benefits of the integrated approach:")
    print("1. Single function call with optional parameter")
    print("2. Backward compatibility maintained")
    print("3. Cleaner API design")
    print("4. Same performance when derivatives not needed")

if __name__ == "__main__":
    try:
        demo_integrated_approach()
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()
