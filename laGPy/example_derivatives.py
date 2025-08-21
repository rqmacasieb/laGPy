#!/usr/bin/env python3
"""
Example script demonstrating the new derivative functionality in laGPy.

This script shows how to:
1. Use the new laGP_with_derivatives function
2. Access analytical derivatives of predictions
3. Compare with finite difference approximations
"""

import numpy as np
import matplotlib.pyplot as plt
from laGPy import laGP, fullGP_with_derivatives

def test_function(x):
    """Test function: f(x) = sin(2πx) + 0.1*noise"""
    return np.sin(2 * np.pi * x) + 0.1 * np.random.randn()

def test_function_2d(x):
    """2D test function: f(x,y) = sin(2πx) * cos(2πy) + 0.1*noise"""
    return np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1]) + 0.1 * np.random.randn(x.shape[0])

def finite_difference_derivative(func, x, h=1e-6):
    """Compute finite difference derivative for comparison"""
    x_plus = x + h
    x_minus = x - h
    return (func(x_plus) - func(x_minus)) / (2 * h)

def example_1d_derivatives():
    """Example with 1D input data"""
    print("=== 1D Example ===")
    
    # Generate training data
    np.random.seed(42)
    n_train = 100
    X_train = np.random.uniform(0, 1, n_train).reshape(-1, 1)
    Z_train = test_function(X_train.flatten())
    
    # Generate test points
    n_test = 50
    X_test = [[0.22]]
    
    # Use laGP with derivatives
    print("Running laGP with derivatives...")
    results = laGP(
        Xref=X_test,
        X=X_train,
        Z=Z_train,
        start=20,  # Start with 20 points
        end=50,    # Use up to 50 points
        method="alc",
        lite=True,
        verb=1,
        compute_derivatives=True
    )
    
    print(f"Prediction shape: {results['mean'].shape}")
    print(f"Derivative shape: {results['dmean'].shape}")
    print(f"Mean predictions: {results['mean'][:5]}")
    print(f"Derivatives: {results['dmean'][:5, 0]}")
    
    # Compare with finite differences
    print("\nComparing with finite differences...")
    fd_derivatives = np.zeros(n_test)
    for i in range(n_test):
        fd_derivatives[i] = finite_difference_derivative(
            lambda x: results['mean'][i], X_test[i, 0]
        )
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot function and predictions
    plt.subplot(2, 1, 1)
    plt.scatter(X_train, Z_train, alpha=0.6, label='Training data', s=20)
    plt.plot(X_test, results['mean'], 'r-', linewidth=2, label='laGP prediction')
    plt.fill_between(X_test.flatten(), 
                     results['mean'] - 2*np.sqrt(results['s2']),
                     results['mean'] + 2*np.sqrt(results['s2']),
                     alpha=0.3, color='red', label='±2σ confidence')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('1D laGP Prediction with Derivatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot derivatives
    plt.subplot(2, 1, 2)
    plt.plot(X_test, results['dmean'][:, 0], 'b-', linewidth=2, label='Analytical derivative')
    plt.plot(X_test, fd_derivatives, 'g--', linewidth=2, label='Finite difference')
    plt.xlabel('x')
    plt.ylabel('df/dx')
    plt.title('Derivatives Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print error statistics
    derivative_error = np.abs(results['dmean'][:, 0] - fd_derivatives)
    print(f"Mean absolute error in derivatives: {np.mean(derivative_error):.6f}")
    print(f"Max absolute error in derivatives: {np.max(derivative_error):.6f}")

def example_2d_derivatives():
    """Example with 2D input data"""
    print("\n=== 2D Example ===")
    
    # Generate training data
    np.random.seed(42)
    n_train = 200
    X_train = np.random.uniform(0, 1, (n_train, 2))
    Z_train = test_function_2d(X_train)
    
    # Generate test points on a grid
    n_test = 20
    x_test = np.linspace(0, 1, n_test)
    y_test = np.linspace(0, 1, n_test)
    X_test = np.array(np.meshgrid(x_test, y_test)).T.reshape(-1, 2)
    
    # Use laGP with derivatives
    print("Running 2D laGP with derivatives...")
    results = laGP(
        Xref=X_test,
        X=X_train,
        Z=Z_train,
        start=30,  # Start with 30 points
        end=80,    # Use up to 80 points
        method="alc",
        lite=True,
        verb=1,
        compute_derivatives=True
    )
    
    print(f"2D prediction shape: {results['mean'].shape}")
    print(f"2D derivative shape: {results['dmean'].shape}")
    print(f"First few mean predictions: {results['mean'][:5]}")
    print(f"First few derivatives w.r.t. x: {results['dmean'][:5, 0]}")
    print(f"First few derivatives w.r.t. y: {results['dmean'][:5, 1]}")
    
    # Reshape for plotting
    mean_2d = results['mean'].reshape(n_test, n_test)
    dmean_dx = results['dmean'][:, 0].reshape(n_test, n_test)
    dmean_dy = results['dmean'][:, 1].reshape(n_test, n_test)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot function
    im1 = axes[0, 0].contourf(x_test, y_test, mean_2d, levels=20)
    axes[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=Z_train, s=20, alpha=0.7)
    axes[0, 0].set_title('2D laGP Prediction')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot derivative w.r.t. x
    im2 = axes[0, 1].contourf(x_test, y_test, dmean_dx, levels=20)
    axes[0, 1].set_title('∂f/∂x')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot derivative w.r.t. y
    im3 = axes[1, 0].contourf(x_test, y_test, dmean_dy, levels=20)
    axes[1, 0].set_title('∂f/∂y')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot variance
    variance_2d = results['s2'].reshape(n_test, n_test)
    im4 = axes[1, 1].contourf(x_test, y_test, variance_2d, levels=20)
    axes[1, 1].set_title('Prediction Variance')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()

def example_full_gp_derivatives():
    """Example using full GP with derivatives"""
    print("\n=== Full GP Example ===")
    
    # Generate training data
    np.random.seed(42)
    n_train = 50
    X_train = np.random.uniform(0, 1, n_train).reshape(-1, 1)
    Z_train = test_function(X_train.flatten())
    
    # Generate test points
    n_test = 100
    X_test = np.linspace(0, 1, n_test).reshape(-1, 1)
    
    # Use full GP with derivatives
    print("Running full GP with derivatives...")
    results = fullGP_with_derivatives(
        Xref=X_test,
        X=X_train,
        Z=Z_train,
        lite=True,
        verb=1
    )
    
    print(f"Full GP prediction shape: {results['mean'].shape}")
    print(f"Full GP derivative shape: {results['dmean'].shape}")
    print(f"Mean predictions: {results['mean'][:5]}")
    print(f"Derivatives: {results['dmean'][:5, 0]}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot function and predictions
    plt.subplot(2, 1, 1)
    plt.scatter(X_train, Z_train, alpha=0.6, label='Training data', s=20)
    plt.plot(X_test, results['mean'], 'r-', linewidth=2, label='Full GP prediction')
    plt.fill_between(X_test.flatten(), 
                     results['mean'] - 2*np.sqrt(results['s2']),
                     results['mean'] + 2*np.sqrt(results['s2']),
                     alpha=0.3, color='red', label='±2σ confidence')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Full GP Prediction with Derivatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot derivatives
    plt.subplot(2, 1, 2)
    plt.plot(X_test, results['dmean'][:, 0], 'b-', linewidth=2, label='Analytical derivative')
    plt.xlabel('x')
    plt.ylabel('df/dx')
    plt.title('Derivatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("laGPy Derivatives Example")
    print("=" * 50)
    
    try:
        # Run examples
        example_1d_derivatives()
        example_2d_derivatives()
        example_full_gp_derivatives()
        
        print("\nAll examples completed successfully!")
        print("\nKey features demonstrated:")
        print("1. laGP(compute_derivatives=True) - Local approximate GP with derivatives")
        print("2. fullGP_with_derivatives() - Full GP with derivatives")
        print("3. Analytical derivatives vs finite differences")
        print("4. Both 1D and 2D input spaces")
        print("5. Access to dmean (∂μ/∂x) and ds2 (∂σ²/∂x)")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
