# laGPy Derivatives Extension

This document describes the new analytical derivative functionality added to the laGPy package.

## Overview

The laGPy package has been extended to support **analytical computation of derivatives** for Gaussian Process predictions. This allows you to compute:

- **∂μ/∂x**: Derivatives of mean predictions with respect to input dimensions
- **∂σ²/∂x**: Derivatives of variance predictions with respect to input dimensions

## New Functions

### 1. `laGP(compute_derivatives=True)`

The main `laGP()` function now supports an optional `compute_derivatives` parameter to return derivatives along with predictions.

```python
from laGPy import laGP

results = laGP(
    Xref=X_test,           # Reference points for prediction
    X=X_train,             # Training inputs
    Z=Z_train,             # Training outputs
    start=20,              # Initial design size
    end=50,                # Final design size
    method="alc",          # Point selection method
    lite=True,             # Use lightweight version
    verb=1,                # Verbosity level
    compute_derivatives=True  # Enable derivative computation
)

# Access results
mean_predictions = results['mean']           # Shape: (n_test,)
variance_predictions = results['s2']         # Shape: (n_test,)
derivatives_mean = results['dmean']          # Shape: (n_test, n_dimensions)
derivatives_variance = results['ds2']        # Shape: (n_test, n_dimensions)
```

### 2. `fullGP_with_derivatives()`

Full GP prediction with derivatives (no local approximation).

```python
from laGPy import fullGP_with_derivatives

results = fullGP_with_derivatives(
    Xref=X_test,           # Reference points for prediction
    X=X_train,             # Training inputs
    Z=Z_train,             # Training outputs
    lite=True,             # Use lightweight version
    verb=1                 # Verbosity level
)
```

### 3. GP Class Methods

New methods added to the `GP` class:

```python
# Full prediction with derivatives
results = gp.predict_with_derivatives(Xref)

# Lightweight prediction with derivatives
results = gp.predict_lite_with_derivatives(Xref)

# Both return dictionaries containing:
# - mean: Mean predictions
# - s2/Sigma: Variance predictions
# - df: Degrees of freedom
# - llik: Log likelihood
# - dmean: Derivatives of mean w.r.t. inputs
# - ds2: Derivatives of variance w.r.t. inputs
```

## Mathematical Background

### Mean Derivatives

For a GP prediction at point x*, the derivative of the mean with respect to input dimension j is:

```
∂μ(x*)/∂x_j = ∂k(x*, X)/∂x_j @ K⁻¹ @ Z
```

where:
- `k(x*, X)` is the covariance vector between x* and training points X
- `K⁻¹` is the inverse covariance matrix of training points
- `Z` is the training outputs

### Variance Derivatives

For the variance prediction, the derivative is:

```
∂σ²(x*)/∂x_j = 2 * phidf * (∂k(x*, X)/∂x_j @ K⁻¹ @ k(x*, X)ᵀ - k(x*, X) @ K⁻¹ @ ∂k(x*, X)ᵀ/∂x_j)
```

where `phidf` is the normalized residual sum of squares.

### Covariance Derivatives

The derivative of the covariance function (Gaussian kernel) with respect to input dimension j is:

```
∂k(x*, x)/∂x_j = -k(x*, x) * (x*_j - x_j) / (d * ||x* - x||)
```

where `d` is the lengthscale parameter.

## Usage Examples

### Basic Usage

```python
import numpy as np
from laGPy import laGP_with_derivatives

# Generate sample data
X_train = np.random.uniform(0, 1, (100, 2))
Z_train = np.sin(2*np.pi*X_train[:, 0]) * np.cos(2*np.pi*X_train[:, 1])

# Test points
X_test = np.array([[0.5, 0.5], [0.7, 0.3]])

# Get predictions with derivatives
results = laGP_with_derivatives(
    Xref=X_test,
    X=X_train,
    Z=Z_train,
    start=20,
    end=50,
    method="alc"
)

# Access derivatives
print("Mean predictions:", results['mean'])
print("Derivatives w.r.t. x:", results['dmean'][:, 0])
print("Derivatives w.r.t. y:", results['dmean'][:, 1])
```

### Gradient-Based Optimization

```python
def objective_function(x):
    """Objective function that uses GP derivatives"""
    x_2d = x.reshape(1, -1)
    results = laGP(
        Xref=x_2d,
        X=X_train,
        Z=Z_train,
        start=20,
        end=50,
        compute_derivatives=True
    )
    return results['mean'][0], results['dmean'][0]

# Use in optimization
from scipy.optimize import minimize

result = minimize(
    objective_function,
    x0=[0.5, 0.5],
    jac=True,
    method='L-BFGS-B'
)
```

## Performance Considerations

### Computational Complexity

- **Standard prediction**: O(n²) for matrix operations
- **Derivative computation**: O(n² × m) where m is the number of input dimensions
- **Memory usage**: Additional storage for derivative matrices

### Optimization Tips

1. **Use `lite=True`** when you only need diagonal covariance (faster)
2. **Limit input dimensions** when derivatives aren't needed for all dimensions
3. **Batch predictions** when computing derivatives at multiple points

## Comparison with Finite Differences

The analytical derivatives are:
- **More accurate** than finite differences
- **Faster** for multiple dimensions
- **Numerically stable** (no step size tuning needed)

```python
# Finite difference approach (for comparison)
def finite_diff_derivative(func, x, h=1e-6):
    x_plus = x + h
    x_minus = x - h
    return (func(x_plus) - func(x_minus)) / (2 * h)

# Analytical approach (more accurate and efficient)
results = laGP(Xref, X, Z, start=20, end=50, compute_derivatives=True)
analytical_derivatives = results['dmean']
```

## Error Handling

The derivative computation handles edge cases:
- **Zero distances**: Derivatives set to 0 when training and test points coincide
- **Numerical stability**: Uses `np.errstate` to handle division by zero gracefully
- **Input validation**: Ensures proper array shapes and dimensions

## Limitations

1. **Kernel function**: Currently only supports Gaussian (RBF) kernel
2. **Isotropic lengthscale**: Uses single lengthscale parameter for all dimensions
3. **Memory usage**: Derivatives require additional storage proportional to input dimensions

## Future Enhancements

Potential improvements:
1. **Multiple kernel support**: Extend to other covariance functions
2. **Anisotropic lengthscales**: Support different lengthscales per dimension
3. **Higher-order derivatives**: Second and third derivatives
4. **GPU acceleration**: Leverage existing GPU infrastructure for derivative computation

## Testing

Run the example script to verify functionality:

```bash
python example_derivatives.py
```

This will demonstrate:
- 1D and 2D derivative computation
- Comparison with finite differences
- Visualization of predictions and derivatives
- Error statistics and validation

## Dependencies

The derivative functionality requires:
- NumPy (for array operations)
- SciPy (for optimization and linear algebra)
- Matplotlib (for examples and visualization)

## Support

For issues or questions about the derivative functionality:
1. Check the example script for usage patterns
2. Verify input data formats and dimensions
3. Ensure the GP model is properly trained before computing derivatives
