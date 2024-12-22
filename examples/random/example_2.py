import numpy as np
from laGPy import laGP, Method

# Generate example data
X = np.random.rand(100, 2)
Z = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(100)
Xref = np.random.rand(5, 2)

# Run with automatic parameter estimation
mean, var, indices, d, g = laGP(
    m=2,                     # 2D input
    start=6,               # Initial points
    end=20,                # Total points to select
    Xref=Xref,             # Reference points
    n=X.shape[0],          # Total available points
    X=X,                   # Input points
    Z=Z,                   # Output values
    d=None,                # Let the algorithm estimate lengthscale
    g=None,                # Let the algorithm estimate nugget
    method=Method.ALC,     # Use ALC selection
    close=30,              # Consider 30 closest points
    param_est=True,        # Enable parameter estimation
    d_range=(1e-6, 1.0),  # Bounds for lengthscale
    g_range=(1e-6, 0.1),  # Bounds for nugget
    est_freq=10,          # Re-estimate every 10 points
    verb=1                # Show optimization progress
)

print(f"Final parameters: lengthscale={d:.6f}, nugget={g:.6f}")
print("Predictions:", mean)
print("Variances:", var)
print("Selected indices:", indices)