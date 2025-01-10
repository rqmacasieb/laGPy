import numpy as np
import laGPy

# Generate example data
X = np.random.rand(100, 10)
Z = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(100)
Xref = np.random.rand(1, 10)

# Create and fit LaGP model
sims = laGPy.laGP(
    Xref=Xref,             # Reference points
    X=X,                   # Input points
    Z=Z,                   # Output value
    start=10,               # Initial points
    end=20,                # Total points to select
    d = 1,
    g = 0.01,
    verb=1                # Show optimization progress
)

print(f"Final parameters: lengthscale={sims['d']:.6f}, nugget={sims['g']:.6f}")
print("Predictions:", sims['mean'])
print("Variances:", sims['s2'])
print("Selected indices:", sims['selected'])