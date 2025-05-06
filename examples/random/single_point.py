import numpy as np
import laGPy

# Generate example data
X = np.random.rand(100, 10) #This should be an array of n_tr x n_dv
Z = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(100) #This should be an array of n_tr x 1
Xref = np.random.rand(1, 10) #This should be an array of 1 x n_dv

# Create and fit LaGP model
sims = laGPy.laGP(
    Xref=Xref,             # Reference point - untried input point to be evaluated
    X=X,                   # Input points
    Z=Z,                   # Output value
    start=10,               # Initial points
    end=60,                # Total points to select
    verb=1                # Show optimization progress
)

print(f"Final parameters: lengthscale={sims['d']:.6f}, nugget={sims['g']:.6f}")
print("Predictions:", sims['mean'])
print("Variances:", sims['s2'])
print("Selected indices:", sims['selected'])