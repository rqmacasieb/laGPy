import numpy as np
import laGPy


# Generate example data
X = np.random.rand(100, 10)
Z = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(100)
Xref = np.random.rand(1, 10)

# Create and fit LaGP model
gp = laGPy.buildGP(X, Z, verb=1)

print("Model successfully built and saved to GPRmodel.gp")