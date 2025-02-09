import numpy as np
import laGPy
import time

# Generate example data
X = np.random.rand(1000, 20)
Z = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(1000)
Xref = np.random.rand(1, 20)
start_time = time.time()



# Create and fit LaGP model
gp = laGPy.buildGP(X, Z, d=9.9157, g=0.0001,verb=1)

elapsed_time = time.time() - start_time
print(f"Elapsed time to execute buildGP: {elapsed_time:.4f} seconds")
print("Model successfully built and saved to GPRmodel.gp")