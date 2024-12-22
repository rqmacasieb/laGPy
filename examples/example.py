import numpy as np
import laGPy

# Generate example data
np.random.seed(42)
X = np.random.rand(100, 2)
y = np.sin(X[:, 0]) + np.cos(X[:, 1])
X_test = np.random.rand(10, 2)

# Create and fit LaGP model
model = laGPy(kernel="gaussian")
mean, var = model.fit_predict(
    X, y, X_test,
    start=6,
    end=20,
    length_scale=0.1,
    nugget=1e-6,
    n_close=40
)

print("Predictions mean:", mean)
print("Predictions variance:", var)