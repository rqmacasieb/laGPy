import numpy as np
import matplotlib.pyplot as plt
from laGPy.laGPy import laGP

def test_function(x):
    """Test function: f(x) = sin(2πx) + 0.1*noise"""
    return np.sin(2 * np.pi * x) + 0.1

def test_function_derivative(x):
    """Analytical derivative: f'(x) = 2π * cos(2πx)"""
    return 2 * np.pi * np.cos(2 * np.pi * x)

def finite_difference_derivative(f, x, h=1e-10):
    """Finite difference derivative approximation"""
    return (f(x + h) - f(x - h)) / (2 * h)

def lagp_fd(X, X_train, Z_train, h=1e-10):
    X_plus = X + h
    X_minus = X - h
    
    lagp_plus = laGP(Xref=X_plus,
        X=X_train,
        Z=Z_train,
        start=20,
        end=40,
        method="alc",
        lite=True,
        verb=0)
    
    lagp_minus = laGP(Xref=X_minus,
        X=X_train,
        Z=Z_train,
        start=20,
        end=40,
        method="alc",
        lite=True,
        verb=0
    )
    return (lagp_plus['mean'] - lagp_minus['mean']) / (2 * h)



np.random.seed(42)
n_train = 50
X_train = np.random.uniform(0, 1, n_train).reshape(-1, 1)
Z_train = test_function(X_train.flatten())

X_test = [[0.1]]
results_with_derivatives = laGP(Xref=X_test,
        X=X_train,
        Z=Z_train,
        start=20,
        end=30,
        method="alc",
        lite=True,
        verb=0,
        compute_derivatives=True)
true_value = test_function(X_test[0][0])
analytical_gp_derivative = results_with_derivatives['dmean']
fd_derivative = lagp_fd(X_test[0][0], X_train, Z_train)
true_derivative = test_function_derivative(X_test[0][0])

print(f"True value: {true_value}")
print(f"GP predicted value: {results_with_derivatives['mean']}")
print(f"Analytical GP derivative: {analytical_gp_derivative}")
print(f"Finite difference derivative: {fd_derivative}")
print(f"True derivative: {true_derivative}")

debug = 0
