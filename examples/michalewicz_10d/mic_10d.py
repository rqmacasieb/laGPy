import numpy as np
import pandas as pd
import sys

#import laGPy from the current release version
from laGPy import laGP, Method

#import laGPy from the local development directory
# sys.path.append('../..')
# from laGPy.laGPy import LaGP

# import training dataset
X = pd.read_csv('mic.dv_pop.csv', header = 0).drop(columns=['real_name']).values #n_tr x n_dv
y = pd.read_csv('mic.obs_pop.csv', header= 0).drop(columns=['real_name']) #n_tr x 1
y = y['func'].values

#read dv values
X_dv = pd.read_csv('dv.dat', header = 0).values.transpose() #1 x n_dv

# Create and fit LaGP model
mean, var, indices, d, g = laGP(
    m=2,                     # 2D input
    start=6,               # Initial points
    end=20,                # Total points to select
    Xref=X_dv,             # Reference points
    n=X.shape[0],          # Total available points
    X=X,                   # Input points
    Z=y,                   # Output values
    d=None,                # Let the algorithm estimate lengthscale
    g=1e-04,                # Let the algorithm estimate nugget
    method=Method.ALC,     # Use ALC selection
    close=30,              # Consider 30 closest points
    param_est=True,        # Enable parameter estimation
    d_range=(1e-4, 1e+4),  # Bounds for lengthscale
    est_freq=6,          # Re-estimate every 10 points
    verb=0                # Show optimization progress
)

print("Prediction mean:", mean)
print("Prediction variance:", var)

# Save the prediction results to a file
with open('output.dat', 'w') as f:
    f.write(f"mean: {mean}\n")
    f.write(f"variance: {var}\n")
