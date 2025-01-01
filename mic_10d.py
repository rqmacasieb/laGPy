import numpy as np
from laGPy.laGPy import LaGP
import pandas as pd

# import training dataset
X = pd.read_csv('mic.dv_pop.csv', header = 0).drop(columns=['real_name']).values #n_tr x n_dv
y = pd.read_csv('mic.obs_pop.csv', header= 0).drop(columns=['real_name']) #n_tr x 1
y = y['func'].values

#read dv values
X_dv = pd.read_csv('dv.dat', header = 0).values.transpose() #1 x n_dv

# Create and fit LaGP model
model = LaGP(kernel="gaussian")
mean, var = model.fit_predict(
    X, y, X_dv,
    start=6,
    end=20,
    length_scale=0.1,
    nugget=1e-6,
    n_close=40
)

print("Prediction mean:", mean)
print("Prediction variance:", var)

# Save the prediction results to a file
with open('output.dat', 'w') as f:
    f.write(f"mean: {mean}\n")
    f.write(f"variance: {var}\n")
