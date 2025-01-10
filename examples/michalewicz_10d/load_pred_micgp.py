import numpy as np
import pandas as pd
import laGPy
import time

# import training dataset
X = pd.read_csv('mic.dv_pop.csv', header = 0).drop(columns=['real_name']).values #n_tr x n_dv
y = pd.read_csv('mic.obs_pop.csv', header= 0).drop(columns=['real_name']) #n_tr x 1
y = y['func'].values

gp = laGPy.loadGP()
print("Model successfully loaded from GPRmodel.gp")

#read dv values
X_dv = pd.read_csv('dv_untried.dat', header = 0).values.transpose() #1 x n_dv

start_time = time.time()
sims = gp.predict(X_dv)
end_time = time.time()
print("Predictions:", sims['mean'])
print("Variances:", sims['Sigma'])
print(f"Prediction time: {end_time - start_time:.6f} seconds")

