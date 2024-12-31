import numpy as np
import pandas as pd
import laGPy

# import training dataset
X = pd.read_csv('mic.dv_pop.csv', header = 0).drop(columns=['real_name']).values #n_tr x n_dv
y = pd.read_csv('mic.obs_pop.csv', header= 0).drop(columns=['real_name']) #n_tr x 1
y = y['func'].values

gp = laGPy.buildGP(X, y, verb=1)

print("Model successfully built and saved to GPRmodel.gp")