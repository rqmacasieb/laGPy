import numpy as np
import pandas as pd
import laGPy

# import training dataset
X = pd.read_csv('mic.dv_pop.csv', header = 0).drop(columns=['real_name']).values #This should be an array of n_tr x n_dv
y = pd.read_csv('mic.obs_pop.csv', header= 0).drop(columns=['real_name'])
y = y['func'].values #This should be an array of n_tr x 1

#read dv values
X_dv = pd.read_csv('dv_untried.dat', header = 0).values.transpose() #This should be an array of 1 x n_dv

# Create and fit LaGP model
sims = laGPy.laGP(
    Xref=X_dv,             # Reference point - untried input point to be evaluated
    start=6,               # Initial points
    end=40,                # Total points to select
    X=X,                   # Input points
    Z=y,                   # Output value
    verb=0,               # Show optimization progress
    method = 'alc'          #alc (default) or nn
)

print(f"Final parameters: lengthscale={sims['d']:.6f}, nugget={sims['g']:.6f}")
print("Predictions:", sims['mean'])
print("Variances:", sims['s2'])
print("Selected indices:", sims['selected'])

# Save the prediction results to a file
with open('output.dat', 'w') as f:
    f.write('obsval\n')
    f.write(' '.join(map(str, sims['mean'])) + '\n')
    f.write(' '.join(map(str, sims['s2'])) + '\n')
    f.write('0\n')
