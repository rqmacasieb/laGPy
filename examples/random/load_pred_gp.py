import numpy as np
import laGPy

gp = laGPy.loadGP()

Xref = np.random.rand(1, 10)
sims = gp.predict_lite(Xref)

print("Successfully loaded model")