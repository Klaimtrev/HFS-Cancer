import scipy.io
import sys
import os

# Add the path to the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CFSmethod import CFS  # Now you can import the module


mat = scipy.io.loadmat('Datasets\colon.mat')
X = mat['X']
X = X.astype(float)
y = mat['Y']
y = y[:, 0]
n_samples, n_features = X.shape  # number of samples and number of features

print("number of samples: ", n_samples, "number of features: ", n_features)
print("y = ",y)

idx = CFS.cfs(X, y)
print(idx)
