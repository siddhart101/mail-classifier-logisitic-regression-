import numpy as np

# Load the data
train_x=np.load("./data_1/train.npz")["x"]
train_y=np.load("./data_1/train.npz")["y"]

# Separate the data by class
X_pos = train_x[train_y == 1]
X_neg = train_x[train_y == -1]

# Compute mean and variance for each class
mpos = np.mean(X_pos, axis=0)
sigpos = np.var(X_pos, axis=0)

mneg = np.mean(X_neg, axis=0)
signeg = np.var(X_neg, axis=0)

# Print results
print("Class +: Mean =", mpos, ", Variance =", sigpos)
print("Class -: Mean =", mneg, ", Variance =", signeg)
