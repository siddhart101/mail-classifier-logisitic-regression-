import numpy as np
# Load the trained parameters



# Gaussian likelihood function
def gaussian_likelihood(x, mu, sigma2):
    return (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))

# Load the test data
test_x=np.load("./data_1/test.npz")["x"]
test_y=np.load("./data_1/test.npz")["y"]


# Use the trained parameters from 3.1
mpos, sigpos = -0.0721922106722285, 1.3031231465734459
mneg, signeg = 0.9401561132214228, 1.9426265036964034

# Predict class
predictions = []
for x in test_x:
    p_pos = gaussian_likelihood(x, mpos, sigpos)
    p_neg = gaussian_likelihood(x, mneg, signeg)
    predictions.append(1 if p_pos >= p_neg else -1)

# Compute accuracy
predictions = np.array(predictions)
accuracy = np.mean(predictions == test_y)
print("Test Accuracy:", accuracy)
