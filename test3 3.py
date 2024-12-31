import numpy as np

# Gaussian likelihood function
def gaussian_likelihood(x, mu, sigma2):
    return (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))

# Load the training data to compute priors
train_data = np.load("./data_1/train.npz")
train_x = train_data["x"]
train_y = train_data["y"]

# Compute priors based on the training data
prior_pos = np.mean(train_y == 1)  # Fraction of positive samples
prior_neg = np.mean(train_y == -1)  # Fraction of negative samples

# Load the test data
test_data = np.load("./data_1/test.npz")
test_x = test_data["x"]
test_y = test_data["y"]

# Use the trained parameters from 3.1
mpos, sigpos = -0.0721922106722285, 1.3031231465734459
mneg, signeg = 0.9401561132214228, 1.9426265036964034

# Predict class without priors
preds = []
for x in test_x:
    p_pos = gaussian_likelihood(x, mpos, sigpos)
    p_neg = gaussian_likelihood(x, mneg, signeg)
    preds.append(1 if p_pos >= p_neg else -1)

# Compute accuracy without priors
preds = np.array(preds)
accuracy = np.mean(preds == test_y)
print("Test Accuracy", accuracy)

# Predict class with priors (MAP)
predictions_with_priors = []
for x in test_x:
    p_pos = prior_pos * gaussian_likelihood(x, mpos, sigpos)
    p_neg = prior_neg * gaussian_likelihood(x, mneg, signeg)
    predictions_with_priors.append(1 if p_pos >= p_neg else -1)

# Compute accuracy with priors
predictions_with_priors = np.array(predictions_with_priors)
accuracy_with_priors = np.mean(predictions_with_priors == test_y)
print("Improved Test Accuracy with Priors:", accuracy_with_priors)
