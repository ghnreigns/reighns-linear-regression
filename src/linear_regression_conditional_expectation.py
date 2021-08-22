import torch
from sklearn.exceptions import NotFittedError
import numpy as np
import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
"""
Generates Linear Regression Population Data that follows Assumptions
"""
torch.manual_seed(1992)

device = 'cuda'


def generate_dataset_simple(bias, beta, num_pop, num_features, error_mean, error_std_dev):

    # Generate x as an array of `n` samples which can take a value between 0 and 100

    x = torch.randint(low=0,  high=100, size=(
        num_pop,)).to(device)

    # Generate the random error of n samples, with a random value from a normal distribution, with mean 0 and std 1.
    # Condition: Homoscedasticity within error terms
    # Condition: Normality of the residuals/true error terms

    e = torch.normal(mean=error_mean, std=error_std_dev,
                     size=(num_pop,)).to(device)  # * std_dev

    # assert torch.mean(e) == 0, torch.std(e) == 1

    # Condition: Linearity between x and y with an small error term e
    y = x * beta + bias + e
    return x, y, e


beta = 9.33  # only 1 coefficient for simple LR with no bias
bias = 11.1
error_mean = 0
error_std_dev = 1

num_pop = 100000000
num_features = 1

x, y, e = generate_dataset_simple(bias=bias, beta=beta, num_pop=num_pop,
                                  num_features=num_features, error_mean=error_mean, error_std_dev=error_std_dev)

x_is_ten = torch.where(x == 10)

y_is_ten = y[x_is_ten]
print(y_is_ten.shape)
y_is_ten_cond_mean = torch.mean(y_is_ten)
print(y_is_ten_cond_mean)


# Sample
num_sample = 1000000
random_sampling_index = torch.randint(low=0,  high=num_pop, size=(
    num_sample,)).to(device)

x_sample = x[random_sampling_index].cpu().numpy()
x_sample = x_sample.reshape(-1, 1)
y_sample = y[random_sampling_index].cpu().numpy()

# plt.scatter(y.cpu().numpy(), e.cpu().numpy())
# plt.plot(y.cpu().numpy(), [0]*len(y.cpu().numpy()))
# plt.show()

lr_SKLEARN = LinearRegression(fit_intercept=True, normalize=False).fit(
    x_sample, y_sample
)

print(lr_SKLEARN.coef_)
print(lr_SKLEARN.intercept_)

pred_SKLEARN = lr_SKLEARN.predict([[10]])
print(pred_SKLEARN)


x_is_ten = np.where(x_sample.flatten() == 10)

y_is_ten = y_sample[x_is_ten]
print(y_is_ten.shape)
y_is_ten_cond_mean = np.mean(y_is_ten)
print(y_is_ten_cond_mean)
