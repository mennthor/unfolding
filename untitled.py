from __future__ import print_function, division
import scipy.stats as scs
import scipy.interpolate as sci
import numpy as np
import matplotlib.pyplot as plt

N = 10000
x = np.random.uniform(1, 5, N)
y = x - 0.05 * x**2 + np.random.normal(0, .2, N)

# under = np.array([0, 1])
# over = np.array([5, 6])
# bins = np.concatenate((under, np.linspace(1,5,20), over))
bins = np.linspace(0,6,20)


plt.hist(x, bins=bins, label="train", alpha=0.25)
plt.hist(y, bins=bins, label="meas no w", alpha=0.25)

weights = np.zeros_like(x)
weights[(y > 2) & (y < 3)] = 1
plt.hist(y, bins=bins, weights=weights, label="meas mid", alpha=0.25)

weights = np.zeros_like(x)
weights[(y < 2) | (y > 3)] = 1
plt.hist(y, bins=bins, weights=weights, label="meas outer", alpha=0.25)

plt.legend(loc="best")
plt.show()