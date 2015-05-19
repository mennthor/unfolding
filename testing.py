import scipy.stats as scs
import scipy.interpolate as sci
import numpy as np
import matplotlib.pyplot as plt

###################################
## plot the test data distribution
import testdata

# get double normal distributed testdata
measured, true = testdata.double_gaus()

# Compare true histogram to continous pdf
x = np.linspace(-0.5, 3.5, 500)
pdf = .5 * (scs.norm.pdf(x, loc=1., scale=.3) + scs.norm.pdf(x, loc=2., scale=.2))

## set binning
# binning of mc_meas in variable y
bins_meas = np.linspace(0.0, 3.0, 20)
n_bins_meas = len(bins_meas)
# binning of the mc_truth f(x) AFTER the unfolding and llh fit. has no effect in this section. doesn't have to be equal to bins_meas.
bins_unfold = np.linspace(0.0, 3.0, 20)

plt.figure()
plt.plot(x, pdf, "k-", label="true pdf")

plt.hist(true, bins=bins_unfold, normed=True, label="true", histtype='stepfilled', alpha=.25)
plt.hist(measured, bins=bins_meas, normed=True, label="meas", histtype='stepfilled', alpha=.25)

plt.legend(loc="best")


###################################
## create the response matrix and plot the binned spline representation of f(x)
import unfold

# position of the bspline knots to represent f(x). doesn't have to be equally spaced
inner_spline_knots = np.arange(0.0, 3.5, 0.5)

# create the class instance
blobel_unfold = unfold.Blobel(bins_meas, bins_unfold, inner_spline_knots)

# the function create_response_matrix return the matrix Aij which maps the true variable x to the measured variable y
A = blobel_unfold.build_response_matrix(measured, true)
print "# Shape of A: {}".format(np.shape(A))
print "# Num of bins meas: {}".format(len(bins_meas[:-1]))

# plot the columns Aj of A contain the histograms of mc_meas when f(x)=pj(x)
plt.figure()
n_splines = np.shape(A)[1]
for j in range(n_splines):
	plt.step(bins_meas[:-1], A[:, j], label="{}".format(j))

plt.legend(loc="best")

plt.show()

