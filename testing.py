import scipy.stats as scs
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

plt.figure()
plt.plot(x, pdf, "k-", label="true pdf")

plt.hist(
	(true, measured),
	bins=20,
	normed=True,
	label=["true", "measured"],
	histtype='stepfilled',
	alpha=.25
	)

plt.legend(loc="best")

###################################
## create the response matrix and plot the binned spline representation of f(x)
import unfold

# binning of mc_meas in variable y
bins_meas = np.linspace(0.5, 3.0, 15)
n_bins_meas = len(bins_meas)
# binning of the mc_truth f(x) AFTER the unfolding and llh fit. has no effect in this section
bins_unfold = np.linspace(0.0, 3.0, 20)
# position of the bspline knots to represent f(x). doesn't have to be equally spaced
inner_spline_knots = np.arange(0.0, 3.5, 0.5)
spline_deg = 3
n_splines = len(inner_spline_knots - spline_deg - 1)

# create the class instance
blobel_unfold = unfold.Blobel(bins_meas, bins_unfold, inner_spline_knots)

# the function create_response_matrix return the matrix Aij which maps the true variable x to the measured variable y
A = np.zeros([n_bins_meas, n_splines])
A = blobel_unfold.build_response_matrix(measured, true)

# plot the columns Aj of A contain the histograms of mc_meas when f(x)=pj(x)
# plt.figure()
# for j in range(n_splines):
# 	plt.plot(bins_meas[:-1], A[:, j])

plt.show()

