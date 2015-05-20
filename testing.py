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
# binning of measured MC (after detector sim) in variable y
bins_meas = np.linspace(0.0, 3.0, 20)
n_bins_meas = len(bins_meas)
# final binning of the MC truth f(x) AFTER the unfolding and llh fit. has no effect in this section. doesn't have to be equal to bins_meas. Not to be confused with the spline knots, which describe the intital discretization to build the response matrix.
bins_unfold = np.linspace(0.0, 3.0, 20)

# plt.figure()
# plt.plot(x, pdf, "k-", label="true pdf")

# plt.hist(true, bins=bins_unfold, normed=True, label="true", histtype='stepfilled', alpha=.25)
# plt.hist(measured, bins=bins_meas, normed=True, label="meas", histtype='stepfilled', alpha=.25)

# plt.legend(loc="best")


###################################
## create the response matrix and plot the binned spline representation of f(x)
import unfold

# position of the bspline knots to represent f(x). doesn't have to be equally spaced
inner_spline_knots = np.arange(0.0, 3.5, 0.5)

# create the class instance
blobel_unfold = unfold.Blobel(bins_meas, bins_unfold, inner_spline_knots)

# the function create_response_matrix return the matrix Aij which maps the true variable x to the measured variable y
A = blobel_unfold.build_response_matrix(measured, true)

## Plot the (here normed to integral under curve = 1) columns Aj of A containing the histograms of mc_meas when f(x)=pj(x). The sum should be the distribution mc_meas.
n_splines = np.shape(A)[1]

plt.figure()
for j in range(n_splines):
	# Use the left bin border and extend value to the right for plotting. So the last element of bins_meas is to much and has to be taken out. This gives the same plot as an plt.hist() with histtype=step.
	plt.step(bins_meas[:-1], A[:, j], where="post", alpha=0.5, lw=2)

# Plot the sum of the single hists in Aj.
plt.step(bins_meas[:-1], np.sum(A, axis=1), "k", where="post", label="sum(Aj)", lw=2)

# Compare with mc_meas, mc_truth and true pdf.
norm_truth = np.dot(np.diff(bins_unfold), np.histogram(true, bins_unfold)[0])
print norm_truth
print np.diff(bins_unfold)
print np.histogram(true, bins_unfold)[0]

plt.plot(x, norm_truth * pdf, "k-", label="true pdf")
plt.hist(true, bins=bins_unfold, normed=False, label="mc_truth", histtype='stepfilled', alpha=.2)
plt.hist(measured, bins=bins_meas, normed=False, label="mc_meas", histtype='stepfilled', alpha=.2)

plt.xlabel("x")
plt.ylabel("num of entries")
plt.title("Decomposition of measured MC in basis functions")
plt.legend(loc="best")

plt.show()

