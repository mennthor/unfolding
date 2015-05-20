import scipy.stats as scs
import scipy.interpolate as sci
import numpy as np
import matplotlib.pyplot as plt

import testdata
import unfold


###################################
## Set the binning
# Get double normal distributed testdata
measured, true, pdfvals = testdata.double_gaus()
x = pdfvals[0]
pdf = pdfvals[1]

# Set binning of measured MC (after detector sim) in variable y. Doesn't have to be euqidistant.
bins_meas = np.linspace(0.0, 3.0, 20)
n_bins_meas = len(bins_meas - 1)
# Final binning of the unfolded function f(x) AFTER the unfolding and llh fit. Doesn't have to be equal to bins_meas. Not to be confused with the spline knots, which describe the intital discretization of the MC truth f0(x) to build the response matrix.
bins_unfold = np.linspace(0.0, 3.0, 11)
n_bins_unfold = len(bins_unfold - 1)


###################################
## Create the response matrix

# Position of the bspline knots to represent f0(x). Doesn't have to be equally spaced
inner_spline_knots = np.arange(0.0, 3.5, 1)

# Create the class instance
blobel_unfold = unfold.Blobel(bins_meas, bins_unfold, inner_spline_knots)

# The function create_response_matrix returns the response matrix Aij which maps the true variable x to the measured variable y
A = blobel_unfold.build_response_matrix(measured, true)

# np.set_printoptions(precision=3, suppress=True, linewidth=200)
# print(A)


###################################
## Fit the basis function coefficients to get the unfolded function f(x)
spline_true_coeff = blobel_unfold.fit_basis_coefficents(measured)
print spline_true_coeff


####################################
## Comparison plots
plt.figure()

# Plot the columns Aj of A containing the histograms of mc_meas when f0(x)=pj(x). The sum should be the distribution mc_meas.
n_splines = np.shape(A)[1]
for j in range(n_splines):
	# Use the left bin border and extend value to the right for plotting. So the last element of bins_meas is to much and has to be taken out. This gives the same plot as an plt.hist() with histtype=step.
	plt.step(bins_meas[:-1], A[:, j], where="post", alpha=0.5, lw=2)

# Plot the sum of the single hists in Aj
plt.step(bins_meas[:-1], np.sum(A, axis=1), "k", where="post", label="sum(Aj)", lw=2)

# Compare with mc_meas, mc_truth and true pdf. Renorm pdf to fit the hists.
norm_truth = np.dot(np.diff(bins_meas), np.histogram(true, bins_meas)[0])
plt.plot(x, norm_truth * pdf, "k-", lw=1, color="red", label="true pdf (renormed)")
plt.hist(true, bins=bins_meas, normed=False, label="mc_truth", histtype='stepfilled', color="grey", alpha=.5)
plt.hist(measured, bins=bins_meas, normed=False, label="mc_meas", histtype='stepfilled', color="blue", alpha=.2)

# Plot parameters
plt.xlim(-0.5, 3.5)
plt.xlabel("x")
plt.ylabel("num of entries")
plt.title("Decomposition of measured MC in basis functions")
plt.legend(loc="best")

# plt.show()

