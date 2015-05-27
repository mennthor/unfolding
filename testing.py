from __future__ import print_function, division
import scipy.stats as scs
import scipy.interpolate as sci
import numpy as np
import matplotlib.pyplot as plt

import datagen
import unfold


###################################
## Set the binning
# Get testdata
measured, true, pdfvals, bins = datagen.example()
# measured, true, pdfvals = datagen.uniform()
# measured, true, pdfvals, bins = datagen.gaus()
# measured, true, pdfvals, bins = datagen.double_gaus()
x = pdfvals[0]
pdf = pdfvals[1]

# Set binning of measured MC (after detector sim) in variable y. Doesn't have to be equidistant, but try to avoid empty bins, if possible.
bins_meas = bins
n_bins_meas = len(bins_meas)-1
# Final binning of the unfolded function f(x) AFTER the unfolding and llh fit. Doesn't have to be equal to bins_meas. Not to be confused with the spline knots, which describe the intital discretization of the MC truth f0(x) to build the response matrix.
bins_unfold = np.linspace(0.0, 3.0, 10)
n_bins_unfold = len(bins_unfold) - 1
# Position of the inner bspline knots to represent f0(x). Doesn't have to be equally spaced or the same as the binning.
inner_spline_knots = np.linspace(bins_meas[0], bins_meas[-1], 8)


###################################
## Create the response matrix
# Create the class instance
blobel_unfold = unfold.Blobel(bins_meas, bins_unfold, inner_spline_knots)
# The function create_response_matrix returns the response matrix Aij which maps the true variable x to the measured variable y
A = blobel_unfold.build_response_matrix(measured, true)


###################################
## Fit the basis function coefficients to get the unfolded function f(x)
tck = blobel_unfold.fit_basis_coefficents(measured)


####################################
## Comparison plots and printing
## Print unfolding parameters
# print(blobel_unfold)
## Show the response matrix
if False:
	np.set_printoptions(precision=3, suppress=True, linewidth=200)
	print("## Response matrix A:\n{}".format(A))
	matfig, matax = plt.subplots(1, 1, facecolor="#E0E0E0")
	cmap = plt.get_cmap("gist_heat")
	matcax = matax.matshow(A, cmap=cmap)#, vmin=0, vmax=1)
	matax.set_xlabel("jth column represents the jth spline function")
	matax.set_ylabel("ith row is ith bin in y")
	matfig.suptitle("Entries of response matrix A", fontsize=16)
	matfig.colorbar(matcax)
	print("A.rows = {}".format(len(A[:,0])))
	print("A.cols = {}".format(len(A[0,:])))

## Plot the columns Aj of A containing the histograms of mc_meas when f0(x)=pj(x). The sum should be the distribution mc_meas.
n_splines = np.shape(A)[1]
if False:
	plt.figure()
	for j in range(n_splines):
		# where="post" with the last entry doubled gives the same plot as plt.hist() with histtype=step.
		plt.step(bins_meas[:], np.append(A[:, j], A[-1, j]), where="post", alpha=0.5, lw=2)
	# Plot the sum of the single hists in Aj for every bin in y
	plt.step(bins_meas[:-1], np.sum(A, axis=1), "k", where="post", label="sum(Aj)", lw=2)
	# Plot parameters
	plt.xlim(-1, 5)
	plt.xlabel("x")
	plt.ylabel("num of entries")
	plt.title("Decomposition of measured MC in basis functions")
	plt.legend(loc="best")

## Compare with mc_meas, mc_truth and true pdf. Renorm pdf to fit the hists.
if True:
	plt.figure()
	# MC truth and measured
	plt.hist(true, bins=bins_meas, normed=False, label="mc_truth", histtype='stepfilled', color="grey", alpha=.5)
	plt.hist(measured, bins=bins_meas, normed=False, label="mc_meas", histtype='stepfilled', color="blue", alpha=.2)

	# Plot the spline composition
	norm = np.sum(np.histogram(true, bins_meas)[0])/n_bins_meas
	y = np.zeros([n_splines, len(x)])
	for j in range(n_splines):
		coeff = np.zeros(n_splines)
		coeff[j] = tck[1][j]
		y[j] = sci.splev(x, (tck[0], coeff ,tck[2]), ext=1)
		plt.plot(x, norm * y[j])
	# Plot the unfolded function f(x) from the fitted coefficents aj by summing all splines
	plt.plot(x, norm * y.sum(axis=0), "k", lw=2, label="sum unfolded")
	# Plot the spline knots
	plt.plot(tck[0], np.zeros(len(tck[0])), "ko")
	# Print the optimized coefficients
	print(tck[1])

	# Plot parameters
	plt.xlim(-1, 5)
	plt.xlabel("x")
	plt.ylabel("Probability")
	plt.title("True and measured MC and renormed unfolded spline sum f(x)")
	plt.legend(loc="best")

plt.show()

