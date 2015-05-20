import numpy as np
import scipy.interpolate as sci


class Blobel():
	"""
	Unfolding method after Blobel:

	- bins_meas: Array which sets the binning for all functions dependent on
				 the measured varible y (Measured MC). The first and last values
				 define the range of y in which the unfolding operates.
	- bins_unfold: Array which sets the binning of the target function f(x)
				   AFTER the unfolding. The binning is applied to the spline
				   representation of f(x) to get single points with error
				   estimation. The first and last values define the range
				   of x in which the target function is defined.
	- spline_knots: Array containing the inner knots for the (cubic) bspline
					basis functions which are used to represent f(x). The
					spline representation is used to discretize the true MC
					to build the response matrix Aij. The necessary outer
					knots are automatically added by scipy.interpolate.splev().
	"""
	def __init__(self, bins_meas, bins_unfold, spline_knots):
		self.bins_meas = bins_meas
		self.bins_unfold = bins_unfold
		self.spline_knots = spline_knots

		# use cubic splines to represent the unfolded target function f(x)
		self.spline_deg = 3

		# add outer knots by repeating first and last knot degree times
		self.pre = np.zeros(self.spline_deg)
		self.post = np.zeros(self.spline_deg)
		self.pre.fill(self.spline_knots[0])
		self.post.fill(self.spline_knots[-1])
		self.spline_knots = np.concatenate((self.pre, self.spline_knots, self.post))

		# number of splines (=coefficients aj) used to represent f(x)
		self.n_splines = len(self.spline_knots - self.spline_deg - 1)


	def build_response_matrix(self, mc_meas, mc_truth):
		"""
		Builds the response matrix Aij. i is the number of bins_meas and
		j is the number of spline_knots. To get Aij we use the mc_truth
		and mc_meas to track how f(x) is mapped to g(y).

		- mc_truth: True data before the MC detector simulation. f(x) is
				    the assumed distribution of the true data and is
				    described by mc_truth.
		- mc_meas: MC data after the detector simulation resulting from
				   mc_truth. g(y) is the measured distribution obtained
				   from MC simulation from f(x) and describes the measured
				   detector values.
		"""
		# response matrix Aij
		self.rows = len(self.bins_meas) - 1
		self.cols = self.n_splines
		self.A = np.zeros([self.rows, self.cols])

		## calculate spline representation of mc_truth f(x).
		# Every column Aj is a histogram mc_meas in y with binning bins_meas, if f(x) would be described by only one single basis function pj. To get the correct representation, set the histogram weights proportional to the basis function pj. This way, if we add up all hisotgrams Aj, we get the original distribution mc_meas. The correct weights (spline coefficients) aj, which really describe f(x) are found later with the maximum likelihood fit. Here we only create a histogram for every basis function.
		self.spline_coeff = np.zeros(self.cols)
		for j in range(self.n_splines):
			# enable only the jth basis spline
			self.spline_coeff.fill(0)
			self.spline_coeff[j] = 1
			# set the hist weight for Aj(y) proportional to pj(x)
			self.tck = (self.spline_knots, self.spline_coeff, self.spline_deg)
			self.basis_hist_weights = sci.splev(mc_meas, self.tck, ext=1)
			# bin mc_meas with weight according to f(x)=pj(x)
			self.hist, self.bins = np.histogram(
				mc_meas,
				bins=self.bins_meas,
				range=(self.bins_meas[0], self.bins_meas[1]),
				normed=False,
				weights = self.basis_hist_weights
				)
			self.A[:, j] = self.hist

		return self.A









