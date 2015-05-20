import numpy as np
import scipy.interpolate as sci
import scipy.optimize as sco


class Blobel():
	"""
	Unfolding method after Blobel:
	Used names in this class:

	- f0(x): True distribution before detection MC simulation. Used to build
			 the response matrix A by mapping f0(x) to g(y).
	- g(y): Measured MC distribution after running the simulation on f0(x).
			Used to build the response matrix A by mapping f0(x) to g(y).
	- f(x): Unfolded target function of the true distribution. f(x) is build
			from the basis functions used to describe f0(x) and the optimal
			coefficients aj calculated by the likelihodd fit.

	Input
	-----

	- bins_meas: Array which sets the binning for all functions dependent on
				 the measured varible y (measured MC). The first and last
				 values define the range of y in which the unfolding operates.
	- bins_unfold: Array which sets the binning of the target function f(x)
				   AFTER the unfolding. The binning is applied to the spline
				   representation of f(x) to get single points with error
				   estimation. The first and last values define the range
				   of x in which the target function is defined.
	- t: Array containing the inner knots for the (cubic) bspline
		 basis functions which are used to represent f0(x). The
		 spline representation is used to discretize the true MC
		 f0(x) to build the response matrix Aij. The necessary outer
		 knots are created by repeating the outermost knots
		 spline_degree times.
	"""
	def __init__(self, bins_meas, bins_unfold, t):
		self.bins_meas = bins_meas
		self.bins_unfold = bins_unfold

		# Use cubic splines to represent f0(x) and f(x)
		self.spline_deg = 3

		## Add necessary outer knots by repeating the first and last knot spline_deg times. Used from splines/bsplines.py.
		# Switch the outer knots creation technique
		mirror = False
		pre = np.zeros(self.spline_deg)
		post = np.zeros(self.spline_deg)
		if mirror:
			# Mirror internal knots at t0, tn-1
			pre = ( t[0] - (t[1:nouter+1] - t[0]) )[::-1]
			post = ( t[-1] + (t[-1] - t[-nouter-1:-1]) )[::-1]
		else:
			# Just repeat the outermost knots p times each
			pre.fill(t[0])
			post.fill(t[-1])
		# Save all knots in one knot array
		self.spline_knots = np.concatenate((pre, t, post))

		# Number of splines (= number of coefficients aj) used to represent f0(x) and f(x)
		self.n_splines = len(self.spline_knots) - self.spline_deg - 1


	def build_response_matrix(self, mc_meas, mc_truth):
		"""
		Builds the response matrix Aij. i is the number of bins_meas and
		j is the number of spline_knots. To get Aij we use the MC truth
		and measured MC to track how the BASIS FUNCTIONS pj(x) are mapped to
		g(y). This is equivalent to setting all aj = 1. Therefore, the resulting
		response matrix is independent of the basis coefficients aj which
		represent the true MC distribution f0(x).
		The aj are later determined in a likelihood fit to get the best
		possible mapping from g(y) to f(x) via A.

		Input
		-----

		- mc_truth: True data before the MC detection simulation. f0(x) is
				    the assumed distribution of the true data.
		- mc_meas: MC data after the detection simulation resulting from
				   mc_truth. g(y) is the measured distribution obtained
				   from simulation based on the true distribution f0(x) and
				   describes the measured values obtained from the detector.

		Returns
		-------

		- A: Response matrix, describing the mapping from the basis functions
			 pj(x) of the true distribution f0(x) = sum_j aj*pj(x) to the
			 measured distribution g(y). The mapping function is:
				 vec{g}(y) = mat{A} * vec{p}
				 or in coordinate form:
				 gi(y) = sum_j (Aij * pj)
		"""
		# Init response matrix Aij
		rows = len(self.bins_meas) - 1
		cols = self.n_splines
		self.A = np.zeros([rows, cols])

		## Calculate the spline representation of the MC truth f0(x).
		# Every column Aj is a histogram mc_meas in y with binning bins_meas, as if f0(x) would be described by only one single basis function pj. To get the correct representation, set the histogram weights proportional to the basis function pj. This way, if we add up all hisotgrams Aj, we get the original distribution mc_meas. The correct weights (spline coefficients) aj, which really describe f0(x) are found later with the maximum likelihood fit. Here we only create a histogram for every basis function.
		self.spline_coeff = np.zeros(cols)
		for j in range(self.n_splines):
			# Enable only the jth basis spline
			self.spline_coeff.fill(0)
			self.spline_coeff[j] = 1
			# Set the hist weight for Aj(y) proportional to pj(x)
			tck = (self.spline_knots, self.spline_coeff, self.spline_deg)
			basis_hist_weights = sci.splev(mc_meas, tck, ext=1)
			# Bin the measured MC with basis_hist_weights according to f0(x)=pj(x)
			hist, bins = np.histogram(
				mc_meas,
				bins=self.bins_meas,
				range=(self.bins_meas[0], self.bins_meas[1]),
				normed=False,
				weights = basis_hist_weights
				)
			self.A[:, j] = hist

		return self.A


	def fit_basis_coefficents(self, mc_meas):
		"""
		Returns the optimal fitting spline coefficents which represent the
		unfolded distribution f(x).
		The response matrix was created before with the assumption that
		aj = 1 for every j. Therefore we get the response matrix A decoupled
		from the coefficients aj. The aj are now chosen via a likelihood fit
		to describe best the distribution g(y) given by the matrix mapping
		equation:
			vec{g}(y) = mat{A} * vec{a}
			or in coordinate form:
			gi(y) = sum_j (Aij * aj)
		The whole information about the true MC distribution f0(x) is already
		included in the response matrix A at this stage.

		Inout
		-----

		- mc_meas: MC data after the detection simulation resulting from
				   mc_truth. g(y) is the measured distribution obtained
				   from simulation based on the true distribution f0(x) and
				   describes the measured values obtained from the detector.

		Returns
		-------

		- spline_true_coeff: Array of coefficients obtained from a likelihood
							 fit describing optimally the mapping from f(x)
							 to g(y). Use
							 	scipy.splev(
							 		x,
							 		(
							 			spline_knots,
							 			spline_true_coeff,
							 			spline_degree
							 			),
									ext=1
									)
							 to obtain the unfolded true function f(x).
		"""
		# Negative log-likelihood which is minimized
		def negllh(a, g_meas):
			"""
			Returns the negative log-likelihood function value for given
			values of the basis function coefficients aj and the measured
			values g_meas. The fitted values g_fitted are calculated by the
			matrix equation:
				vec{g}_fitted(a) = mat{A} * vec{a}

			Input
			-----

			- a: Coefficent array at which the neg log-likelihood is evaluated.
			- g_meas: Measured MC values.

			Returns
			-------

			Single value of the neg log-likelihood function for the given
			parameters.
			"""
			# Only for unbounded methods: push up the negllh for parameters a<0 to restrict to positive values of a
			if np.any(a < 0):
				return np.inf

			# Calculate the value of g(y) for the current coefficents
			g_fitted = np.dot(self.A, a)

			## Effectively constrain to positive coefficents by pushing up the neg log-likelihood when aj<0 occurs
			# Init with 0. If none of the below is true this is either the case lim x->0 (x*log x) = 0 or 0*log(non-zero) = 0
			measxlog = np.zeros_like(g_fitted)
			for i in range(len(g_fitted)):
				# Case: non-zero value * log(0) = -inf -> return +inf for negllh
				if (g_meas[i]>0) & (g_fitted[i]<=0):
					return np.inf
				# Case: if both values are non-zero/finite then just calculate normally
				if (g_meas[i]>0) & (g_fitted[i]>0):
					measxlog[i] = g_meas[i] * np.log(g_fitted[i])

			return np.sum( g_fitted - measxlog )

		# Get the measured values gi(y) from measured MC distribtuion
		g_meas, g_bins = np.histogram(
			mc_meas,
			bins=self.bins_meas,
			range=(self.bins_meas[0], self.bins_meas[1]),
			normed=False
			)

		# Fit coefficients a, starting from aj = 1 for every j. Coefficients must be positive.
		# bounds = [[0, None] for i in range(self.n_splines)]
		x0 = np.ones_like(self.spline_coeff)
		x0.fill(1.1)
		opt_res = sco.minimize(
			negllh,
			x0=x0,
			args=(g_meas),
			# bounds=bounds,
			method="Powell"
			)

		# Get the basis function coefficients from the result
		spline_true_coeff = opt_res.x

		print "Succes: {}".format(opt_res.success)
		print "Num of iter: {}".format(opt_res.nit)

		# Return tck tuple for scipy.interpolate.splev
		return (self.spline_knots, spline_true_coeff, self.spline_deg)






