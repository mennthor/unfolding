from __future__ import print_function, division
import numpy as np
import scipy.interpolate as sci
import scipy.optimize as sco


class Blobel():
	"""
	Unfolding method after Blobel.

	Used names in this class:
	- f0(x): True distribution before detection MC simulation. Used to build
			 the response matrix A by mapping f0(x) to g(y).
	- g(y): Measured MC distribution after running the simulation on f0(x).
			Used to build the response matrix A by mapping f0(x) to g(y).
	- f(x): Unfolded target function of the true distribution. f(x) is build
			from the basis functions used to describe f0(x) and the optimal
			coefficients aj calculated by the likelihodd fit.

	Parameters
	----------
	bins_meas : array-like
		Array which sets the binning for all functions dependent on
		the measured varible y (measured MC). The first and last
		values define the range of y in which the unfolding operates.
	bins_unfold : array-like
		Array which sets the binning of the target function f(x)
		AFTER the unfolding. The binning is applied to the spline
		representation of f(x) to get single points with error
		estimation. The first and last values define the range
		of x in which the target function is defined.
	t : array-like
		Array containing the inner knots for the (cubic) bspline
		basis functions which are used to represent f0(x). The
		spline representation is used to discretize the true MC
		f0(x) to build the response matrix Aij. The necessary outer
		knots are created by repeating the outermost knots
		spline_degree times.
	"""
	def __init__(self, bins_meas, bins_unfold, t):
		self.bins_meas = bins_meas
		self.bins_unfold = bins_unfold

		# Resulting number of bins
		self.n_bins_meas = len(self.bins_meas) - 1
		self.n_bins_unfold = len(self.bins_unfold) - 1

		# Use cubic splines to represent f0(x) and f(x)
		self.spline_deg = 3

		## Add necessary outer knots by repeating the first and last knot spline_deg times. Used from splines/bsplines.py.
		# Switch the outer knots creation technique
		extend = True
		pre = np.zeros(self.spline_deg)
		post = np.zeros(self.spline_deg)
		if extend:
			# Extend internal knot t1 at t0 and tn-1 at tn, with increasing distance t1-t0, tn-tn-1 times spline_deg
			pre = [(t[0] - (i+1)*(t[1]-t[0])) for i in range(self.spline_deg)  ][::-1]
			post = [(t[-1] + (i+1)*(t[-1]-t[-2])) for i in range(self.spline_deg)]
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
		Builds the response matrix Aij.

		i is the number of bins_meas and
		j is the number of spline_knots. To get Aij we use the MC truth
		and measured MC to track how the BASIS FUNCTIONS pj(x) are mapped to
		g(y). This is equivalent to setting all aj = 1. Therefore, the resulting
		response matrix is independent of the basis coefficients aj which
		represent the true MC distribution f0(x).
		The aj are later determined in a likelihood fit to get the best
		possible mapping from g(y) to f(x) via A.

		Parameters
		----------
		mc_truth : array-like
			True data before the MC detection simulation. f0(x) is
			the assumed distribution of the true data.
		mc_meas : array-like
			MC data after the detection simulation resulting from
			mc_truth. g(y) is the measured distribution obtained
			from simulation based on the true distribution f0(x) and
			describes the measured values obtained from the detector.

		Returns
		-------
		A : array-like
			Response matrix, describing the mapping from the basis functions
			pj(x) of the true distribution f0(x) = sum_j aj*pj(x) to the
			measured distribution g(y). The mapping function is:
				vec{g}(y) = mat{A} * vec{p}
			or in coordinate form:
				gi(y) = sum_j (Aij * pj)
		"""
		## Init response matrix Aij
		# Number of rows is determined by the binning in y of the measured distribution
		rows = len(self.bins_meas) - 1
		# Each coloum j represents one basis function pj(x), so the number of splines is the number of columns
		cols = self.n_splines
		self.A = np.zeros([rows, cols])

		## Calculate the spline representation of the MC truth f0(x).
		# Every entry Aij of A is the probability that the component fj of the distribution of the true variable x is mapped to the component gi(y) of the distribution of the measured variable y.
		# Now f0(x) is described by a sum of basis functions pj. Therefore every column Aj is a histogram of mc_meas in y with binning bins_meas, as if f0(x) would be described by only one single basis function pj. So we and up with as many columns as there are basis function used to describe f0(x).
		# To get the correct representation, set the histogram weights proportional to the basis functions pj. Because sum_j pj = 1, if we add up all hisotgrams Aj unnormalized, we would get the original distribution mc_meas. But as Aij are probabilities we normalize A so that the row entries add up to one in every row. This way there are noc missing entries in gi = sum_j Aij fj.
		# The correct weights (spline coefficients) aj, which really describe f0(x) are found later with the maximum likelihood fit. Here we only create a histogram for every basis function to describe the mapping probabilities. This also makes sure, that the unfolding procedure is independent of the MC truth used to build the response matrix.
		self.spline_coeff = np.zeros(cols)
		for j in range(self.n_splines):
			# Enable only the jth basis spline
			self.spline_coeff.fill(0)
			self.spline_coeff[j] = 1
			# Set the hist weights for the histogram filled in Aj(y) proportional to pj(x). ext=1 cuts off to 0 outside the knots.
			tck = (self.spline_knots, self.spline_coeff, self.spline_deg)
			basis_hist_weights = sci.splev(mc_truth, tck, ext=1)
			# Bin the measured MC with basis_hist_weights according to f0(x)=pj(x). The first and last entry in bins_meas define the borders, points outside are ignored by numpy.histogram.
			hist, _ = np.histogram(
				mc_meas,
				bins=self.bins_meas,
				density=False,
				weights=basis_hist_weights
				)
			# Fill the created histogram to the jth column of A. In every column j of A is the binned distribution of g(y) weighted with pj(x).
			self.A[:, j] = hist

		# Normalization constant to normalize g(y) to 1 to represent a pdf. So every entry in A has to be divided by this number.
		# norm = np.dot(
		# 	np.histogram(mc_meas, self.bins_meas, density=False)[0],
		# 	np.diff(self.bins_meas)
		# 	)
		# self.A /= norm

		#  Normalization ansatz by Max Noethe:
		rowsums = self.A.sum(axis=1)[:,np.newaxis]
		# binwidth = np.diff(self.bins_meas)[:,np.newaxis]
		self.A = self.A / rowsums
		# self.A = self.A * binwidth

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

		Parameters
		----------

		mc_meas : array-like
			MC data after the detection simulation resulting from
			mc_truth. g(y) is the measured distribution obtained
			from simulation based on the true distribution f0(x) and
			describes the measured values obtained from the detector.

		Returns
		-------
		tck : tuple
			Tuple of (spline_knots, spline_true_coeff, spline_degree) from
			the fit describing optimally the mapping from f(x) to g(y). Use
				scipy.splev(x, tck,	ext=1)
			to obtain the unfolded true function f(x).
		"""
		# Get the binned measured values gi(y) from the measured MC distribtuion
		g_meas, _ = np.histogram(
			mc_meas,
			bins=self.bins_meas,
			density=False
			)

		# Fit coefficients a, starting from aj = 1 for every j. Coefficients must be positive.
		# bounds = [[0, None] for i in range(self.n_splines)]
		x0 = np.ones_like(self.spline_coeff)
		opt_res = sco.minimize(
			# self.chisquared,
			self.negllh,
			x0=x0,
			args=(g_meas),
			# jac=False,
			# bounds=bounds,
			method="Nelder-Mead"
			)

		# Get the basis function coefficients from the fit result
		spline_true_coeff = opt_res.x

		# Return tck tuple for scipy.interpolate.splev. The tuple tck completely describes the unfolded function f(x) by its spline representation.
		return (self.spline_knots, spline_true_coeff, self.spline_deg)



	def chisquared(self, a, g_meas):
		"""
		Returns the chi-square function value for given values
		of the basis function coefficients aj and the measured
		values g_meas.

		The fitted values g_fitted are calculated by the
		matrix equation:
			vec{g}_fitted(a) = mat{A} * vec{a}

		Parameters
		----------
		a : array-like
			Coefficent array at which the chi-square is evaluated.
		g_meas : array-like
			Contains measured MC values.

		Returns
		-------
		chi2 : float
			Single value of the chi-square function for the given parameters.
		"""
		# Only for unbounded methods: push up the negllh for parameters a<0 to effectively restrict to positive values of a
		if np.any(a < 0):
			return np.inf

		# Calculate the value of g(y) for the current coefficents
		g_fitted = np.dot(self.A, a)

		# chi-square value to be minimized
		return np.sum((g_meas - g_fitted)**2)

	def negllh(self, a, g_meas):
		"""
		Returns the negative log-likelihood function value for given
		values of the basis function coefficients aj and the measured
		values g_meas.

		The fitted values g_fitted are calculated by the
		matrix equation:
			vec{g}_fitted(a) = mat{A} * vec{a}

		Parameters
		----------
		a : array-like
			Coefficent array at which the chi-square is evaluated.
		g_meas : array-like
			Contains measured MC values.

		Returns
		-------
		negllh : float
			Single value of the neg log-likelihood function for the given
			parameters.
		"""
		# Only for unbounded methods: push up the negllh for parameters a<0 to effectively restrict to positive values of a
		if np.any(a < 0):
			return np.inf

		# Calculate the value of g(y) for the current coefficents
		g_fitted = np.dot(self.A, a)

		## Catch log(0) cases
		# Init with 0. If none of the below is true this is either the case lim x->0 (x*log x) = 0 or 0*log(non-zero) = 0
		measxlog = np.zeros_like(g_fitted)
		for i in range(len(g_fitted)):
			# Case: non-zero * log(0) = -inf -> return +inf for negllh
			if (g_meas[i]>0) & (g_fitted[i]<=0):
				return np.inf
			# Case: if both values are non-zero/finite then just calculate normally
			if (g_meas[i]>0) & (g_fitted[i]>0):
				measxlog[i] = g_meas[i] * np.log(g_fitted[i])

		return np.sum( g_fitted - measxlog )



	def __str__(self):
		"""
		Returns a string containing pre-formatted information about the
		unfolding parameters given at class creation.
		"""
		return  "\n" \
				"## Unfolding parameters: \n" \
				"# Measured g(y) binning:\n" \
				"    {}\n" \
				"# Resulting num of y bins: {}\n" \
				"# Unfolded f(x) binning:\n" \
				"    {}\n" \
				"# Resulting num of x bins: {}\n" \
				"# Spline knots to represent f0(x):\n" \
				"    Inner: {}\n" \
				"    Outer: {}\n" \
				"# Spline degree: {}\n" \
				"# Resulting num of splines: {}\n" \
				"".format(
					self.bins_meas,
					self.n_bins_meas,
					self.bins_unfold,
					self.n_bins_unfold,
					self.spline_knots[self.spline_deg:-self.spline_deg],
					np.concatenate((
						self.spline_knots[:self.spline_deg],
						self.spline_knots[-self.spline_deg:]
						)),
					self.spline_deg,
					self.n_splines,
					)

