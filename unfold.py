import numpy as np


class Blobel():
	"""
	docstring for Blobel:
	- bins_meas: array which sets the binning for all functions dependent on
				 the measured varible y. The first and last values define
				 the range of y in which the unfolding operates.
	- bins_unfold: array which sets the binning of the target function f(x)
				   which is unfolded. the binning is applied to the spline
				   representation of f(x). The first and last values define
				   the range of x in which the target function is defined.
	- spline_knots: array containing the inner knots for the (cubic) bspline
					basis functions which are used to represent f(x). the outer
					knots are automatically added by repeating spline_knots[0]
					abd spline_knots[-1] 3 times each.
	"""
	def __init__(self, bins_meas, bins_unfold, spline_knots):
		self.bins_meas = bins_meas
		self.bins_unfold = bins_unfold
		self.range_meas = range_meas
		self.range_unfold = range_unfold
		self.spline_knots = spline_knots

		# use cubic splines to represent the unfolded target function f(x)
		self.spline_deg = 3

		# add the outer knots
		self.pre = np.zeros(spline_deg)
		self.post = np.zeros(spline_deg)
		self.pre.fill(spline_knots[0])
		self.post.fill(spline_knots[-1])
		self.spline_knots = np.concatenate((self.pre, self.spline_knots, self.post))

		# number of splines used to represent f(x)
		self.n_splines = len(self.spline_knots - self.spline_deg - 1)


		def build_response_matrix(self, mc_meas, mc_truth):
			"""
			Builds the response matrix Aij where i is in bins_meas and
			j in spline_knots.
			"""