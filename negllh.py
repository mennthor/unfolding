import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt


# Negative log-likelihood which is minimized
def negllh(a, g_meas):
	"""
	Returns the negative log-likelihood function value for given
	values of the basis function coefficients aj and the measured
	values g_meas. The fitted values g_fitted are calculated by the
	matrix equation:
		vec{g}_fitted(a) = mat{A} * vec{a}
	"""
	a = np.array(a, copy=False, ndmin=1)
	A = np.zeros((len(g_meas), len(a)))
	A[1,1] = 1
	A[2,3] = 5
	# Calculate the value of g(y) for the current coefficents
	g_fitted = np.dot(A, a)
	### Calculate piecewise to catch inf and 0*inf cases
	## inf cases in log(0)
	log0_mask = g_fitted > 0
	log = np.zeros_like(g_fitted)
	log[log0_mask] = np.log(g_fitted[log0_mask])
	log[np.logical_not(log0_mask)] = -np.inf
	# ## lim x->0 (x*log x) = 0
	# meas0_mask = g_meas > 0
	# measxlog = np.zeros_like(g_fitted)
	# # Friendly cases with finite or -inf result
	# measxlog[log0_mask] = g_meas[log0_mask] * log[log0_mask]
	# # Bad cases with 0*log(0)=0
	# measxlog[np.logical_not(log0_mask & meas0_mask)] = 0

	return np.sum( g_fitted - g_meas * log )


g_meas = [10, 12, 9, 7]

a = np.linspace(0, 10, 100)
y = np.zeros_like(a)
for i in range(len(a)):
	y[i] = negllh([a[i], 1, 1, 1], g_meas)

plt.plot(a, y)
plt.show()



