import numpy as np
import scipy.stats as scs

def double_gaus(N=10000, locL=1., locR=2., sigmaL=.3, sigmaR=.2, sigmaS=.2):
	"""
	Returns N true and measured test data points. The true function is a
	double normal distribution at locR and locR with stddev sigmaL and sigmaR.
	The truth gets smeared with a normal distribution with sigmaS and shifted
	by an acceptance function.
	"""
	#true distribution
	true = np.concatenate((
		np.random.normal(locL, sigmaL, N/2),
		np.random.normal(locR, sigmaR, N/2)
		))

	# smearing
	measured = true + np.random.normal(0., sigmaS, N)

	# acceptance
	measured += 0.2 * (true - 2.)**2

	# Also return the generating true pdf
	x = np.linspace(-0.5, 3.5, 500)
	pdf = np.zeros([2, len(x)])
	pdf[0] = x
	pdf[1] = .5 * (scs.norm.pdf(x, loc=1., scale=.3) + scs.norm.pdf(x, loc=2., scale=.2))

	return measured, true, pdf

