import numpy as np

def double_gaus(N=10000, locL=1., locR=2., sigmaL=.3, sigmaR=.3, sigmaS=.1):
	"""
	Returns N true and measured test data points. The true function is a
	double normal distribution at locR and locR with stddev sigmaL and sigmaR.
	The truth gets smeared with a normal distribution with sigmaS and shifted
	by an acceptance function.
	"""
	#true distribution
	true = np.zeros(N)
	true[N/2:] = np.random.normal(locL, sigmaL, N/2)
	true[:N/2] = 1. * np.random.normal(locR, sigmaR, N/2)

	# smearing
	measured = true + np.random.normal(0., sigmaS, N)

	# acceptance
	measured += 0.2 * (true - 1.)**2

	return measured, true

