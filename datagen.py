import numpy as np
import scipy.stats as scs

# Provides several functions that generate test data to check the unfolding algortihm.

def double_gaus(N=10000, locL=1., locR=2., sigmaL=.3, sigmaR=.2, sigmaS=.2, n_bins_meas=10):
	"""
	Returns N true and measured test data points, the generating true pdf and
	a default binning which avoids empty bins in the measured variable.
	The true function is adouble normal distribution at locR and locR
	with stddev sigmaL and sigmaR. The truth gets smeared with a normal
	distribution with sigmaS and additionally shifted.
	"""
	#true distribution
	true = np.concatenate((
		np.random.normal(locL, sigmaL, N/2),
		np.random.normal(locR, sigmaR, N/2)
		))
	# smearing
	measured = true + np.random.normal(0., sigmaS, N)
	# shifting
	measured += 0.2 * (true - 2.)**2
	# Also return the generating true pdf
	x = np.linspace(-1, 5, 500)
	pdf = np.zeros([2, len(x)])
	pdf[0] = x
	pdf[1] = .5 * (scs.norm.pdf(x, loc=locL, scale=sigmaL) \
				 + scs.norm.pdf(x, loc=locR, scale=sigmaR))

	# Default binning
	default_binning = np.linspace(np.amin(measured), np.amax(measured), n_bins_meas)

	return measured, true, pdf, default_binning


def gaus(N=10000, loc=2., sigma=.2, sigmaS=.5, n_bins_meas=10):
	"""
	Returns N true and measured test data points, the generating true pdf and
	a default binning which avoids empty bins in the measured variable.
	The true function is a single normal distribution at loc with stddev sigma.
	The truth gets smeared with a normal distribution with sigmaS and
	additionally shifted.
	"""
	#true distribution
	true = np.random.normal(loc, sigma, N)
	# smearing
	measured = true + np.random.normal(0., sigmaS, N)
	# shifting
	measured += 0.2 * (true - 2.)**2
	# Also return the generating true pdf
	x = np.linspace(-1, 5, 500)
	pdf = np.zeros([2, len(x)])
	pdf[0] = x
	pdf[1] = scs.norm.pdf(x, loc=loc, scale=sigma)

	# Default binning
	default_binning = np.linspace(np.amin(measured), np.amax(measured), n_bins_meas)

	return measured, true, pdf, default_binning


def uniform(N=10000, left=0., right=3., sigma=.5, n_bins_meas=10):
	"""
	Returns a uniformly distributed truth in [left,right] and a gaussian
	measured distribution centered in the middle of the intervall [left,right]
	with stddev sigma.
	"""
	#true distribution
	true = np.random.uniform(left, right, N)
	# measured distribution
	measured = np.random.normal((left+right)/2., sigma, N)
	# Also return the generating true pdf
	x = np.linspace(-1, 5, 500)
	pdf = np.zeros([2, len(x)])
	pdf[0] = x
	pdf[1] = scs.uniform.pdf(x, left, right-left)
	# Default binning
	default_binning = np.linspace(np.amin(measured), np.amax(measured), n_bins_meas)

	return measured, true, pdf, default_binning


def example():
	"""
	Returns 9000 points each of a uniformly distributed truth and a
	gaussian measured distribution centered in the middle of the intervall
	with FIXED values.
	"""
	N=9000
	#true distribution -> uniform between 0 and 3
	true = np.zeros(N)
	true[:N/3-1] = 0.5
	true[N/3:2*N/3-1] = 1.5
	true[2*N/3:] = 2.5
	# measured distribution -> fixed normal distribution with sigma=.5 and mean=1.5
	measured = np.zeros(N)
	measured[:1300-1] = 0.5
	measured[1300:1300+6000-1] = 1.5
	measured[1300+6000:] = 2.5
	# Also return the generating true pdf
	x = np.linspace(-1, 5, 500)
	pdf = np.zeros([2, len(x)])
	pdf[0] = x
	pdf[1] = scs.uniform.pdf(x, 0, 3)

	return measured, true, pdf, [0,1,2,3]


# def blobel(N=10000):
# 	"""
# 	Example function from Blobel, S. 253
# 	"""
# 	#true distribution
# 	bk = np.array([1., 10., 5.])
# 	xk = np.array([.4, .8, 1.5])
# 	gk = np.array([2., .2, .2])
# 	true =
# 	# smearing
# 	measured = true + np.random.normal(0., sigmaS, N)
# 	# shifting
# 	measured += 0.2 * (true - 2.)**2
# 	# Also return the generating true pdf
# 	x = np.linspace(-1, 5, 500)
# 	pdf = np.zeros([2, len(x)])
# 	pdf[0] = x
# 	pdf[1] = .5 * (scs.norm.pdf(x, loc=locL, scale=sigmaL) \
# 				 + scs.norm.pdf(x, loc=locR, scale=sigmaR))

# 	# Default binning
# 	default_binning = np.linspace(np.amin(measured), np.amax(measured), n_bins_meas)

# 	return measured, true, pdf, default_binning


