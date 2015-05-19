import scipy.stats as scs
import numpy as np
import matplotlib.pyplot as plt

import testdata

# get double normal distributed testdata
measured, true = testdata.double_gaus()

# Compare true histogram to continous pdf
x = np.linspace(-0.5, 3.5, 500)
pdf = .5 * (scs.norm.pdf(x, loc=1., scale=.3) + scs.norm.pdf(x, loc=2., scale=.2))

plt.plot(x, pdf, "k-", label="true pdf")

plt.hist(
	(true, measured),
	bins=20,
	normed=True,
	label=["true", "measured"],
	histtype='stepfilled',
	alpha=.25
	)

plt.legend(loc="best")
plt.show()
