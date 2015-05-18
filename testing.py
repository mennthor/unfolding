import matplotlib.pyplot as plt

import testdata

measured, true = testdata.double_gaus()

plt.hist((measured, true), bins=20, label=["measured", "true"], histtype='step', alpha=1.0)
plt.legend(loc="best")
plt.show()
