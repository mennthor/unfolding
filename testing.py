import numpy as np
import matplotlib.pyplot as plt

import testdata


measured, true = testdata.double_gaus()

plt.hist((measured, true), bins=20, label=["measured", "true"],
		 histtype='bar', alpha=0.5,
		 )
plt.legend(loc="best")
plt.show()
