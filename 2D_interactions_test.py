import matplotlib.pyplot as plt
import numpy as np

import interactions

L = 15
alpha = 1.0
Jbase = interactions.powerlaw_pbc_2D(L, alpha)

plt.imshow(Jbase)
plt.show()
