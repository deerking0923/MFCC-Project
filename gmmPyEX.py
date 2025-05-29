"""
===============
GMM 
===============
"""

import numpy as np
from sklearn.mixture import GaussianMixture
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
gm = GaussianMixture(n_components=2).fit(X)
print(gm.means_)
print(gm.predict([[1, 0], [7, 3]]))
print(gm.score_samples([[1, 1],[10, 5]]))
print(gm.score([[1, 1],[10, 5]]))
