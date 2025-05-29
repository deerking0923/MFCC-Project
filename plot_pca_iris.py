"""
=========================================================
PCA example with Iris Data-set
=========================================================

Principal Component Analysis applied to the Iris dataset.

See `here <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ for more
information on this dataset.

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 데이터 불러오기 (Iris 데이터셋)
data = load_iris()
X = data.data
y = data.target

# PCA 수행 (2개 주성분으로 축소)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 결과 출력
print("원본 데이터 형태:", X.shape)
print("PCA 변환 후 형태:", X_pca.shape)

# 시각화
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of IRIS Dataset')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.grid(True)
plt.show()
