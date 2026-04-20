from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

X = load_iris().data # 4D data

pca = PCA(n_components=2) # Crush to 2D
X_reduced = pca.fit_transform(X) # Apply math

print(pca.explained_variance_ratio_) # Check retained info

