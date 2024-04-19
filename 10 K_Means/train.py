import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from KMeans import KMeans

np.random.seed(42)
X, y = datasets.make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
print (X.shape)

clusters = len(np.unique(y))
print(clusters)

model = KMeans(K=clusters, max_iters=150, plot_steps=False)
y_pred = model.predict(X)

model.plot()