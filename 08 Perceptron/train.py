import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Perceptron import Perceptron

X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

model = Perceptron(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:,0], X_train[:,1], marker="o", c=y_train)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-model.weights[0] * x0_1 - model.bias) / model.weights[1]
x1_2 = (-model.weights[0] * x0_2 - model.bias) / model.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin - 3, ymax + 3])

plt.show()