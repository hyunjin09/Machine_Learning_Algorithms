import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

model = LogisticRegression(lr=0.0001, iters=10000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)
