import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from NaiveBayes import NaiveBayes

X, y = datasets.make_classification(n_samples=1000, n_features=100, n_classes=2, random_state=1234)
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.25, random_state=1234)

model = NaiveBayes()
model.fit(X, y)
pred = model.predict(X_test)
acc = np.sum(pred == y_test) / len(y_test)
print(acc)