import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree

data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

model = DecisionTree()
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = np.sum(pred == y_test) / len(y_test)
print(acc)
