import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr
        self.iters = iters

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iters):
            z = X.dot(self.weights) + self.bias
            y_pred = sigmoid(z)

            dw = (1 / n_samples) * X.T.dot(y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - dw * self.lr
            self.bias = self.bias - db * self.lr

    def predict(self, X):
        z = X.dot(self.weights) + self.bias
        y_pred = sigmoid(z)
        class_pred = [1 if y > 0.5 else 0 for y in y_pred]
        return class_pred
