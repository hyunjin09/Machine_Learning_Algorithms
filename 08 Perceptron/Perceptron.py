import numpy as np

def unit_step_func(x):
    return np.where((x > 0), 1, 0)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.activation_func(y_pred)
            dw = np.mean(2 * np.dot(X.T, (y_pred - y)))
            db = np.mean(2 * (y_pred - y))
            self.weights = self.weights - dw
            self.bias = self.bias - db

    def predict(self, X):
        return self.activation_func(np.dot(X, self.weights) + self.bias)