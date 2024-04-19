import numpy as np
#X : (n, features)

# a = np.zeros((1, 1))
# b = np.zeros(1)
# print((b).shape)
# print(np.dot(b,a).shape)


class LinearRegression:
    
    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = X.dot(self.weights) + self.bias #(n,)

            dw = (2/n_samples) * np.dot(X.T, (y_pred-y))
            db = (2/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = X.dot(self.weights) + self.bias
        return y_pred
