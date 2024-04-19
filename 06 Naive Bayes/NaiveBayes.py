import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        #mean, var, prior for each class
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.prior = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y==c]
            self.mean[idx,:] = np.mean(X_c, axis=0)
            self.var[idx,:] = np.var(X_c, axis=0)
            self.prior[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        pred = [self._predict(x) for x in X]
        return np.array(pred)
    
    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.prior[idx])
            posterior = np.sum(np.log(self._pdf(x, idx)))
            posterior = posterior + prior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
            
    def _pdf(self, x, class_idx):
        numerator = np.exp(-((x - self.mean[class_idx])**2) / (2 * self.var[class_idx]))
        denominator = np.sqrt(2 * np.pi * self.var[class_idx])
        return numerator / denominator