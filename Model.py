import numpy as np

class Model:
    def __init__(self, components=50, lr=0.05, epochs=500, reg=1e-3):
        self.components = components
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.mean = None
        self.std = None
        self.W_pca = None
        self.W = None
        self.b = None
        self.classWeights = None

    def standardize(self, X):
        return (X - self.mean) / (self.std)

    def softmax(self, Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    # PCA
    def fit_pca(self, X):
        cov = np.cov(X, rowvar=False)
        eigenVals, eigenVecs = np.linalg.eigh(cov)
        idx = np.argsort(eigenVals)[::-1]
        self.W_pca = eigenVecs[:, idx[:self.components]]

    def pca_transform(self, X):
        return X @ self.W_pca

    # train
    def fit(self, X, y):
        # standardise
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        Xs = self.standardize(X)

        # PCA
        self.fit_pca(Xs)
        Xp = self.pca_transform(Xs)

        n, d = Xp.shape
        n_classes = 4

        # weights
        counts = np.bincount(y, minlength=n_classes)
        self.classWeights = n / (n_classes * counts)

        self.W = np.zeros((d, n_classes))
        self.b = np.zeros(n_classes)
        Y = np.eye(n_classes)[y]

        for _ in range(self.epochs):
            logits = Xp @ self.W + self.b
            prob = self.softmax(logits)

            sample_weights = self.classWeights[y]
            weighted_error = (prob - Y) * sample_weights[:, np.newaxis]
            grad_W = (Xp.T @ weighted_error) / n + self.reg * self.W
            grad_b = weighted_error.mean(axis=0)

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

    # predict
    def predict(self, X):
        Xs = self.standardize(X)
        Xp = self.pca_transform(Xs)
        logits = Xp @ self.W + self.b
        prob = self.softmax(logits)
        return np.argmax(prob, axis=1)