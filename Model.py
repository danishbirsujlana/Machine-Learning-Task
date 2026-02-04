import numpy as np

class Model:
    def __init__(self, n_components=50, lr=0.05, epochs=500, reg=1e-3):
        self.n_components = n_components
        self.lr = lr
        self.epochs = epochs
        self.reg = reg

        # placeholders
        self.mean = None
        self.std = None
        self.W_pca = None
        self.W = None
        self.b = None
        self.class_weights = None

    # ---------- utilities ----------
    def _standardize(self, X):
        return (X - self.mean) / (self.std)

    def _softmax(self, Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    # ---------- PCA ----------
    def _fit_pca(self, X):
        cov = np.cov(X, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        self.W_pca = eigvecs[:, idx[:self.n_components]]

    def _pca_transform(self, X):
        return X @ self.W_pca

    # ---------- training ----------
    def fit(self, X, y):
        # standardization
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)

        # PCA
        self._fit_pca(Xs)
        Xp = self._pca_transform(Xs)

        n, d = Xp.shape
        n_classes = 4

        # class weights (macro-F1 friendly)
        counts = np.bincount(y, minlength=n_classes)
        self.class_weights = n / (n_classes * counts)

        # init params
        self.W = np.zeros((d, n_classes))
        self.b = np.zeros(n_classes)

        # one-hot labels
        Y = np.eye(n_classes)[y]

        for _ in range(self.epochs):
            logits = Xp @ self.W + self.b
            probs = self._softmax(logits)

            sample_weights = self.class_weights[y]  # shape (n,)
            weighted_error = (probs - Y) * sample_weights[:, np.newaxis]
            grad_W = (Xp.T @ weighted_error) / n + self.reg * self.W
            grad_b = weighted_error.mean(axis=0)

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

    # ---------- prediction ----------
    def predict(self, X):
        Xs = self._standardize(X)
        Xp = self._pca_transform(Xs)

        logits = Xp @ self.W + self.b
        probs = self._softmax(logits)
        return np.argmax(probs, axis=1)