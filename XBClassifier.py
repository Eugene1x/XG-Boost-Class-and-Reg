import numpy as np

class DecisionStumpClassifier:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_class = None
        self.right_class = None

    def fit(self, X, Y):
        m, n = X.shape
        best_error = float('inf')

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for thresh in thresholds:
                left_mask = X[:, feature] <= thresh
                right_mask = ~left_mask

                left_prob = np.mean(Y[left_mask]) if np.any(left_mask) else 0
                right_prob = np.mean(Y[right_mask]) if np.any(right_mask) else 0

                preds = np.where(left_mask, left_prob, right_prob)
                error = np.mean((Y - preds) ** 2)

                if error < best_error:
                    best_error = error
                    self.feature_index = feature
                    self.threshold = thresh
                    self.left_class = left_prob
                    self.right_class = right_prob

    def predict(self, X):
        return np.where(X[:, self.feature_index] <= self.threshold,
                        self.left_class, self.right_class)