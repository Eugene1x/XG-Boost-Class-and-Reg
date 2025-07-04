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
    
class GBC:
    def __init__(self, numberEstimators, learningRate):
        self.numberEstimators = numberEstimators
        self.learningrate = learningRate
        self.models = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, Y):
        pred = np.zeros(len(Y))  

        for i in range(self.numberEstimators):
            prob = self.sigmoid(pred)
            gradient = Y - prob  

            stump = DecisionStumpClassifier()
            stump.fit(X, gradient)

            update = stump.predict(X)
            pred += self.learningrate * update

            self.models.append(stump)

    def predict_proba(self, X):
        pred = np.zeros(X.shape[0])
        for model in self.models:
            pred += self.learningrate * model.predict(X)
        return self.sigmoid(pred)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)