import numpy as np
import matplotlib.pyplot as plt

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None
    

    def fit(self,X,Y):
        m, n = X.shape
        bestError = float('inf')


        for feature in range(n):
            threshholds = np.unique(X[:, feature])
            for thresh in threshholds:
                leftMask = X[:, feature] <= thresh
                rightMask = ~leftMask


                leftValue = np.mean(Y[leftMask]) if np.any(leftMask) else 0
                rightValue = np.mean(Y[rightMask]) if np.any(rightMask) else 0

                preds = np.where(leftMask, leftValue, rightValue)
                error = np.mean((Y - preds ) ** 2)

                if error < bestError:
                    bestError = error
                    self.feature_index = feature
                    self.threshold = thresh
                    self.left_value = leftValue
                    self.right_value = rightValue

    def predict(self,X):
        return np.where(X[:,self.feature_index] <= self.threshold, 
                        self.left_value, self.right_value)
    

class GBR:
    def __init__(self, numberEstimators, learningRate):
        self.numberEstimators = numberEstimators
        self.learningrate = learningRate
        self.models = []

    def fit(self, X, Y):
        pred = np.zeros(len(Y))


        for i in range(self.numberEstimators):
            res = Y - pred

            stump = DecisionStump
            stump.fit(X,res)
