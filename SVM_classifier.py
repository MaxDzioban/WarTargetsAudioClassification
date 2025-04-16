import numpy as np

class SVM_classifier():
    def __init__(self, learning_rate=0.0001, no_of_iterations=5000, lambda_parameter=0.001):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        y_label = np.where(self.Y <= 0, -1, 1)

        for index, x_i in enumerate(self.X):
            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1
            if condition:
                dw = 2 * self.lambda_parameter * self.w
                db = 0
            else:
                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
                db = y_label[index]
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    def decision_function(self, X):
        return np.dot(X, self.w) - self.b

    def predict_proba(self, X):
        decision = self.decision_function(X)
        probs = 1 / (1 + np.exp(-decision))
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        decision = self.decision_function(X)
        return np.where(decision >= 0, 1, 0)
