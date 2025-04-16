import numpy as np

from SVM_classifier import SVM_classifier

class SVM_OvO:
    def __init__(self, learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter
        self.models = dict()

    def fit(self, X, y):
        self.classes = np.unique(y)
        for i in range(len(self.classes)):
            for j in range(i + 1, len(self.classes)):
                class_i = self.classes[i]
                class_j = self.classes[j]
                idx = np.where((y == class_i) | (y == class_j))
                X_pair = X[idx]
                y_pair = y[idx]

                y_binary = np.where(y_pair == class_i, 1, 0)

                clf = SVM_classifier(self.learning_rate, self.no_of_iterations, self.lambda_parameter)
                clf.fit(X_pair, y_binary)
                self.models[(class_i, class_j)] = clf

    def predict_proba(self, X):
        prob_votes = np.zeros((X.shape[0], len(self.classes)))

        for (class_i, class_j), clf in self.models.items():
            prob = clf.predict_proba(X)[:, 1]
            prob_votes[:, class_i] += prob
            prob_votes[:, class_j] += (1 - prob)

        prob_votes /= len(self.models)
        return prob_votes

    def predict(self, X, confidence_threshold=0.45):
        probas = self.predict_proba(X)
        predictions = []

        for prob in probas:
            max_prob = np.max(prob)
            if max_prob < confidence_threshold:
                predictions.append(None)
            else:
                predictions.append(np.argmax(prob))
        return predictions
