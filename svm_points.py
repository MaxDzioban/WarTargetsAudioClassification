import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class SVM_classifier():
    def __init__(self, learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01):
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

    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        predicted_labels = np.sign(output)
        y_hat = np.where(predicted_labels <= -1, 0, 1)
        return y_hat


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

    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.classes)))
        for (class_i, class_j), clf in self.models.items():
            pred = clf.predict(X)
            for index, p in enumerate(pred):
                if p == 1:
                    votes[index, class_i] += 1
                else:
                    votes[index, class_j] += 1

        return np.argmax(votes, axis=1)

np.random.seed(42)

X_class0 = np.random.normal(loc=[1, 5], scale=0.5, size=(30, 2))
X_class1 = np.random.normal(loc=[4, 4], scale=0.5, size=(30, 2))
X_class2 = np.random.normal(loc=[7, 1], scale=0.5, size=(30, 2))
X = np.vstack([X_class0, X_class1, X_class2])
y = np.array([0]*30 + [1]*30 + [2]*30)

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.title("Multiclass Dataset (3 classes)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = SVM_OvO(learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

print(classification_report(Y_test, predictions))

clf = model.models[(0, 1)]
w = clf.w
b = clf.b

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.title("Data visualization and SVM decision boundaries (One-vs-One)")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")

pairs = [(0,1), (0,2), (1,2)]
colors = ['r', 'g', 'b']
x_values = np.linspace(X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1, 200)

for (class_i, class_j), color in zip(pairs, colors):
    clf = model.models[(class_i, class_j)]
    w = clf.w
    b = clf.b
    y_values = (b - w[0] * x_values) / w[1]
    plt.plot(x_values, y_values, color=color, label=f'Decision boundary {class_i} vs {class_j}')

plt.legend()
plt.show()
