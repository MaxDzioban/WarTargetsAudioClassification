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


def load_data():
    x = np.loadtxt("ar15_mfcc.csv", delimiter=",")
    y = np.loadtxt("speech_mfcc.csv", delimiter=",")
    z = np.loadtxt("tank_mfcc.csv", delimiter=",")

    X = np.vstack([x, y, z])
    labels = (
        [0] * len(x) +  # AR15
        [1] * len(y) +  # SPEECH
        [2] * len(z)    # TANK
    )
    y_all = np.array(labels)
    return X, y_all

# ==== Тренування та оцінка ====

# def main():
#     X, y = load_data()

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

#     clf = SVM_OvO()
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)

#     print("✅ Classification report:")
#     print(classification_report(y_test, y_pred, target_names=["AR15", "SPEECH", "TANK"]))

#     # (необов'язково) візуалізація
#     plt.scatter(range(len(y_test)), y_test, label="True", marker="o")
#     plt.scatter(range(len(y_pred)), y_pred, label="Predicted", marker="x")
#     plt.legend()
#     plt.title("True vs Predicted Labels")
#     plt.show()
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def main():
    X, y = load_data()

    # Масштабуємо дані
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Розділяємо на train і test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )

    # Збереження окремих датасетів, якщо потрібно
    # np.save("X_train.npy", X_train)
    # np.save("X_test.npy", X_test)
    # np.save("y_train.npy", y_train)
    # np.save("y_test.npy", y_test)

    print(f"Train size: {X_train.shape[0]} samples")
    print(f"Test size: {X_test.shape[0]} samples")

    # Навчання моделі
    clf = SVM_OvO()
    clf.fit(X_train, y_train)

    # Прогноз
    y_pred = clf.predict(X_test)

    # Оцінка
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["AR15", "SPEECH", "TANK"]))

    # Візуалізація
    plt.scatter(range(len(y_test)), y_test, label="True", marker="o")
    plt.scatter(range(len(y_pred)), y_pred, label="Predicted", marker="x")
    plt.legend()
    plt.title("True vs Predicted Labels")
    plt.show()

if __name__ == "__main__":
    main()