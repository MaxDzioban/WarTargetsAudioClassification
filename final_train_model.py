import os
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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


def load_data(csv_files):
    data = []
    labels = []

    for class_index, file_path in enumerate(csv_files):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        class_data = np.loadtxt(file_path, delimiter=",")
        data.append(class_data)
        labels += [class_index] * len(class_data)

    X = np.vstack(data)
    y = np.array(labels)
    return X, y

def parse_args():
    parser = argparse.ArgumentParser(description="Train OvO SVM on MFCC CSV data")
    parser.add_argument("csv_files", nargs="+", help="Paths to CSV files (one per class)")
    return parser.parse_args()



def main():
    args = parse_args()
    X, y = load_data(args.csv_files)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    print(f"Train size: {X_train.shape[0]} samples")
    print(f"Test size: {X_test.shape[0]} samples")

    clf = SVM_OvO()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Classification report:")
    class_names = [os.path.splitext(os.path.basename(f))[0] for f in args.csv_files]
    print(classification_report(y_test, y_pred, target_names=class_names))

    plt.scatter(range(len(y_test)), y_test, label="True", marker="o")
    plt.scatter(range(len(y_pred)), y_pred, label="Predicted", marker="x")
    plt.legend()
    plt.title("True vs Predicted Labels")
    plt.show()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
    plt.colorbar(scatter)
    plt.title("PCA Data Projection")
    plt.show()
if __name__ == "__main__":
    main()
