import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

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

    CONFIDENCE_THRESHOLD = 0.45 

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    print(f"Train size: {X_train.shape[0]} samples")
    print(f"Test size: {X_test.shape[0]} samples")

    clf = SVM_OvO()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test, confidence_threshold=CONFIDENCE_THRESHOLD)
    y_proba = clf.predict_proba(X_test)

    class_names = [os.path.splitext(os.path.basename(f))[0] for f in args.csv_files]
    y_pred_named = [p if p is not None else len(class_names) for p in y_pred]
    class_names_extended = class_names + ["unknown"]

    print(classification_report(y_test, y_pred_named, target_names=class_names_extended))

    for i, probs in enumerate(y_proba):
        class_prob_str = ", ".join([f"{class_names[j]}: {probs[j]:.2f}" for j in range(len(class_names))])
        max_prob = np.max(probs)
        predicted_class = class_names[np.argmax(probs)]
        if max_prob < CONFIDENCE_THRESHOLD:
            print(f"file {i+1}: {class_prob_str} = none class (max {max_prob:.2f})")
        else:
            print(f"file {i+1}: {class_prob_str} = {predicted_class} (max {max_prob:.2f})")

    plt.scatter(range(len(y_test)), y_test, label="True", marker="o")
    plt.scatter(range(len(y_pred_named)), y_pred_named, label="Predicted", marker="x")
    plt.legend()
    plt.title("True vs Predicted Labels")
    plt.show()

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap="viridis")
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    max_range = (X_pca.max(axis=0) - X_pca.min(axis=0)).max() / 2.0
    mid_x = (X_pca[:, 0].max() + X_pca[:, 0].min()) * 0.5
    mid_y = (X_pca[:, 1].max() + X_pca[:, 1].min()) * 0.5
    mid_z = (X_pca[:, 2].max() + X_pca[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.title("PCA 3D Projection (Cube View)")
    plt.show()

    n_unknown = sum(p is None for p in y_pred)
    print(f"\n number of files classified as 'unknown': {n_unknown} from {len(y_pred)}")


if __name__ == "__main__":
    main()
