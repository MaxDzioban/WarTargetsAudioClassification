import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class NearestMeanClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.centroids = {cls: np.mean(X[y == cls], axis=0) for cls in self.classes}

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [np.linalg.norm(x - centroid) for centroid in self.centroids.values()]
            closest_class = list(self.centroids.keys())[np.argmin(distances)]
            predictions.append(closest_class)
        return np.array(predictions)

def load_data(file_paths):
    data = []
    labels = []
    for i, path in enumerate(file_paths):
        class_data = np.loadtxt(path, delimiter=",")
        data.append(class_data)
        labels += [i] * len(class_data)
    X = np.vstack(data)
    y = np.array(labels)
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Nearest Mean Classifier for multiple classes")
    parser.add_argument("csv_files", nargs="+", help="Paths to CSV files (each representing one class)")
    args = parser.parse_args()

    X, y = load_data(args.csv_files)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = NearestMeanClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    class_names = [f"Class {i}" for i in np.unique(y)]
    print("Nearest Mean Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    plt.scatter(range(len(y_test)), y_test, label="True", marker="o")
    plt.scatter(range(len(y_pred)), y_pred, label="Predicted", marker="x")
    plt.legend()
    plt.title("True vs Predicted Labels (Nearest Mean)")
    plt.show()

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    for i in np.unique(y):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=f"Class {i}", marker='o')
    plt.legend()
    plt.title("PCA Visualization of Classes")
    plt.show()

if __name__ == "__main__":
    main()
