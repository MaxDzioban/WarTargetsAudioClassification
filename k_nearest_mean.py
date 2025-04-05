import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class NearestMeanClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.centroids = {}

        for cls in self.classes:
            class_samples = X[y == cls]
            centroid = np.mean(class_samples, axis=0)
            self.centroids[cls] = centroid

    def predict(self, X):
        predictions = []
        for x in X:
            min_dist = float('inf')
            chosen_class = None
            for cls, centroid in self.centroids.items():
                dist = np.linalg.norm(x - centroid)
                if dist < min_dist:
                    min_dist = dist
                    chosen_class = cls
            predictions.append(chosen_class)
        return np.array(predictions)


def load_data():
    x = np.loadtxt("/Users/max/Downloads/Breast-Cancer-Wisconsin-Diagnostic-Dataset-Analysis-main/ar15_mfcc.csv", delimiter=",")
    y = np.loadtxt("/Users/max/Downloads/Breast-Cancer-Wisconsin-Diagnostic-Dataset-Analysis-main/speech_mfcc.csv", delimiter=",")
    z = np.loadtxt("/Users/max/Downloads/Breast-Cancer-Wisconsin-Diagnostic-Dataset-Analysis-main/tank_mfcc.csv", delimiter=",")

    X = np.vstack([x, y, z])
    labels = (
        [0] * len(x) +  # AR15
        [1] * len(y) +  # SPEECH
        [2] * len(z)    # TANK
    )
    y_all = np.array(labels)
    return X, y_all

def main():
    X, y = load_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = NearestMeanClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("📊 Nearest Mean Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["AR15", "SPEECH", "TANK"]))

    plt.scatter(range(len(y_test)), y_test, label="True", marker="o")
    plt.scatter(range(len(y_pred)), y_pred, label="Predicted", marker="x")
    plt.legend()
    plt.title("True vs Predicted Labels (Nearest Mean)")
    plt.show()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    plt.scatter(X_2d[y==0,0], X_2d[y==0,1], label="AR15", marker='o')
    plt.scatter(X_2d[y==1,0], X_2d[y==1,1], label="SPEECH", marker='x')
    plt.scatter(X_2d[y==2,0], X_2d[y==2,1], label="TANK", marker='^')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
