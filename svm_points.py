import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from SVM_OvO import SVM_OvO
from SVM_classifier import SVM_classifier


def main():
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

if __name__ == "__main__":
    main()
