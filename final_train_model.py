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

from SVM_classifier import SVM_classifier
from SVM_OvO import SVM_OvO

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
    parser.add_argument("--mode", type=str, choices=["debug", "pca", "scatter" , "none"], default="none",
                        help="Execution mode: debug, pca, scatter, none")
    return parser.parse_args()

def run_pca_visualization(X_scaled, y):
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
    mid = X_pca.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    plt.title("PCA 3D Projection (Cube View)")
    plt.show()

def run_debug_report(y_test, y_pred, y_proba, class_names, threshold):
    y_pred_named = [p if p is not None else len(class_names) for p in y_pred]
    class_names_extended = class_names + ["unknown"]
    print(classification_report(y_test, y_pred_named, target_names=class_names_extended))

    for i, probs in enumerate(y_proba):
        class_prob_str = ", ".join([f"{class_names[j]}: {probs[j]:.2f}" for j in range(len(class_names))])
        max_prob = np.max(probs)
        predicted_class = class_names[np.argmax(probs)]
        if max_prob < threshold:
            print(f"file {i+1}: {class_prob_str} = none class (max {max_prob:.2f})")
        else:
            print(f"file {i+1}: {class_prob_str} = {predicted_class} (max {max_prob:.2f})")

    n_unknown = sum(p is None for p in y_pred)
    print(f"\nNumber of files classified as 'unknown': {n_unknown} from {len(y_pred)}")

def run_scatter_plot(y_test, y_pred, class_names):
    y_pred_named = [p if p is not None else len(class_names) for p in y_pred]
    plt.scatter(range(len(y_test)), y_test, label="True", marker="o")
    plt.scatter(range(len(y_pred_named)), y_pred_named, label="Predicted", marker="x")
    plt.legend()
    plt.title("True vs Predicted Labels")
    plt.show()

def main():
    args = parse_args()
    CONFIDENCE_THRESHOLD = 0.45

    X, y = load_data(args.csv_files)
    class_names = [os.path.splitext(os.path.basename(f))[0] for f in args.csv_files]
    X_scaled = StandardScaler().fit_transform(X)

    if args.mode == "pca":
        run_pca_visualization(X_scaled, y)
        return

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    clf = SVM_OvO()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test, confidence_threshold=CONFIDENCE_THRESHOLD)
    y_proba = clf.predict_proba(X_test)

    if args.mode in ["debug"]:
        run_debug_report(y_test, y_pred, y_proba, class_names, CONFIDENCE_THRESHOLD)

    if args.mode in ["scatter"]:
        run_scatter_plot(y_test, y_pred, class_names)

    if args.mode in ["none"]:
        y_pred_named = [p if p is not None else len(class_names) for p in y_pred]
        from sklearn.utils.multiclass import unique_labels
        true_labels_present = set(unique_labels(y_test, y_pred_named))
        target_names_dynamic = [class_names[i] if i < len(class_names) else "unknown" for i in sorted(true_labels_present)]
        print(classification_report(
            y_test, y_pred_named,
            target_names=target_names_dynamic,
            zero_division=0
        ))

        for i, probs in enumerate(y_proba):
            class_prob_str = ", ".join([f"{class_names[j]}: {probs[j]:.2f}" for j in range(len(class_names))])
            max_prob = np.max(probs)
            predicted_class = class_names[np.argmax(probs)]
            if max_prob < CONFIDENCE_THRESHOLD:
                print(f"file {i+1}: {class_prob_str} = unknown class (max {max_prob:.2f})")

if __name__ == "__main__":
    main()
