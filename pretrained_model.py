import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

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


def save_model_bundle(model, scaler, class_names, threshold, path="model_bundle.pkl"):
    bundle = {
        "model": model,
        "scaler": scaler,
        "class_names": class_names,
        "threshold": threshold
    }
    joblib.dump(bundle, path)


def load_model_bundle(path="model_bundle.pkl"):
    bundle = joblib.load(path)
    return bundle["model"], bundle["scaler"], bundle["class_names"], bundle["threshold"]


def train_and_save(csv_files):
    X, y = load_data(csv_files)
    class_names = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
    threshold = 0.45

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    clf = SVM_OvO()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test, confidence_threshold=threshold)
    y_pred_named = [p if p is not None else len(class_names) for p in y_pred]
    class_names_extended = class_names + ["unknown"]

    print("Classification report (with 'unknown'):")
    all_labels = list(range(len(class_names))) + [len(class_names)]
    print(classification_report(y_test, y_pred_named, labels=all_labels, target_names=class_names_extended))

    save_model_bundle(clf, scaler, class_names, threshold)
    print("Model saved to model_bundle.pkl")


def predict_with_pretrained(vector_file, bundle_path="model_bundle.pkl"):
    clf, scaler, class_names, threshold = load_model_bundle(bundle_path)

    if vector_file.endswith(".npy"):
        X = np.load(vector_file)
    else:
        X = np.loadtxt(vector_file, delimiter=",")

    if X.ndim == 1:
        X = X.reshape(1, -1)

    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled, confidence_threshold=threshold)
    
    # for i, pred in enumerate(y_pred):
    #     if pred is None:
    #         print(f"Sample {i + 1}: Unknown class")
    #     else:
    #         print(f"Sample {i + 1}: {class_names[pred]}")
    
    predictions = []
    for i, pred in enumerate(y_pred):
        if pred is None:
            predictions.append("Unknown class")
        else:
            predictions.append(class_names[pred])
    return predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+", help="Paths to CSV files (one per class or a single feature vector for prediction)")
    parser.add_argument("--mode", choices=["train", "predict"], default="train", help="Mode: 'train' or 'predict'")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train_and_save(args.csv_files)
    else:
        if len(args.csv_files) != 1:
            raise ValueError("In 'predict' mode, exactly one file with the feature vector must be specified.")
        predict_with_pretrained(args.csv_files[0])

if __name__ == "__main__":
    main()
