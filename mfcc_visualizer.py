import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

def load_all_csv_files(folder):
    data = []
    labels = []
    for file in sorted(os.listdir(folder)):
        if file.endswith("_mfcc.csv"):
            class_name = file.replace("_mfcc.csv", "")
            filepath = os.path.join(folder, file)
            features = np.loadtxt(filepath, delimiter=",")
            if features.ndim == 1:
                features = features.reshape(1, -1)
            data.append(features)
            labels.extend([class_name] * len(features))
    X = np.vstack(data)
    return X, np.array(labels)

def plot_sample_distribution(folder):
    counts = {}
    for file in sorted(os.listdir(folder)):
        if file.endswith("_mfcc.csv"):
            class_name = file.replace("_mfcc.csv", "")
            filepath = os.path.join(folder, file)
            features = np.loadtxt(filepath, delimiter=",")
            n = 1 if features.ndim == 1 else features.shape[0]
            counts[class_name] = n

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()))
    plt.title("Sample Count per Class")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("sample_count_per_class.png")
    plt.close()

def plot_pca_distribution(folder):
    X, y = load_all_csv_files(folder)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="tab10")
    plt.title("PCA Projection of MFCC Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig("pca_projection.png")
    plt.close()

def plot_mfcc_distributions(folder):
    X, y = load_all_csv_files(folder)
    df = pd.DataFrame(X[:, :3], columns=["MFCC1", "MFCC2", "MFCC3"])
    df["Class"] = y

    for coeff in ["MFCC1", "MFCC2", "MFCC3"]:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x=coeff, hue="Class", common_norm=False, fill=True, alpha=0.3)
        plt.title(f"Distribution of {coeff} by Class")
        plt.tight_layout()
        plt.savefig(f"{coeff.lower()}_distribution.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize MFCC feature distributions")
    parser.add_argument("--input", required=True, help="Folder containing *_mfcc.csv files")
    parser.add_argument("--plot", required=True, choices=["count", "pca", "mfcc"], help="Type of plot to generate")
    args = parser.parse_args()

    if args.plot == "count":
        plot_sample_distribution(args.input)
    elif args.plot == "pca":
        plot_pca_distribution(args.input)
    elif args.plot == "mfcc":
        plot_mfcc_distributions(args.input)

if __name__ == "__main__":
    main()
