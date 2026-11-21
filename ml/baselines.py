"""Minimal Baselines and Leakage Probe."""
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from ml.config import ARTIFACTS_DIR, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_snv(arr):
    mean = arr.mean(axis=1, keepdims=True)
    std  = arr.std(axis=1,  keepdims=True)
    return (arr - mean) / (std + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"G:\My Drive\coalrock_project2\processed_data"))
    args = parser.parse_args()

    logging.info("Loading arrays...")
    X_train = np.load(args.data_dir / "X_train.npy")
    y_train = np.load(args.data_dir / "y_train.npy")
    X_test = np.load(args.data_dir / "X_test.npy")
    y_test = np.load(args.data_dir / "y_test.npy")
    
    # We only apply SNV to make it fair / simple, or just use raw if models prefer.
    # The prompt says "1-NN on SNV", others can use SNV too.
    X_train_snv = X_train[:, 0, :]
    X_test_snv = X_test[:, 0, :]
    
    results = {}
    
    # 1. Leakage probe (T1.6)
    logging.info("Running cosine similarity leakage probe...")
    # Just compute similarity of first 500 test to train to save memory
    sims = cosine_similarity(X_test_snv[:500], X_train_snv)
    max_sims = sims.max(axis=1)
    frac_leak = (max_sims > 0.999).mean()
    results["leakage_probe"] = {
        "frac_above_0.999": float(frac_leak)
    }
    logging.info(f"Fraction above 0.999 similarity: {frac_leak:.4f}")
    
    # 2. Baselines (Single seed)
    models = {
        "1-NN": KNeighborsClassifier(n_neighbors=1, metric="cosine"),
        "RBF-SVM": SVC(kernel="rbf", C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    }
    
    for name, model in models.items():
        logging.info(f"Training {name}...")
        model.fit(X_train_snv, y_train)
        preds = model.predict(X_test_snv)
        acc = accuracy_score(y_test, preds)
        results[name] = {"accuracy": float(acc)}
        logging.info(f"{name} Accuracy: {acc:.4f}")

    # PLS-DA
    logging.info("Training PLS-DA...")
    # Convert labels to one-hot for PLS-DA
    n_classes = len(np.unique(y_train))
    y_train_oh = np.eye(n_classes)[y_train]
    pls = PLSRegression(n_components=15)
    pls.fit(X_train_snv, y_train_oh)
    preds_pls = pls.predict(X_test_snv).argmax(axis=1)
    acc_pls = accuracy_score(y_test, preds_pls)
    results["PLS-DA"] = {"accuracy": float(acc_pls)}
    logging.info(f"PLS-DA Accuracy: {acc_pls:.4f}")

    with open(ARTIFACTS_DIR / "baselines.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
