"""Main evaluation script for SpectraSense."""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, 
    cohen_kappa_score, matthews_corrcoef, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, average_precision_score, top_k_accuracy_score
)
from statsmodels.stats.proportion import proportion_confint
from scipy import stats

from ml.config import ARTIFACTS_DIR, MODELS_DIR, DATA_DIR, FUEL_CLASSES, BASE_DIR

# Add project root to path so we can import backend.model
sys.path.insert(0, str(BASE_DIR))
from backend.model import PCViT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class DummyLabelEncoder:
    def __init__(self):
        self.classes_ = np.array(FUEL_CLASSES + [f"Rock {i}" for i in range(12)])

def compute_bootstrap_ci(y_true, y_pred, metric_fn, n_resamples=2000, alpha=0.05):
    """Compute bootstrap confidence interval for a given metric."""
    np.random.seed(42)
    n = len(y_true)
    scores = []
    for _ in range(n_resamples):
        indices = np.random.choice(n, n, replace=True)
        scores.append(metric_fn(y_true[indices], y_pred[indices]))
    return np.percentile(scores, [alpha/2 * 100, (1 - alpha/2) * 100])

def evaluate_model(y_true, y_pred, y_prob, label_encoder, cost_ratio=10.0):
    """Compute comprehensive evaluation metrics."""
    # Extract class names and indices
    classes = label_encoder.classes_
    is_fuel = np.array([c in FUEL_CLASSES for c in classes])
    
    # Compute multi-class metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    try:
        top2_acc = top_k_accuracy_score(y_true, y_prob, k=2)
        top3_acc = top_k_accuracy_score(y_true, y_prob, k=3)
    except ValueError:
        top2_acc, top3_acc = None, None

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    
    per_class_metrics = {}
    for i, cls in enumerate(classes):
        if i < len(precision):
            per_class_metrics[cls] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i])
            }

    conf_mat = confusion_matrix(y_true, y_pred)
    conf_mat_norm = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Bootstrap CIs
    acc_ci = compute_bootstrap_ci(y_true, y_pred, accuracy_score)
    macro_f1_ci = compute_bootstrap_ci(y_true, y_pred, lambda yt, yp: f1_score(yt, yp, average='macro'))

    # Binary Error Decomposition
    true_is_fuel = is_fuel[y_true]
    pred_is_fuel = is_fuel[y_pred]
    
    coal_to_coal = np.sum(true_is_fuel & pred_is_fuel)
    rock_to_rock = np.sum((~true_is_fuel) & (~pred_is_fuel))
    coal_to_rock = np.sum(true_is_fuel & (~pred_is_fuel))
    rock_to_coal = np.sum((~true_is_fuel) & pred_is_fuel)
    
    # Binary safety metrics
    # Positive class = Gangue (Rock)
    true_gangue = ~true_is_fuel
    pred_gangue = ~pred_is_fuel
    prob_gangue = np.sum(y_prob[:, ~is_fuel], axis=1) # Sum probabilities of rock classes
    
    gangue_tp = np.sum(true_gangue & pred_gangue)
    gangue_fn = np.sum(true_gangue & ~pred_gangue)
    gangue_fp = np.sum(~true_gangue & pred_gangue)
    gangue_tn = np.sum(~true_gangue & ~pred_gangue)
    
    gangue_recall = gangue_tp / (gangue_tp + gangue_fn) if (gangue_tp + gangue_fn) > 0 else 0
    miss_rate = 1.0 - gangue_recall
    specificity = gangue_tn / (gangue_tn + gangue_fp) if (gangue_tn + gangue_fp) > 0 else 0
    gangue_precision = gangue_tp / (gangue_tp + gangue_fp) if (gangue_tp + gangue_fp) > 0 else 0
    
    gangue_f2 = (5 * gangue_precision * gangue_recall) / (4 * gangue_precision + gangue_recall) if (gangue_precision + gangue_recall) > 0 else 0
    
    def cp_ci(k, n, alpha=0.05):
        if n == 0: return [0, 0]
        lower = stats.beta.ppf(alpha/2, k, n-k+1) if k > 0 else 0
        upper = stats.beta.ppf(1-alpha/2, k+1, n-k) if k < n else 1
        return [lower, upper]

    recall_ci = cp_ci(gangue_tp, gangue_tp + gangue_fn)
    specificity_ci = cp_ci(gangue_tn, gangue_tn + gangue_fp)
    
    # Needs valid true/prob for AUC
    try:
        roc_auc = roc_auc_score(true_gangue, prob_gangue)
        pr_auc = average_precision_score(true_gangue, prob_gangue)
    except Exception:
        roc_auc, pr_auc = 0.0, 0.0
    
    expected_cost = (gangue_fn * cost_ratio + gangue_fp * 1.0) / len(y_true) if len(y_true) > 0 else 0
    
    metrics = {
        "multi_class": {
            "accuracy": acc,
            "accuracy_ci_95": acc_ci.tolist(),
            "macro_f1": macro_f1,
            "macro_f1_ci_95": macro_f1_ci.tolist(),
            "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_acc,
            "cohen_kappa": kappa,
            "mcc": mcc,
            "top2_accuracy": top2_acc,
            "top3_accuracy": top3_acc
        },
        "per_class": per_class_metrics,
        "error_decomposition": {
            "coal_to_coal": int(coal_to_coal),
            "rock_to_rock": int(rock_to_rock),
            "coal_to_rock": int(coal_to_rock),
            "rock_to_coal": int(rock_to_coal)
        },
        "binary_safety": {
            "gangue_recall": gangue_recall,
            "gangue_recall_ci_95": recall_ci,
            "miss_rate": miss_rate,
            "specificity": specificity,
            "specificity_ci_95": specificity_ci,
            "precision": gangue_precision,
            "f2_score": gangue_f2,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        },
        "expected_cost": expected_cost,
        "confusion_matrix_raw": conf_mat.tolist(),
        "confusion_matrix_norm": conf_mat_norm.tolist()
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpectraSense Model")
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "pcvit_expert.pth", help="Path to model weights")
    parser.add_argument("--le-path", type=Path, default=BASE_DIR / "backend" / "assets" / "label_encoder.pkl", help="Path to label encoder")
    parser.add_argument("--x-test", type=Path, required=True, help="Path to X_test.npy")
    parser.add_argument("--y-test", type=Path, required=True, help="Path to y_test.npy")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR, help="Output directory")
    parser.add_argument("--cost-ratio", type=float, default=10.0, help="Miss-to-false-alarm cost ratio")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading label encoder from {args.le_path}")
    import pickle
    with open(args.le_path, 'rb') as f:
        le = pickle.load(f)
    
    logging.info(f"Loading test data from {args.x_test} and {args.y_test}")
    X_test = np.load(args.x_test)
    y_test = np.load(args.y_test)
    
    logging.info(f"Loading model from {args.model_path}")
    model = PCViT(num_classes=len(le.classes_))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True))
    model.eval()
    
    logging.info("Running inference...")
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()
    
    y_pred = np.argmax(probs, axis=1)
    
    logging.info("Computing metrics...")
    metrics = evaluate_model(y_test, y_pred, probs, le, cost_ratio=args.cost_ratio)
    
    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved to {metrics_path}")
    
    preds_path = args.output_dir / "predictions.csv"
    with open(preds_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["scan_id", "true_label_id", "true_label_str", "pred_label_id", "pred_label_str"] + [f"prob_{c}" for c in le.classes_]
        writer.writerow(header)
        for i in range(len(y_test)):
            row = [
                i, 
                y_test[i], 
                le.inverse_transform([y_test[i]])[0],
                y_pred[i], 
                le.inverse_transform([y_pred[i]])[0]
            ] + probs[i].tolist()
            writer.writerow(row)
    logging.info(f"Per-sample predictions saved to {preds_path}")

if __name__ == "__main__":
    main()
