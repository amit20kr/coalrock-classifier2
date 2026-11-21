"""Synthetic batch construction and simulation."""
import argparse
import logging
import json
import numpy as np
import torch
import pickle
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from backend.model import PCViT
from backend.pipeline import FUEL_CLASSES
from ml.config import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_gangue_mass(X, y, le, model, batch_size=128):
    gangue_indices = [i for i, c in enumerate(le.classes_) if c not in FUEL_CLASSES]
    tensor = torch.FloatTensor(X)
    masses = []
    
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            chunk = tensor[start:start+batch_size]
            logits = model(chunk)
            probs = torch.softmax(logits, dim=1)
            chunk_mass = probs[:, gangue_indices].sum(dim=1).numpy()
            masses.append(chunk_mass)
            
    return np.concatenate(masses)

def simulate_batches(fuel_masses, gangue_masses, contamination, batch_size, n_batches, threshold, k, seed=42):
    rng = np.random.default_rng(seed)
    detects = 0
    
    n_gangue = int(np.round(batch_size * contamination))
    n_fuel = batch_size - n_gangue
    
    for _ in range(n_batches):
        batch = []
        if n_fuel > 0:
            batch.extend(rng.choice(fuel_masses, n_fuel, replace=True))
        if n_gangue > 0:
            batch.extend(rng.choice(gangue_masses, n_gangue, replace=True))
            
        batch = np.array(batch)
        flagged = (batch > threshold).sum()
        if flagged >= k:
            detects += 1
            
    return detects / n_batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"G:\My Drive\coalrock_project2\processed_data"))
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "pcvit_expert.pth")
    parser.add_argument("--le-path", type=Path, default=BASE_DIR / "backend" / "assets" / "label_encoder.pkl")
    args = parser.parse_args()

    with open(args.le_path, "rb") as f:
        le = pickle.load(f)
        
    model = PCViT(num_classes=len(le.classes_))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Load validation and test data
    X_val = np.load(args.data_dir / "X_val.npy")
    y_val = np.load(args.data_dir / "y_val.npy")
    X_test = np.load(args.data_dir / "X_test.npy")
    y_test = np.load(args.data_dir / "y_test.npy")

    fuel_idx = [i for i, c in enumerate(le.classes_) if c in FUEL_CLASSES]
    
    val_fuel_mask = np.isin(y_val, fuel_idx)
    val_fuel_masses = get_gangue_mass(X_val[val_fuel_mask], y_val[val_fuel_mask], le, model)
    val_gangue_masses = get_gangue_mass(X_val[~val_fuel_mask], y_val[~val_fuel_mask], le, model)

    test_fuel_mask = np.isin(y_test, fuel_idx)
    test_fuel_masses = get_gangue_mass(X_test[test_fuel_mask], y_test[test_fuel_mask], le, model)
    test_gangue_masses = get_gangue_mass(X_test[~test_fuel_mask], y_test[~test_fuel_mask], le, model)

    # 1. Sweep on validation
    thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]
    ks = [1, 2, 3, 4, 5]
    best_cost = float('inf')
    best_params = None
    
    # Cost ratio: 1 False Alarm (reject clean coal) = 1, 1 Miss (accept 5% gangue) = 10
    cost_miss = 10
    cost_fa = 1
    
    logging.info("Sweeping threshold and k on validation data...")
    for t in thresholds:
        for k in ks:
            fa_rate = simulate_batches(val_fuel_masses, val_gangue_masses, 0.0, 100, 1000, t, k)
            hit_rate = simulate_batches(val_fuel_masses, val_gangue_masses, 0.05, 100, 1000, t, k)
            miss_rate = 1.0 - hit_rate
            
            cost = (fa_rate * cost_fa) + (miss_rate * cost_miss)
            if cost < best_cost:
                best_cost = cost
                best_params = (t, k)
                
    opt_t, opt_k = best_params
    logging.info(f"Optimal parameters: Threshold={opt_t}, k={opt_k} (Val Cost: {best_cost:.3f})")

    # 2. Simulate on Test Data with optimal params
    batch_sizes = [20, 100, 500]
    contaminations = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
    
    results = {
        "optimal_threshold": opt_t,
        "optimal_k": opt_k,
        "simulation": {}
    }
    
    logging.info("Simulating on test data...")
    for b in batch_sizes:
        results["simulation"][b] = {}
        for c in contaminations:
            det_rate = simulate_batches(test_fuel_masses, test_gangue_masses, c, b, 2000, opt_t, opt_k)
            results["simulation"][b][str(c)] = det_rate
            logging.info(f"Batch={b}, Contam={c*100}%, Detect Rate={det_rate*100:.1f}%")

    with open(ARTIFACTS_DIR / "batch_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info("Saved batch_metrics.json")

if __name__ == "__main__":
    main()
