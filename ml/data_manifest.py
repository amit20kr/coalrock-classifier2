"""Script to compute and save data manifest."""
import argparse
import json
import logging
from pathlib import Path

from ml.config import ARTIFACTS_DIR, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def compute_manifest(data_dir: Path, output_path: Path):
    """Compute and save the data manifest."""
    # TODO: Implement content hashes and file counts for raw, deduped, train, val, test arrays
    manifest = {
        "raw_counts": 0,
        "train_counts": 0,
        "val_counts": 0,
        "test_counts": 0,
        "hashes": {}
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    logging.info(f"Manifest written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute data manifest")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Path to data directory")
    parser.add_argument("--output", type=Path, default=ARTIFACTS_DIR / "manifest.json", help="Output path for manifest.json")
    args = parser.parse_args()
    
    compute_manifest(args.data_dir, args.output)
