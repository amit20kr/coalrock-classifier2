"""Temperature scaling and calibration metrics."""
import argparse
import logging
from pathlib import Path
from ml.config import ARTIFACTS_DIR, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-data", type=Path, default=DATA_DIR / "val.npz")
    args = parser.parse_args()
    # TODO: Implement temperature scaling on validation set
    # TODO: Report ECE (15 bins), MCE, Brier, NLL
    # TODO: Reliability diagrams before/after
    pass

if __name__ == "__main__":
    main()
