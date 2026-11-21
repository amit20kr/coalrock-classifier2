"""Configuration constants and paths for the SpectraSense project."""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "backend" / "models"

# Project setup
RANDOM_SEED = 42
NUM_CLASSES = 24

# Classes
import json
with open(BASE_DIR / "backend/assets/class_list.json", "r", encoding="utf-8") as f:
    ALL_CLASSES = json.load(f)
FUEL_CLASSES = [c for c in ALL_CLASSES if "coal" in c.lower()]

GANGUE_CLASSES = [c for c in ALL_CLASSES if c not in FUEL_CLASSES]

# Preprocessing params
SNV_PARAMS = {}

# Model inputs
INPUT_SHAPE = (3, 1500) # SNV, 1st derivative, 2nd derivative of NIR spectra
