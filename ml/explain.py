"""Model explainability and attribution."""
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # TODO: Implement occlusion importance and integrated gradients on SNV/derivative channels.
    # TODO: Report attribution mass in 1400/1900/2200nm windows vs random windows.
    pass

if __name__ == "__main__":
    main()
