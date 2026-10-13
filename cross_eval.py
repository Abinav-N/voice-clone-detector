"""
Root entrypoint for cross-dataset evaluation:
python cross_eval.py --asvspoof manifests/test.csv --itw manifests/itw_test.csv
"""
import sys
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cross_eval import main

if __name__ == "__main__":
    main()
