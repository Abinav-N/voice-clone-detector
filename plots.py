"""
Root entrypoint for plot generator:
python plots.py --scores results/test.csv --history checkpoints/history.csv
"""
import sys
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plots import main

if __name__ == "__main__":
    main()
