"""
Generic manifest-based dataset with feature caching, augmentation, and logmel/mfcc support.

Task 21 - Bug-hardening:
  - Corrupt-file errors are caught per-sample and replaced with a zero tensor so
    training does not crash on a single bad or unreadable clip.
  - Safe cache writes with try/except so permission/disk errors do not halt training.
  - Import path hardening: works when invoked from repo root, src/, or imported as a package.
"""
import os
import sys
import hashlib
from pathlib import Path
from typing import Optional

# Universal import hardening
_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from features import extract_features_from_path, FIXED_FRAMES, N_MFCC
except ImportError:
    from src.features import extract_features_from_path, FIXED_FRAMES, N_MFCC

LABEL_MAP = {"bonafide": 0, "real": 0, "spoof": 1, "fake": 1, "bona-fide": 0}


class VoiceSpoofDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        feature_type: str = "mfcc",
        augment: bool = False,
    ):
        self.manifest_path = manifest_csv
        self.feature_type = feature_type.lower()
        self.augment = augment

        self.df = pd.read_csv(manifest_csv)
        assert {"filepath", "label"}.issubset(self.df.columns), \
            f"Manifest {manifest_csv} must have 'filepath' and 'label' columns"

        self.df["label_idx"] = self.df["label"].str.lower().map(LABEL_MAP)
        if self.df["label_idx"].isna().any():
            bad = self.df[self.df["label_idx"].isna()]["label"].unique()
            raise ValueError(f"Unrecognized label values: {bad}. Expected one of {list(LABEL_MAP.keys())}")

        # Subset for smoke tests if max_samples is specified
        if max_samples is not None and max_samples > 0:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)

        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, filepath: str) -> str:
        """Generate a unique, safe cache filename for an audio file and feature type."""
        abs_path = os.path.abspath(filepath)
        h = hashlib.sha256(f"{abs_path}_{self.feature_type}".encode("utf-8")).hexdigest()[:16]
        stem = Path(filepath).stem[:24]
        return os.path.join(self.cache_dir, f"{stem}_{self.feature_type}_{h}.npy")

    def precompute_cache(self):
        """Precompute all features in this dataset to .npy files with corrupt file handling."""
        if not self.cache_dir:
            raise ValueError("Cannot precompute cache: cache_dir was not specified.")
        print(f"Precomputing {self.feature_type} features ({len(self.df)} clips) into {self.cache_dir}...")
        for filepath in tqdm(self.df["filepath"], desc=f"Caching {self.feature_type}"):
            cache_file = self._get_cache_path(filepath)
            if not os.path.isfile(cache_file):
                try:
                    feat = extract_features_from_path(filepath, feature_type=self.feature_type, augment=False)
                    np.save(cache_file, feat)
                except Exception as e:
                    print(f"[Warning] Failed caching {filepath}: {e}. Storing zero features.")
                    feat = np.zeros((3, N_MFCC, FIXED_FRAMES), dtype=np.float32)
                    try:
                        np.save(cache_file, feat)
                    except Exception:
                        pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]

        feat = None
        try:
            # If data augmentation is active, augment waveform on the fly (bypass static cache)
            if self.augment:
                feat = extract_features_from_path(filepath, feature_type=self.feature_type, augment=True)
            elif self.cache_dir:
                cache_file = self._get_cache_path(filepath)
                if os.path.isfile(cache_file):
                    try:
                        feat = np.load(cache_file)
                    except Exception:
                        feat = extract_features_from_path(filepath, feature_type=self.feature_type, augment=False)
                        try:
                            np.save(cache_file, feat)
                        except Exception:
                            pass
                else:
                    feat = extract_features_from_path(filepath, feature_type=self.feature_type, augment=False)
                    try:
                        np.save(cache_file, feat)
                    except Exception:
                        pass
            else:
                feat = extract_features_from_path(filepath, feature_type=self.feature_type, augment=False)
        except Exception as e:
            # Task 21: Corrupt-file handling - return zero feature array of proper shape
            feat = np.zeros((3, N_MFCC, FIXED_FRAMES), dtype=np.float32)

        if feat is None or feat.shape != (3, N_MFCC, FIXED_FRAMES):
            feat = np.zeros((3, N_MFCC, FIXED_FRAMES), dtype=np.float32)

        label = int(row["label_idx"])
        return torch.from_numpy(feat).float(), torch.tensor(label, dtype=torch.long)
