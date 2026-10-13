"""
Generic manifest-based dataset with feature caching, augmentation, and logmel/mfcc support.

Expected CSV format:
    filepath,label[,attack_id]
"""
import os
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from features import extract_features_from_path

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
        """Precompute all features in this dataset to .npy files."""
        if not self.cache_dir:
            raise ValueError("Cannot precompute cache: cache_dir was not specified.")
        print(f"Precomputing {self.feature_type} features ({len(self.df)} clips) into {self.cache_dir}...")
        for filepath in tqdm(self.df["filepath"], desc=f"Caching {self.feature_type}"):
            cache_file = self._get_cache_path(filepath)
            if not os.path.isfile(cache_file):
                feat = extract_features_from_path(filepath, feature_type=self.feature_type, augment=False)
                np.save(cache_file, feat)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]

        # If data augmentation is active, augment waveform on the fly (bypass static cache)
        if self.augment:
            feat = extract_features_from_path(filepath, feature_type=self.feature_type, augment=True)
        elif self.cache_dir:
            cache_file = self._get_cache_path(filepath)
            if os.path.isfile(cache_file):
                feat = np.load(cache_file)
            else:
                feat = extract_features_from_path(filepath, feature_type=self.feature_type, augment=False)
                np.save(cache_file, feat)
        else:
            feat = extract_features_from_path(filepath, feature_type=self.feature_type, augment=False)

        label = int(row["label_idx"])
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)
