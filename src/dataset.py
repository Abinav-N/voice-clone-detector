"""
Generic manifest-based dataset.

Expected CSV format (no dataset-specific assumptions):
    filepath,label
    /data/for/train/real/001.wav,bonafide
    /data/for/train/fake/001.wav,spoof

Use src/build_manifest.py to generate this from ASVspoof protocols, a
real/fake folder layout, or any other source.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset

from features import extract_features_from_path

LABEL_MAP = {"bonafide": 0, "real": 0, "spoof": 1, "fake": 1}


class VoiceSpoofDataset(Dataset):
    def __init__(self, manifest_csv: str):
        self.df = pd.read_csv(manifest_csv)
        assert {"filepath", "label"}.issubset(self.df.columns), \
            "manifest must have 'filepath' and 'label' columns"
        self.df["label_idx"] = self.df["label"].str.lower().map(LABEL_MAP)
        if self.df["label_idx"].isna().any():
            bad = self.df[self.df["label_idx"].isna()]["label"].unique()
            raise ValueError(f"Unrecognized label values: {bad}. "
                              f"Expected one of {list(LABEL_MAP.keys())}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = extract_features_from_path(row["filepath"])   # (3, n_mfcc, T)
        label = int(row["label_idx"])                         # 0 = real, 1 = spoof
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)
