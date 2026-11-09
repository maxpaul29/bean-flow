import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from ucimlrepo import fetch_ucirepo

SEED = 42
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "dry_bean.csv"


class DryBeanDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def set_seed(seed=SEED):
    # Seed every random number generator that touches the experiment
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_raw_dry_bean():
    # Download Dry Bean from UCI once and cache it as CSV (data/*.csv is git-ignored)
    if CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH)

    dry_bean = fetch_ucirepo(id=602)
    df = dry_bean.data.features.copy()
    df["Class"] = dry_bean.data.targets["Class"].values
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def load_dry_bean_splits(train_fraction=0.8, split_seed=SEED):
    # Return train_set and test_set (torch Subsets of one DryBeanDataset, X float32 with 16 features, y long with class ids 0..6)
    df = _load_raw_dry_bean()
    X = torch.tensor(df.drop(columns=["Class"]).values, dtype=torch.float32)
    # class names sorted alphabetically -> ids 0..6
    y = torch.tensor(pd.Categorical(df["Class"]).codes, dtype=torch.long)

    # fixed split seed so every run uses the same split
    # no validation set, we tune nothing: architecture, lr and clip threshold are fixed by the experiment design
    dataset = DryBeanDataset(X, y)
    train_set, test_set = random_split(
        dataset, [train_fraction, 1 - train_fraction], generator=torch.Generator().manual_seed(split_seed))

    # standardize because raw scales differ by orders of magnitude (Area ~ 1e4, Eccentricity ~ 1)
    # mean and std come from the train part only, so no test information leaks into training
    train_X = X[train_set.indices]
    dataset.data = (X - train_X.mean(dim=0)) / train_X.std(dim=0)

    return train_set, test_set

def make_loader(dataset, batch_size=32, seed=SEED):
    # Shuffled DataLoader with its own seeded generator, so batch order is reproducible
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
