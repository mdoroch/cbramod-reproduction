import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = "data/processed"
# Paper protocol: subjects 1-15 / 16-20 / 21-25
TRAIN_SUBJECTS = list(range(1, 16))
VAL_SUBJECTS = list(range(16, 21))
TEST_SUBJECTS = list(range(21, 26))


def train_subjects(n, seed):
    """All 15 train subjects, or a seeded random subset of n of them (low-data experiment)."""
    if n == len(TRAIN_SUBJECTS):
        return TRAIN_SUBJECTS
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(TRAIN_SUBJECTS, size=n, replace=False).tolist())


def euclidean_alignment(x, subject):
    """Euclidean Alignment (He & Wu, 2019): per subject, x <- R^(-1/2) x with R the mean trial
    covariance X X^T (labels not used). Each subject's mean covariance becomes the identity."""
    out = np.empty_like(x)
    for s in np.unique(subject):
        xs = x[subject == s].astype(np.float64)  # outliers up to ~2800: square in float64
        R = np.einsum("nct,ndt->cd", xs, xs) / len(xs)
        w, V = np.linalg.eigh(R)
        out[subject == s] = np.einsum("cd,ndt->nct", (V / np.sqrt(w)) @ V.T, xs)
    return out


def make_dataset(subjects, ea):
    X = np.load(f"{DATA_DIR}/X.npy", mmap_mode="r")
    y = np.load(f"{DATA_DIR}/y.npy")
    subject = np.load(f"{DATA_DIR}/subject.npy")
    idx = np.flatnonzero(np.isin(subject, subjects))
    x = X[idx] / 100  # units of 100 µV, as in CBraMod pretraining
    if ea:
        x = euclidean_alignment(x, subject[idx])
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y[idx]).float(), torch.from_numpy(subject[idx]))


def make_loaders(batch_size, train_subjects, pin_memory, ea):
    datasets = {
        "train": make_dataset(train_subjects, ea),
        "val": make_dataset(VAL_SUBJECTS, ea),
        "test": make_dataset(TEST_SUBJECTS, ea),
    }
    return {
        name: DataLoader(ds, batch_size=batch_size, shuffle=name == "train", pin_memory=pin_memory)
        for name, ds in datasets.items()
    }
