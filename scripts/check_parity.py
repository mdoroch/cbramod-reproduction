"""Check that our data and models are numerically identical to the official code.

Needs: bash scripts/get_official_code.sh, bash scripts/download_weights.sh, python prepare_data.py
Run from the repo root: python scripts/check_parity.py
"""
import glob
import importlib.util
import sys
import types

import numpy as np
import scipy.io
import scipy.signal
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "third_party/CBraMod")
from models.model_for_shu import Model as OfficialCBraMod  # noqa: E402

from src.cbramod import CBraModClassifier  # noqa: E402
from src.data import TEST_SUBJECTS, TRAIN_SUBJECTS, VAL_SUBJECTS, make_dataset  # noqa: E402
from src.simpleconv import EEGSimpleConv  # noqa: E402


def load_official_simpleconv():
    path = "third_party/EEGSimpleConv/models/EEGSimpleConv.py"  # its package name clashes with CBraMod's `models`
    spec = importlib.util.spec_from_file_location("official_simpleconv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EEGSimpleConv


def check_data():
    """Official preprocessing_shu.py: sorted files, [:75] / [75:100] / [100:], resample to 800, label - 1, x / 100."""
    files = sorted(glob.glob("data/mat/*.mat"))
    subject = np.load("data/processed/subject.npy")
    for official_files, subjects in [(files[:75], TRAIN_SUBJECTS), (files[75:100], VAL_SUBJECTS), (files[100:], TEST_SUBJECTS)]:
        assert {int(f.split("sub-")[1][:3]) for f in official_files} == set(subjects)
    print(f"split: train {np.isin(subject, TRAIN_SUBJECTS).sum()}, val {np.isin(subject, VAL_SUBJECTS).sum()}, "
          f"test {np.isin(subject, TEST_SUBJECTS).sum()} trials")

    ours = make_dataset([21], ea=False)  # sub-021, sessions 1-5 in file order
    mats = [scipy.io.loadmat(f) for f in files if "sub-021" in f]
    x = np.concatenate([scipy.signal.resample(m["data"], 800, axis=2) for m in mats]) / 100
    y = np.concatenate([m["labels"][0] - 1 for m in mats])
    assert np.array_equal(ours.tensors[1].numpy(), y)
    diff = np.abs(ours.tensors[0].numpy() - x).max()
    print(f"data (sub-021): max |ours - official| = {diff:.2e}")
    assert diff < 1e-5
    return ours.tensors[0][:16]


def check_cbramod(x):
    ours = CBraModClassifier(32, 4, "large", 0.1, weights="weights/pretrained_weights.pth").eval()  # strict load
    params = types.SimpleNamespace(use_pretrained_weights=False, classifier="all_patch_reps", dropout=0.1, cuda=0)
    official = OfficialCBraMod(params).eval()
    official.load_state_dict(ours.state_dict())  # strict: same parameter names and shapes
    with torch.no_grad():
        diff = (ours(x) - official(x.reshape(-1, 32, 4, 200))).abs().max().item()
    print(f"CBraMod: pretrained weights load strictly, max |ours - official| logit = {diff:.2e}")
    assert diff < 1e-5


def check_simpleconv(x):
    ours = EEGSimpleConv(n_channels=32, sfreq=200, fm=109, n_convs=4, resampling=70, kernel_size=8).eval()
    official = load_official_simpleconv()(109, 4, 70, 8, n_chan=32, n_classes=1, sfreq=200).eval()
    official.load_state_dict(ours.state_dict())
    with torch.no_grad():
        diff = (ours(x) - official(x).squeeze(-1)).abs().max().item()
    n_params = sum(p.numel() for p in ours.parameters())
    print(f"EEGSimpleConv: {n_params / 1e6:.2f}M params, max |ours - official| logit = {diff:.2e}")
    assert diff < 1e-5


if __name__ == "__main__":
    x = check_data()
    check_cbramod(x)
    check_simpleconv(x)
    print("all parity checks passed")
