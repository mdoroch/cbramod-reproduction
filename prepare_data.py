"""Convert SHU-MI .mat files into arrays at 200 Hz.

Follows the official CBraMod preprocessing (preprocessing/preprocessing_shu.py):
FFT resampling 1000 -> 800 samples per trial, labels 1/2 -> 0/1. No filtering, no clipping.
Output: data/processed/{X,y,subject,session}.npy, X in µV with shape (11988, 32, 800).
"""
import glob
import os

import numpy as np
import scipy.io
import scipy.signal

MAT_DIR = "data/mat"
OUT_DIR = "data/processed"


def main():
    X, y, subject, session = [], [], [], []
    for path in sorted(glob.glob(f"{MAT_DIR}/*.mat")):
        name = os.path.basename(path)  # sub-001_ses-01_task_motorimagery_eeg.mat
        mat = scipy.io.loadmat(path)
        labels = mat["labels"][0] - 1  # 1 = left, 2 = right -> 0 / 1
        X.append(scipy.signal.resample(mat["data"], 800, axis=2).astype(np.float32))  # 250 -> 200 Hz
        y.append(labels)
        subject.append(np.full(len(labels), int(name[4:7])))
        session.append(np.full(len(labels), int(name[12:14])))

    os.makedirs(OUT_DIR, exist_ok=True)
    for key, arrays in [("X", X), ("y", y), ("subject", subject), ("session", session)]:
        np.save(f"{OUT_DIR}/{key}.npy", np.concatenate(arrays))
    print(f"saved {sum(len(a) for a in y)} trials to {OUT_DIR}")


if __name__ == "__main__":
    main()
