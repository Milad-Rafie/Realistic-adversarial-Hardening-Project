import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import pathlib
BOTNET_ROOT = pathlib.Path(__file__).parent.parent.resolve()   # .../src/botnet
DATA_ROOT    = BOTNET_ROOT.parent / "data"                     # .../src/data

sys.path.insert(0, str(BOTNET_ROOT))

import time
from datetime import datetime

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from fence.neris_attack_tf2 import Neris_attack
from pgd.pgd_attack_art       import PgdRandomRestart
from training.helpers         import get_processing_data

import warnings
warnings.filterwarnings("ignore")


def attack(
    method: str,
    model_path: str,
    samples_path: str,
    labels_path: str,
    distance: float,
    iterations: int,
    only_botnet: bool = True
):
    # 1) Load raw samples & labels
    X = np.load(samples_path)
    y = np.load(labels_path)
    if only_botnet:
        idx = np.where(y == 1)[0]
        X = X[idx]
        y = y[idx]

    # 2) Load model (no compile to avoid legacy‐h5 issues)
    model = load_model(model_path, compile=False)

    # 3) Load scaler & feature‐bounds from data root
    scaler, mins, maxs, mask_idx, eq_min_max = get_processing_data({
        "scaler_path":    str(DATA_ROOT / "neris" / "scaler.pkl"),
        "min_features":   str(DATA_ROOT / "neris" / "minimum.txt"),
        "max_features":   str(DATA_ROOT / "neris" / "maximum.txt"),
        "mask_idx":       str(DATA_ROOT / "neris" / "mutable_idx.npy"),
        "eq_min_max_idx": str(DATA_ROOT / "neris" / "eq_min_max_idx.npy"),
    })

    method = method.lower()
    if method == "pgd":
        y_scalar = y.reshape(-1, 1)
        pgd_gen = PgdRandomRestart(
            model=model,
            eps=distance,
            alpha=1,
            num_iter=iterations,
            restarts=5,
            scaler=scaler,
            mins=mins,
            maxs=maxs,
            mutable_idx=mask_idx,
            eq_min_max_idx=eq_min_max,
        )
        X_adv = pgd_gen.run_attack(X, y_scalar)

    elif method == "neris":
        neris_gen = Neris_attack(
            model_path=str(model_path),
            iterations=iterations,
            distance=distance,
            scaler=scaler,
            mins=mins,
            maxs=maxs,
        )
        adv_list = []
        for i, (x0, y0) in enumerate(zip(X, y)):
            if i % 1000 == 0:
                print(f"[NERIS] attacking sample {i}/{len(X)}")
            x0 = x0.reshape(1, -1)
            adv = neris_gen.run_attack(x0, int(y0))
            adv_list.append(adv)
        X_adv = np.vstack(adv_list)

    else:
        raise ValueError(f"Unknown attack method: {method}")

    # 4) Compute success rate: how often a true‐1 is flipped to 0
    probs = model.predict(X_adv).squeeze()
    preds = (probs >= 0.5).astype(int)
    success_rate = 100 * np.mean(preds == 0)

    return X_adv, success_rate


if __name__ == "__main__":
    MODEL_PATH   = BOTNET_ROOT / "out" / "neris" / "clean_10epochs" / "clean_model.h5"
    SAMPLES_PATH = DATA_ROOT   / "neris" / "testing_samples.npy"
    LABELS_PATH  = DATA_ROOT   / "neris" / "testing_labels.npy"

    for m in ("pgd", "neris"):
        t0 = datetime.now()
        X_adv, sr = attack(
            method=str(m),
            model_path=str(MODEL_PATH),
            samples_path=str(SAMPLES_PATH),
            labels_path=str(LABELS_PATH),
            distance=12,
            iterations=100,
        )
        print(f"\n{m.upper()} → generated {X_adv.shape[0]} adversarials in {datetime.now()-t0}")
        print(f"{m.upper()} success rate: {sr:.2f}%\n")
