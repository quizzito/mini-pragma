"""
Environment parity check for mini-pragma.

Run this FIRST on your Mac (CPU) and again on Colab/Kaggle (GPU).
It must succeed in both places with the same package versions before
you write a single line of model code. This is the M1 exit criterion.

Usage:
    python scripts/env_check.py
"""
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost


def main() -> None:
    print("=" * 60)
    print("mini-pragma environment check")
    print("=" * 60)
    print(f"Python        : {sys.version.split()[0]}")
    print(f"Platform      : {platform.platform()}")
    print(f"NumPy         : {np.__version__}")
    print(f"Pandas        : {pd.__version__}")
    print(f"scikit-learn  : {sklearn.__version__}")
    print(f"XGBoost       : {xgboost.__version__}")
    print(f"PyTorch       : {torch.__version__}")

    if torch.cuda.is_available():
        device = "cuda"
        print(f"Accelerator   : CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Accelerator   : MPS (Apple Silicon) — note: you're on Intel Mac, "
              "this should NOT trigger; if it does, check your torch install.")
    else:
        device = "cpu"
        print("Accelerator   : none (CPU only) — expected on your Intel Mac.")

    # Trivial tensor op to prove the installed torch build actually works,
    # not just imports cleanly.
    x = torch.randn(4, 4, device=device)
    y = torch.randn(4, 4, device=device)
    z = (x @ y).sum().item()
    print(f"Sanity matmul : sum={z:.4f} (any finite float is fine — just proving compute runs)")

    print("=" * 60)
    print("PASS: environment is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
