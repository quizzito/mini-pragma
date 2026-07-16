# mini-PRAGMA

A scaled-down, from-scratch reimplementation of the architecture in
*PRAGMA: Revolut Foundation Model* (Ostroukhov et al., arXiv:2604.08649),
trained on fully synthetic banking event data, to learn — hands-on — how
tabular/event-sequence foundation models are built and evaluated.

This is a personal learning project. All data is synthetic. No proprietary
or real financial data is used. See `DECISION_LOG.md` for the reasoning
behind every non-trivial choice made along the way.

## Status
- [x] M0 — Framing & business case
- [x] M1 — Environment & repo scaffold (this commit)
- [ ] M2 — Synthetic data generation
- [ ] M3 — Tokenizer & pipeline
- [ ] M4 — Baselines (XGBoost)
- [ ] M5 — Foundation model pretraining
- [ ] M6 — Downstream adaptation & evaluation
- [ ] M7 — Ablations
- [ ] M8 — Business case quantification
- [ ] M9 — Write-up

## Repo structure
```
mini-pragma/
├── data_gen/         # synthetic event/profile data generation (M2)
├── tokenizer/        # key-value-time tokenization scheme (M3)
├── model/            # event/profile/history encoders (M5)
├── baselines/        # XGBoost task-specific baselines (M4)
├── probes/           # embedding probes + fine-tuning (M6)
├── results/          # metrics tables, plots (M6-M8)
├── scripts/          # env checks, training entrypoints
├── notebooks/        # Colab/Kaggle setup notes
├── tests/            # pytest unit tests (run locally before any GPU time)
├── DECISION_LOG.md
└── requirements.txt
```

## Setup — local (Intel Mac, Python 3.11)

```bash
cd mini-pragma
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/env_check.py
pytest tests/ -v
```

Both commands above must pass before moving to M2. Expected output:
`Accelerator : none (CPU only)` — that's correct on this machine.

## Setup — Colab / Kaggle (GPU)

See `notebooks/colab_kaggle_setup.md` for exact cells to paste. Same
`requirements.txt`, same `scripts/env_check.py` — only the accelerator
line in the output should differ (CUDA instead of CPU).

## Reference
Ostroukhov, M. et al. "PRAGMA: Revolut Foundation Model." arXiv:2604.08649, 2026.
