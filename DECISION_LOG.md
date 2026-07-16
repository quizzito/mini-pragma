# Decision Log

One entry per non-trivial choice. Timestamp, decision, reasoning, alternatives considered.
This file is what turns into the essay later — keep entries short and honest.

---

### 2026-07-15 — Project framing
**Decision:** Build "mini-PRAGMA" — a scaled-down (1–10M param) reimplementation of the
key-value-time tokenization + two-branch encoder architecture from the PRAGMA paper
(arXiv:2604.08649), trained on fully synthetic banking event data.

**Kill criterion set:** If the foundation model doesn't beat XGBoost baselines
(accounting for confidence intervals) on any of 3 downstream tasks after M6, stop
and write it up as a negative/instructive result rather than scaling further.

**Alternatives considered:** Using a real public dataset (e.g., a Kaggle fraud
dataset) instead of synthetic data — rejected for v1 because it lacks the
multi-source event richness (transactions + app events + comms + trading) that
makes PRAGMA's architecture interesting to reimplement; synthetic data lets us
control this directly and is faster to get right.

---

### 2026-07-15 — Compute strategy
**Decision:** Kaggle Notebooks (30 GPU-hrs/week, P100, background execution) as
primary GPU resource; Colab free tier (dynamic 15–30 hrs/week, T4) as overflow;
Intel Mac (CPU only, no MPS) for all dev/debugging/unit tests before any GPU run.

**Reasoning:** Kaggle's background execution matters most for a solo, part-time
schedule — training can continue after closing the laptop, which Colab's free
tier doesn't reliably support.

---

<!-- Add new entries below as you go. Suggested next entries:
- Why this synthetic data schema / label-generation rules (M2)
- Why this masking ratio / vocab size (M3)
- Why this model width/depth for the tiny variant (M5)
- Any ablation you dropped and why (M7)
-->
