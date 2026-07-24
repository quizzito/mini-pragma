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

### 2026-07-16 — M2 synthetic data generation complete
**Decision:** Finalized synthetic data generator for 5 event types (card_payment,
topup, app_event, communication, trading) and 3 downstream task labels
(credit_default, fraud, engagement), each driven by a made-up behavioral rule
plus random noise.

**Validated label rates (at n=2000 users):** credit_default 6.0%, fraud 5.9%,
engagement 33.8% — all in realistic ranges. Confirmed via `validate_data.py`
that each label correlates with its intended driver (e.g. credit_default
correlates strongly with spend-to-balance ratio: 0.13 for non-defaulters vs.
0.75 for defaulters) and shows no spurious correlation with user_id.

**Known limitation:** balance, tenure, and event-type mix are close to uniform
distributions (an artifact of using `random.uniform`/`random.choice` with equal
weights), whereas real banking data is typically long-tailed/skewed (per the
paper's own §2.1.3 discussion). This makes our synthetic world somewhat easier
than reality. Not fixing this for v1 — noting it as a scope simplification to
flag in the final write-up, and a candidate revisit if downstream results look
suspiciously clean.

**Validation note:** first checked label rates at n=100 users and got noisy
readings (7% fraud, 41% engagement); re-ran at n=2000 to get a stable estimate.
Lesson: always sanity-check label rates at a large enough sample before trusting
them.

<!-- Add new entries below as you go. Suggested next entries:
- Why this synthetic data schema / label-generation rules (M2)
- Why this masking ratio / vocab size (M3)
- Why this model width/depth for the tiny variant (M5)
- Any ablation you dropped and why (M7)
-->

### 2026-07-16 — M3 core tokenizer complete
**Decision:** Built the full key-value-time tokenization pipeline: shared key
vocabulary (20 fields), per-field categorical vocabularies (13 fields),
percentile-bucketed numerical encoding (amount, fee), log-compressed elapsed
time, and sine/cosine cyclical time (hour, day-of-week). Verified end-to-end
on two real consecutive events from the generated dataset.

**Known limitation:** `fee` boundaries are degenerate (all 0.0) because our
data generator never varies fees — every event's fee bucket will be identical,
carrying no signal. Deferred fixing this (would require regenerating data in
data_gen/generate_events.py); acceptable for now since it doesn't break
anything, just makes `fee` an uninformative field for the model.

**Bugs caught during testing:** tokenizer initially crashed on `user_id`
(a row identifier, not a semantic field — now explicitly skipped) and on NaN
values in type-specific fields (e.g. `view` is NaN for a card_payment row,
since only app_events have a view — now explicitly skipped via `pd.isna()`).
Both were caught immediately as loud crashes rather than silently corrupting
tokenization, which is the better failure mode.

**Still TODO for M3:** text-field tokenization (description, currently only
5 fixed categories so treated as categorical — real free text would need
subword tokenization, deferred as a stretch goal); profile-state tokenization
(reusing the same key/categorical functions, not yet wired up); batching this
across a full user history rather than one event at a time.

### 2026-07-16 — M3 complete (core scope)
**Decision:** Added profile-state tokenization, reusing the same key/categorical/
numerical functions built for events. Verified on real profile data — correctly
skips user_id and the 3 label fields, buckets tenure_months and balance sensibly.

**Scope closed for M3:** declaring the tokenizer done for core cases (all 5
event types + profile state, categorical + numerical + time encoding all
working end-to-end on real data).

**Deferred as explicit stretch goals, not blockers:**
- Free-text subword tokenization (description is currently just 5 fixed
  categories in our synthetic data, so categorical treatment is sufficient
  for now; a real deployment would need this per §2.2 of the paper)
- "Life-long event" time-since-milestone encoding for profile state (§2.1.2)
- Whole-user-history tokenization (profile + full event list -> tensors) —
  deferred to M5, since it's really part of building the model's input
  pipeline rather than the tokenizer itself


### 2026-07-16 — M4 complete: XGBoost baselines
**Decision:** Built hand-engineered features (total spend, total topup, event
counts by type, total events) via pandas groupby, joined with each of the 3
labels, stratified 80/20 train/test split, trained XGBoost per task.

**Baseline results (reference row for M6):**
| Task | ROC-AUC | PR-AUC | Base rate |
|---|---|---|---|
| credit_default | 0.755 | 0.226 | 6.0% |
| fraud | 0.618 | 0.125 | 5.9% |
| engagement | 0.583 | 0.390 | 33.8% |

**Interpretation:** credit_default shows the strongest baseline signal
(~3.8x random-floor PR-AUC), consistent with the strong spend-to-balance
correlation confirmed in M2 data validation. Fraud and engagement are weaker
(~2.1x and ~1.15x random floor respectively) — engagement being the hardest
task mirrors the PRAGMA paper's own finding that Communication Engagement is
their most sample-starved, hardest-to-model task (§3.2), despite our data
being entirely synthetic and independently constructed.

**Kill criterion reminder (from M0):** these numbers are now the bar the
foundation model (M5-M6) needs to clear on at least one task to avoid a
negative/stop result.

### 2026-07-17 — M5 scope decision
**Decision:** Building mini-PRAGMA v1 as a simplified two-stage architecture:
one Event Encoder + one History Encoder, trained with masked modeling.
Profile state is folded in as a special "profile" item prepended to each
user's sequence, rather than a separate dedicated branch.

**Explicitly deferred for v1 (all are real techniques from the paper that
solve problems at its scale, which don't yet apply at ours):**
- **Separate Profile Encoder branch** (paper §2.3.2) — deferred in favor of
  treating profile state as one more sequence item, handled by the same
  Event Encoder. Revisiting this later would let us reproduce the paper's
  own profile-state ablation (§3.4.2) ourselves, comparing "profile folded
  into event sequence" vs. "dedicated profile branch."
- **RoPE for time encoding** (paper §2.3.2-2.3.4) — deferred in favor of
  feeding our already-built log-compressed elapsed-time value as a plain
  numeric input, rather than rotating token vectors by time. Simpler to
  implement, less theoretically elegant, but sufficient at our scale.
- **LoRA fine-tuning** (paper §3.1.2) — deferred since our models are only
  a few million parameters; full fine-tuning of the whole model in M6 will
  already be fast and cheap. LoRA solves a cost problem that appears at
  billion-parameter scale, which we don't have.
- **Sequence packing / dynamic batching** (paper §2.4) — using simple
  padding instead, since our dataset (thousands, not millions, of users)
  is small enough that padding waste doesn't meaningfully affect training
  time or cost.

**Rationale:** each deferred piece is a real, correct solution to a real
problem at the paper's scale (26M users, up to 1B parameters) — but at our
scale (2000 users, low-millions of parameters), the problem it solves barely
exists yet. Matching architectural complexity to actual problem size, not
skipping for convenience.

### 2026-07-17 — M5 architecture: flattened-first, hierarchical later
**Decision:** Building the flattened single-Transformer version first (paper's
own acknowledged "viable baseline," §1), as a working checkpoint before
attempting the full two-stage Event Encoder + History Encoder hierarchy
(Figure 4). Rationale: de-risks the training loop, masking objective, and
checkpointing against a simpler architecture first; gives a baseline
embedding to compare the hierarchical version against later. Hierarchical
version is a planned follow-on milestone, not abandoned scope — since we're
training on free Colab GPU time, the constraint is our own time, not cost.

### 2026-07-17 — M5 step 3: flatten + pad, max_length correction
**Decision:** Built `flatten_and_pad`, combining all of a user's profile +
event key-value tokens into one flat sequence with a prepended [USR] token,
padded/truncated to a fixed length. Added [PAD] and [USR] special tokens to
the key vocabulary (now 22 keys, was 20) and a PAD_VALUE_ID=-1 for padded
value positions.

**Bug caught before it mattered:** initially guessed max_length=50, which
silently truncated most users' histories (measured flattened lengths across
50 users: min=26, max=243, avg=138.8 — since each history "item" like a
card_payment expands into multiple key-value tokens). Corrected to
max_length=250 after measuring real data instead of guessing. Lesson:
always measure real flattened/tokenized lengths before picking a fixed
sequence length -- the same principle behind the paper's own §2.4 truncation
analysis, just at a much smaller scale.

**Known gap, deferred:** flatten_and_pad currently drops all time-encoding
info (elapsed_time, hour/day sin-cos) computed earlier in tokenize_event/
tokenize_user_history. Need to decide how to reintroduce this once the base
sequence + model pipeline is confirmed working end-to-end.

### 2026-07-17 — M5 step 4: PyTorch Dataset + DataLoader working
**Decision:** Built `UserHistoryDataset`, wrapping our tokenizer functions
(tokenize_user_history, flatten_and_pad) so PyTorch can index into any user
and get back key_ids/value_ids tensors of fixed length. Verified DataLoader
batching works correctly: batch shape [8, 250] for batch_size=8.

**Note:** boundaries are currently recomputed from a fresh sample each time
this script runs (in the __main__ block) rather than saved/loaded from a
fixed file. Fine for now since we're not deep into training yet, but this
should be saved once (e.g. as JSON) before real pretraining runs, so
train/eval use identical boundaries rather than potentially-different
random samples.

### 2026-07-17 — M5 step 5: boundaries saved to disk, dataset refactored
**Decision:** Added compute_all_boundaries/save_boundaries/load_boundaries
to the tokenizer, computing all 5 numerical field boundaries once and
saving to tokenizer/boundaries.json. Refactored UserHistoryDataset to load
from this file instead of recomputing boundaries ad-hoc. Prevents a subtle
train/eval mismatch bug where different random samples could produce
slightly different bucket cutoffs.

### 2026-07-17 — M5 step 6: token embedding layer + correct value vocab size
**Decision:** Built TokenEmbedding (key embedding + value embedding, summed
per equation 1 in the paper §2.3.1). Added get_max_value_id() to the
tokenizer to correctly size the value embedding table from real data rather
than guessing -- confirmed max value_id = 9 (10 numerical buckets, ids 0-9;
no categorical field exceeds this). Also handles PAD_VALUE_ID=-1 by shifting
to index 0 before embedding lookup, since nn.Embedding requires non-negative
indices.

### 2026-07-17 — M5 steps 7-8: Transformer encoder + USR pooling
**Decision:** Added MiniPragmaEncoder (wraps nn.TransformerEncoderLayer,
batch_first=True, with src_key_padding_mask support to ignore [PAD]
positions) and pool_usr_token (extracts position 0 as the summary user
embedding, matching z_h,0 in the paper's Figure 4). All shape tests pass:
[1,5,32] through the encoder, pooled down to [1,32] -- one vector per user.

**All individual model pieces now built and verified in isolation:**
TokenEmbedding, PositionalEmbedding, MiniPragmaEncoder, pool_usr_token.
Next session: assemble into one MiniPragma class, then build the masked
modeling training objective.

### 2026-07-17 — M5 step 9: full MiniPragma model assembled, first end-to-end run
**Decision:** Combined TokenEmbedding + PositionalEmbedding + MiniPragmaEncoder
+ pool_usr_token into one MiniPragma class. First successful end-to-end run
on real batched data: encoder_output=[8,250,32], user_embedding=[8,32],
34,464 total parameters (untrained, random weights -- this just confirms
the architecture is wired correctly, no training has happened yet).

**Next:** masked modeling objective -- randomly mask tokens in the input,
add a prediction head, compute reconstruction loss. This is the actual
self-supervised training signal.

### 2026-07-17 — M5 steps 10-12: masking, MLM head, loss -- full objective working
**Decision:** Built mask_values (randomly masks ~15% of non-pad value
tokens), MLMHead (linear layer predicting value_id 0-9 from encoder output),
and wired up cross-entropy loss computed only on masked positions. First
real loss value: 2.7554 on untrained random weights (slightly above the
ln(10)=2.303 "pure random" baseline, consistent with our value_id
distribution being imbalanced across categorical fields of different sizes
-- not a bug).

**All pieces for the training loop now exist:** Dataset/DataLoader, full
model forward pass, masking, prediction head, loss computation. Next:
the actual training loop (optimizer + multiple epochs), first tested tiny
on Mac CPU, then a real run on Colab.