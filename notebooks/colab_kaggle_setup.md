# Colab / Kaggle setup — first cells to run

Paste these as the first cells of a new Colab or Kaggle notebook.
Goal: same repo, same requirements.txt, same env_check.py pass as on your Mac.

## Cell 1 — clone your repo (after you've pushed it to GitHub in step 5 below)

```python
!git clone https://github.com/<your-username>/mini-pragma.git
%cd mini-pragma
```

If you haven't pushed to GitHub yet, upload the folder as a zip instead:
```python
from google.colab import files
uploaded = files.upload()  # upload mini-pragma.zip
!unzip -q mini-pragma.zip
%cd mini-pragma
```

## Cell 2 — install dependencies
Colab/Kaggle already ship numpy/pandas/torch, so this is usually fast.

```python
!pip install -q -r requirements.txt
```

## Cell 3 — confirm GPU is attached
Colab: Runtime > Change runtime type > T4 GPU
Kaggle: Settings (right panel) > Accelerator > GPU P100

```python
!nvidia-smi
```

## Cell 4 — run the same parity check you ran locally

```python
!python scripts/env_check.py
```

You should see `Accelerator : CUDA (...)` here, vs. `Accelerator : none (CPU only)`
on your Mac. That's the *only* expected difference in the output — everything else
(package versions) should match. If they don't match, pin versions in
requirements.txt and reinstall on both sides before moving to M2.

## Saving progress across sessions (do this now, not when you need it)

Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
# save checkpoints to /content/drive/MyDrive/mini-pragma-checkpoints/
```

Kaggle: use the notebook's own persistent output directory (`/kaggle/working/`),
or attach a private Kaggle Dataset as a checkpoint store — Kaggle notebooks
support background execution, so a long run can keep going after you close the tab.
