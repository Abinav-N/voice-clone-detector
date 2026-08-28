# AI Voice Clone Detector

Binary classifier that tells real human speech apart from AI-generated / cloned
speech, using MFCC-based spectral features and a lightweight 2D CNN.

## Why this works as a standout project
- Audio deepfake detection is under-explored compared to text/image projects.
- It maps directly onto a real fraud vector (cloned-CEO phone scams, voice-auth bypass).
- The pipeline (MFCC → CNN → EER evaluation) mirrors the actual methodology used
  in the ASVspoof challenge, so your results are directly comparable to published work.

---

## 1. Datasets

You need **bonafide (real)** and **spoof (AI-generated / cloned)** speech, ideally
from more than one source so you can test generalization (a model that only sees
one TTS engine can shortcut on training on artifacts specific to that engine).

| Dataset | What it is | Access | Notes |
|---|---|---|---|
| **ASVspoof 2019 LA** | Gold-standard benchmark. ~121k utterances, 19 different TTS/voice-conversion attack types | Free, requires registration: https://datashare.ed.ac.uk/handle/10283/3336 | Use this as your primary train/eval set — it's what makes results comparable to published papers |
| **ASVspoof 2021 LA/DF** | Harder, includes codec/compression conditions | Same portal | Good stretch goal for "robustness" section |
| **In-the-Wild (Müller et al., 2022)** | Real celebrity speech vs. real deepfakes found in the wild | Free download, no registration: https://deepfake-total.com/in_the_wild | Closest to the actual scam scenario — great for a cross-dataset generalization test |
| **WaveFake** | GAN-vocoder generated fakes (MelGAN, HiFi-GAN, etc.) | Free on GitHub/Zenodo | Good extra attack diversity |
| **Fake-or-Real (FoR)** | Kaggle-hosted, easiest to get running fast | https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset | Use this on Day 1 if ASVspoof registration is still pending, so you're not blocked |

**Recommended strategy:** start with FoR (instant download) to get the pipeline
working end-to-end on Day 1–2, then swap in ASVspoof 2019 LA once it's approved
(usually same-day) as your real benchmark. Reserve In-the-Wild purely as an unseen
test set — never train on it — that's your "does this generalize to real scam audio" number.

Every script in this repo reads from a simple **CSV manifest** (`filepath,label`,
label = `bonafide` or `spoof`) rather than any one dataset's native format, so
switching datasets just means regenerating the manifest. See `src/build_manifest.py`.

---

## 2. Approach

```
raw audio (.wav/.flac)
   → resample to 16kHz mono
   → extract MFCC + delta + delta-delta (3 x 40 x T)
   → fixed-length pad/crop
   → 2D CNN (conv blocks → global pool → FC)
   → sigmoid: P(spoof)
```

- **Features:** 40 MFCCs + first/second derivatives, stacked as 3 "channels" —
  this is a standard, well-understood representation and keeps the model small
  enough to train on a laptop/free Colab GPU in a few hours.
- **Model:** ~4 conv blocks, batch norm, global average pooling, 2 FC layers.
  Small enough to train fast, big enough to clearly beat a linear baseline.
- **Metric:** Equal Error Rate (EER) — the standard ASVspoof metric — plus
  accuracy/AUC. EER matters more than accuracy here because these datasets are
  often class-imbalanced.

---

## 3. Day-by-day plan (4–5 days)

**Day 1 — Setup & data**
- Install deps (`requirements.txt`), download FoR (fast) + submit ASVspoof registration.
- Run `src/build_manifest.py` to generate `manifests/train.csv`, `val.csv`, `test.csv`.
- Sanity check: play a few real vs fake clips, look at waveform/spectrogram by eye.

**Day 2 — Feature pipeline**
- Get `src/features.py` extracting MFCC/delta/delta2 correctly (check output shapes).
- Build the `Dataset`/`DataLoader` in `src/dataset.py`, verify a batch loads correctly.
- Swap in ASVspoof manifest once available.

**Day 3 — Train baseline model**
- Run `src/train.py`, watch train/val loss and EER per epoch.
- Iterate: learning rate, batch size, dropout, maybe class-weighted loss if imbalanced.
- Save best checkpoint by validation EER (not accuracy).

**Day 4 — Evaluation & generalization**
- Run `src/evaluate.py` on the ASVspoof held-out eval set → report EER/AUC/ROC.
- Run the *same* trained model on In-the-Wild (no fine-tuning) → this cross-dataset
  number is the most interesting one for a "real-world scam detection" narrative.
- Error analysis: which attack types does it miss? Confusion matrix by attack ID.
- Optional: light augmentation (add noise, MP3-compress) and re-check robustness.

**Day 5 — Polish & demo**
- Build a tiny CLI or Gradio demo (`src/infer.py` is the core; wrapping in Gradio
  is ~20 lines) so you can drop in a live recording and get REAL/FAKE + confidence.
- Write up results: EER table, ROC curve, cross-dataset generalization gap,
  a couple of example spectrograms of real vs. fake for the report/slides.

---

## 4. Repo layout

```
voice-clone-detector/
├── README.md
├── requirements.txt
├── manifests/                 # generated CSV manifests go here
└── src/
    ├── build_manifest.py      # dataset-specific -> generic CSV manifest
    ├── features.py            # MFCC/delta/delta2 extraction
    ├── dataset.py             # PyTorch Dataset/DataLoader
    ├── model.py                # CNN architecture
    ├── train.py                # training loop, checkpointing, EER-based model selection
    ├── evaluate.py             # EER / AUC / ROC / confusion matrix on a manifest
    └── infer.py                 # classify a single wav file (REAL / FAKE + confidence)
```

## 5. Quickstart

```bash
pip install -r requirements.txt

# 1. Build manifests (edit paths inside for your dataset layout)
python src/build_manifest.py --dataset for --root /path/to/for_dataset --out manifests/

# 2. Train
python src/train.py --train manifests/train.csv --val manifests/val.csv \
    --epochs 20 --batch-size 32 --out checkpoints/

# 3. Evaluate
python src/evaluate.py --checkpoint checkpoints/best.pt --manifest manifests/test.csv

# 4. Classify a single file
python src/infer.py --checkpoint checkpoints/best.pt --wav path/to/clip.wav
```

## 6. Stretch goals (if you finish early)
- Swap MFCC for log-mel spectrograms and compare — does the CNN do better with
  a richer time-frequency representation?
- Try a lightweight end-to-end raw-waveform model (e.g. a small SincNet/RawNet-style
  front end) and compare against the MFCC+CNN pipeline.
- Test robustness to phone-call codec compression (8kHz, GSM/AMR) — this is the
  realistic condition for the "fake CEO phone call" scenario.
- Package as a Gradio app with live microphone input for a compelling demo.
