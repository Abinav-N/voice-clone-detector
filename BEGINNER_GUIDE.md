# Beginner's Walkthrough: AI Voice Clone Detector

This assumes you've never trained a machine learning model or used a terminal
much before. Follow it top to bottom, in order. Every command is copy-pasteable.

---

## Part 0: What you're actually building (in plain English)

1. You feed the computer thousands of short audio clips, each labeled
   **"real human"** or **"AI-generated."**
2. For each clip, you convert the sound wave into a picture-like grid of
   numbers called an **MFCC** (Mel-Frequency Cepstral Coefficients) — think
   of it as a fingerprint of *how the frequencies of the voice change over
   time.* Real voices and synthetic voices leave subtly different fingerprints.
3. You feed those fingerprints into a **CNN** (Convolutional Neural Network)
   — the same type of model used for image recognition, just applied to this
   audio "picture" instead of a photo.
4. The CNN learns to tell the two fingerprint patterns apart.
5. At the end, you give it a brand new clip it's never seen, and it outputs
   "REAL" or "FAKE" with a confidence percentage.

That's the whole project. Everything below is just the mechanics of doing this
on your computer.

---

## Part 1: Set up your computer (Day 1, ~30–60 min)

### 1.1 Install Python
You need Python 3.10 or 3.11.

- **Windows/Mac:** go to https://www.python.org/downloads/ and download the
  installer. **On Windows, tick "Add Python to PATH" during install** — this
  is the single most common thing beginners forget.
- **Mac (alternative, easier long-term):** install via Homebrew:
  `brew install python@3.11`

Check it worked — open a terminal (Mac: Terminal app; Windows: search for
"PowerShell") and type:
```bash
python3 --version
```
You should see something like `Python 3.11.x`. (On Windows it may just be
`python` instead of `python3` — try both.)

### 1.2 Install a code editor
Download **VS Code**: https://code.visualstudio.com/ — free, and it has a
built-in terminal so you don't need to juggle windows.

### 1.3 Get the project folder
You already have the `voice-clone-detector/` folder from earlier in this
chat. Download it, unzip if needed, and open the **folder** (not just a file)
in VS Code: `File → Open Folder`.

### 1.4 Open a terminal inside VS Code
`Terminal → New Terminal` (top menu). This opens a terminal already pointed
at your project folder — you'll run every command below from here.

### 1.5 Create a virtual environment
This keeps this project's Python packages separate from everything else on
your computer, so nothing conflicts.

```bash
python3 -m venv venv
```

Activate it (you must do this **every time** you open a new terminal for this
project):

- Mac/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

You'll know it worked because your terminal prompt now starts with `(venv)`.

### 1.6 Install the project's dependencies
```bash
pip install -r requirements.txt
```
This takes a few minutes — it's downloading PyTorch (the ML library),
librosa (audio processing), and a handful of smaller packages. If you hit an
error here, copy the exact error text and we'll troubleshoot it — don't
guess-fix it.

**Checkpoint:** run `python3 src/model.py` from inside the project folder. If
it prints `output shape: torch.Size([4, 2])` and a parameter count, your
environment is correctly set up and you can move on.

---

## Part 2: Get a dataset (Day 1, ~30–90 min depending on download speed)

Start with the **Fake-or-Real (FoR)** dataset — no registration wait, and
it's on Kaggle.

### 2.1 Create a free Kaggle account
https://www.kaggle.com/account/login — sign up if you don't have one.

### 2.2 Download the dataset
Go to: https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset
Click **Download** (top right). It's a few GB — this may take a while.

### 2.3 Unzip it
You'll get a folder with subfolders like `for-2sec`, `for-norm`, etc. — these
are different preprocessed versions of the same data. Use **`for-norm`**
(volume-normalized, real-length clips) to start.

Inside `for-norm` you'll find `training/`, `validation/`, `testing/`, and
inside each of those, folders like `real/` and `fake/`. That's exactly the
layout our `build_manifest.py` script expects, one level at a time.

### 2.4 Point the manifest builder at it
Run this **once per split** (training, then validation, then testing) — or,
simpler for a first pass, just run it on the `training/` folder and let our
script do its own train/val/test split internally:

```bash
python3 src/build_manifest.py --dataset folder --root "/path/to/for-norm/training" --out manifests/
```

Replace `/path/to/...` with wherever you unzipped it. On Mac, you can drag
the folder into the terminal window after typing the start of the command,
and it'll auto-fill the path.

**What this does:** it walks through `real/` and `fake/`, finds every audio
file, and writes three CSV files — `manifests/train.csv`,
`manifests/val.csv`, `manifests/test.csv` — each just a list of file paths
and labels. You should see a summary printed like:
```
train: 3200 rows (1610 bonafide, 1590 spoof) -> manifests/train.csv
```

If the numbers look wildly off (e.g. 0 spoof files), your folder names
probably don't match `real`/`fake` exactly — check the actual subfolder
names and either rename them or tell me the real names and I'll adjust the
script.

### 2.5 (Later, once approved) swap in ASVspoof 2019 LA
This is the "real" academic benchmark and worth doing once your pipeline
works end-to-end on FoR. Register at
https://datashare.ed.ac.uk/handle/10283/3336 (free, usually approved same
day). Once downloaded, use:
```bash
python3 src/build_manifest.py --dataset asvspoof \
  --root /path/to/asvspoof2019 \
  --protocol /path/to/ASVspoof2019.LA.cm.train.trn.txt \
  --audio-dir /path/to/ASVspoof2019_LA_train/flac \
  --out manifests/
```

---

## Part 3: Train the model (Day 2–3)

```bash
python3 src/train.py --train manifests/train.csv --val manifests/val.csv \
    --epochs 20 --batch-size 32 --out checkpoints/
```

**What you'll see:** a progress bar per "epoch" (one full pass through the
training data), then a summary line like:
```
Epoch 1: train_loss=0.6821 val_loss=0.5932 val_EER=18.40% val_acc=79.10%
```

- **Loss** — lower is better; a rough measure of "how wrong" the model's
  predictions are. It should trend down over epochs.
- **val_EER (Equal Error Rate)** — the key metric. Lower is better. This is
  the standard metric for this exact task. Getting under 10% is a solid
  result for a first pass on FoR; ASVspoof papers with heavier models get
  down to 1-5%, so don't expect to beat that with this lightweight CNN — the
  point is showing you understand the methodology, not chasing state of the
  art.
- **val_acc** — plain accuracy, easier to sanity-check by eye.

Training on a laptop CPU: expect maybe 5–20 minutes per epoch depending on
dataset size, so a full run could be an hour or more. If you have any GPU
(even a modest one) or access to **Google Colab** (free GPU, no setup), it'll
be much faster — see the Colab note in Part 6.

Every time validation EER improves, the script saves the best model so far to
`checkpoints/best.pt`. If your terminal or laptop dies mid-training, you
still have the best checkpoint up to that point.

**If training crashes:** copy the full error message. Common first-time
issues are "file not found" (a path in the manifest is wrong) or "out of
memory" (lower `--batch-size` to 16 or 8).

---

## Part 4: Evaluate it (Day 4)

```bash
python3 src/evaluate.py --checkpoint checkpoints/best.pt --manifest manifests/test.csv
```

This prints EER, AUC (another standard metric, 1.0 = perfect, 0.5 = random
guessing), a confusion matrix, and a precision/recall breakdown. This is the
headline result you'll put in your writeup or slides.

**For the strongest result in your report:** once you also have ASVspoof
trained, download the **In-the-Wild** dataset
(https://deepfake-total.com/in_the_wild — real deepfake scam-style audio) and
run `evaluate.py` on it *without retraining*. That "did it generalize to
audio it's never seen, from a completely different source" number is the
most convincing evidence your project actually addresses the real-world
problem, not just a lab benchmark.

---

## Part 5: Classify a single new clip (Day 4–5)

```bash
python3 src/infer.py --checkpoint checkpoints/best.pt --wav /path/to/some_clip.wav
```

Output looks like:
```
/path/to/some_clip.wav
  -> FAKE (AI-generated)  (P(fake)=87.3%, P(real)=12.7%)
```

Try this on a clip you generate yourself with any free TTS tool (e.g.
ElevenLabs free tier, or a quick Google TTS sample) plus a clip of your own
voice recorded on your phone — that side-by-side is a great live demo moment.

---

## Part 6: Optional — run it on Google Colab instead of your laptop

If your laptop is slow or has no GPU, Colab (free, browser-based, includes a
GPU) is worth it:

1. Go to https://colab.research.google.com, new notebook.
2. Upload your `voice-clone-detector` folder (zip it, upload, unzip in Colab
   with `!unzip voice-clone-detector.zip`), or push the folder to a private
   GitHub repo and `!git clone` it.
3. Upload your dataset to Google Drive, mount it (`from google.colab import
   drive; drive.mount('/content/drive')`), and point `build_manifest.py` at
   the Drive path.
4. Run the exact same commands as above, just prefixed with `!`, e.g.
   `!python3 src/train.py --train manifests/train.csv ...`
5. Download `checkpoints/best.pt` back to your laptop when done (for
   `infer.py`/demos) via the Colab file browser.

---

## Part 7: Build a simple demo (Day 5, optional but impressive)

Once you have `checkpoints/best.pt`, say the word and I'll build a small
Gradio app: you drop in or record a clip, click a button, and it shows
REAL/FAKE with a confidence bar — this is the thing to actually show people
rather than a terminal output.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | You forgot to activate the venv (`source venv/bin/activate`), or `pip install -r requirements.txt` didn't finish |
| `FileNotFoundError` during manifest building | The `--root` path is wrong, or subfolders aren't named exactly `real`/`fake` |
| Training loss stays flat / doesn't improve | Try a smaller `--lr` (e.g. `1e-4`), or double check labels in your manifest CSV aren't swapped |
| `CUDA out of memory` | You're on GPU with too large a batch — lower `--batch-size` |
| Everything predicts the same class | Usually a class-imbalance issue — check the bonafide/spoof counts printed by `build_manifest.py` |

---

## Where you should be, day by day

- **End of Day 1:** environment installed, `python3 src/model.py` runs, FoR
  dataset downloaded, `manifests/*.csv` generated with sane real/fake counts.
- **End of Day 2:** first full training run completes without crashing;
  you've seen EER trending downward across epochs.
- **End of Day 3:** trained checkpoint with EER you're reasonably happy with
  (aim for the model beating a coin-flip by a wide margin — under ~15% EER
  is a good first target on FoR).
- **End of Day 4:** `evaluate.py` results written down, ideally plus a
  cross-dataset test on In-the-Wild.
- **End of Day 5:** demo working, writeup/slides drafted.

Ping me at any step with the exact error message or output you're seeing —
paste it directly rather than describing it, and I can pinpoint the fix fast.
