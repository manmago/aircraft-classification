"""Generate the Streamlit app's assets. Run ONCE locally (needs the dataset
and the trained model):

    python scripts/build_app_assets.py

Produces:
  app/assets/metrics.json          (test-set accuracy / balanced acc / macro-F1)
  app/assets/confusion_matrix.png  (test-set confusion matrix)
  app/samples/<class>/*.jpg        (a few images per class for the random picker)

NOTE on licensing: the copied samples are FGVC images (copyrighted). They're for
LOCAL testing. For a PUBLIC deploy, replace app/samples with your own or
CC-licensed photos before force-adding them to git.
"""
import sys
import json
import shutil
import random
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_dataloaders
from src.predict import load_model

ASSETS = PROJECT_ROOT / "app" / "assets"
SAMPLES = PROJECT_ROOT / "app" / "samples"
DATA = PROJECT_ROOT / "data"
MODEL = PROJECT_ROOT / "models" / "resnet18_aircraft5.pt"
N_PER_CLASS = 6

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model, classes = load_model(str(MODEL), device=device)
_, _, test_loader, classes = get_dataloaders(str(DATA))
test_ds = test_loader.dataset

# --- run the model over the test set ---
preds, trues = [], []
model.eval()
with torch.no_grad():
    for imgs, labels in test_loader:
        out = model(imgs.to(device))
        preds += out.argmax(1).cpu().tolist()
        trues += labels.tolist()

ASSETS.mkdir(parents=True, exist_ok=True)

metrics = {
    "accuracy": accuracy_score(trues, preds),
    "balanced_accuracy": balanced_accuracy_score(trues, preds),
    "macro_f1": f1_score(trues, preds, average="macro"),
}
(ASSETS / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("metrics:", metrics)

(ASSETS / "classification_report.txt").write_text(
    str(classification_report(trues, preds, target_names=classes, digits=3)))
print("saved classification_report.txt")

disp = ConfusionMatrixDisplay(confusion_matrix(trues, preds), display_labels=classes)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
plt.tight_layout()
fig.savefig(str(ASSETS / "confusion_matrix.png"), dpi=120)
print("saved confusion_matrix.png")

# --- copy a few sample images per class (local testing only; see licensing note) ---
by_class = {}
for i in range(len(test_ds)):
    path, name = test_ds.path_and_label(i)
    by_class.setdefault(name, []).append(path)

for name, paths in by_class.items():
    out_dir = SAMPLES / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in random.sample(paths, min(N_PER_CLASS, len(paths))):
        shutil.copy(p, out_dir / Path(p).name)
print("copied sample images to", SAMPLES)
print("\nDone. Now run:  streamlit run app/streamlit_app.py")
