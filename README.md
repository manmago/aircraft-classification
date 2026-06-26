# ✈️ Airplanes-ML

Identify a commercial aircraft (Boeing / Airbus first) from a photo of the
aircraft or one of its parts, using machine learning.

## Project status
🚧 Phase 0 — setup. See the roadmap below.

## Roadmap
- [ ] **0. Setup** — venv, structure, git, dependencies
- [ ] **1. Data** — acquire & explore a labeled aircraft image dataset
- [ ] **2. Baseline** — train a simple model end-to-end (any accuracy)
- [ ] **3. Improve** — transfer learning with a pretrained CNN, tune & evaluate
- [ ] **4. Inference** — upload a photo → predicted aircraft
- [ ] **5. Polish** — simple app/UI + write-up

## Setup (Windows)

This project uses **Python 3.12** (newer versions break ML libraries).

```powershell
# 1. Create the virtual environment
py -3.12 -m venv .venv

# 2. Activate it (do this every time you work on the project)
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

When activated, your terminal prompt shows `(.venv)`.
To leave the environment: `deactivate`.

## Project structure
```
Airplanes-ML/
├── data/         # datasets (NOT committed to git — too large)
├── notebooks/    # Jupyter notebooks for exploration
├── src/          # reusable Python code (data loading, model, training)
├── models/       # saved trained models (NOT committed to git)
├── requirements.txt
└── README.md
```
