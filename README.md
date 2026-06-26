# Aircraft recognition

As an aviation freak, I started memorizing and learning to differentiate aircraft models. 
Since I am struggling sometimes, I decided to use the knowledge gained from my last uni module
to build an image classifyer/recognition tool, which can tell what Boeing/Airbus model an image depicts. 

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
