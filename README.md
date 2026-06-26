# Aircraft recognition

As an aviation geek, I started memorizing and learning to differentiate aircraft models. 
Since I am struggling sometimes, I decided to use the knowledge gained from my last uni module
to build an image classifyer/recognition tool using the FGVC¹ dataset, which can tell what Boeing/Airbus model an image depicts.

Currently, **5** classes are selected:
- Boeing 737
- Boeing 777
- Airbus A320 
- Airbus A340
- Airbus A380

## Setup (Windows)

This project uses **Python 3.13** (very newest versions can lag ML libraries).

```powershell
# 1. Create the virtual environment
py -3.13 -m venv .venv

# 2. Activate it (do this every time you work on the project)
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install the standard dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Install PyTorch with NVIDIA GPU (CUDA) support — NOT from requirements.txt,
#    because the GPU build lives on PyTorch's own package index.
#    Get the exact command for your setup at https://pytorch.org/get-started/locally/
#    (select: Stable / Windows / Pip / Python / the newest CUDA shown). Example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
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

# References

¹ Fine-Grained Visual Classification of Aircraft, S. Maji, J. Kannala, E. Rahtu, M. Blaschko, A. Vedaldi, arXiv.org, 2013