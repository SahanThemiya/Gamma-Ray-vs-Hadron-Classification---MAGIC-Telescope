# Gamma-Ray-vs-Hadron-Classification---MAGIC-Telescope
Classifies Cherenkov telescope events as gamma ray signals or hadron background using image parameters from the MAGIC telescope in La Palma, Spain.
Binary classification of Cherenkov telescope events recorded by the MAGIC telescope in La Palma, Spain. The goal is to separate genuine gamma-ray signals from the dominant hadronic cosmic-ray background using Hillas image parameters extracted from telescope camera images.

---

## Dataset

[UCI MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)

| Property | Value |
|---|---|
| Samples | 19,020 |
| Features | 10 (Hillas parameters) |
| Classes | gamma (65.2%) / hadron (34.8%) |
| Format | CSV, no header |

Download `magic04.data` and place it in the project root before running.

### Features

| Feature | Description |
|---|---|
| fLength | Major axis of shower ellipse [mm] |
| fWidth | Minor axis of shower ellipse [mm] |
| fSize | log10 of total photon content |
| fConc | Ratio of two highest pixels to fSize |
| fConc1 | Ratio of highest pixel to fSize |
| fAsym | Distance from highest pixel to center [mm] |
| fM3Long | 3rd root of 3rd moment along major axis [mm] |
| fM3Trans | 3rd root of 3rd moment along minor axis [mm] |
| fAlpha | Angle of major axis with vector to origin [deg] |
| fDist | Distance from origin to ellipse center [mm] |

---

## Project Structure

```
├── magic04.data              # raw data (download separately)
├── main.py                   # full pipeline as a script
├── notebook.ipynb            # self-contained Jupyter notebook
├── requirements.txt
├── plots/                    # auto-generated output plots
└── src/
    ├── __init__.py
    ├── data.py               # load(), split(), StandardScaler
    ├── eda.py                # EDA plot functions
    └── models.py             # model definitions and evaluation
```

---

## Setup

```bash
git clone https://github.com/SahanThemiya/Gamma-Ray-vs-Hadron-Classification---MAGIC-Telescope.git
cd Gamma-Ray-vs-Hadron-Classification---MAGIC-Telescope

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Download `magic04.data` from UCI and place it in the project root.

---

## Run

**Script:**
```bash
python main.py
```

**Notebook:**
```bash
jupyter notebook notebook.ipynb
```

Plots are saved to `plots/`.

---

## Models

Both models trained on 80/20 stratified split with class imbalance handling.

| Model | Imbalance Strategy |
|---|---|
| RandomForest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight=neg/pos` |

---

## Results

| Model | Accuracy | ROC-AUC | PR-AUC | Gamma Recall |
|---|---|---|---|---|
| RandomForest | 0.88 | 0.9402 | 0.9618 | **0.95** |
| XGBoost | **0.89** | **0.9422** | **0.9631** | 0.92 |

XGBoost edges out on overall accuracy and AUC metrics. RandomForest achieves higher gamma recall (0.95 vs 0.92) — more relevant in a real telescope context where missing a genuine gamma-ray event is costlier than passing background.

### Key Finding

`fAlpha` is the dominant discriminator (~26% importance in both models), consistent with physics: gamma-ray showers originate from a known point source and thus have low shower orientation angles, while hadronic showers are isotropic. Removing `fAlpha` and `fDist` produces a harder, more purely ML-driven benchmark.

### Confusion Matrix Summary (test set, n=3,804)

**RandomForest:** 1025 TN | 313 FP | 128 FN | 2338 TP

**XGBoost:** 1103 TN | 235 FP | 199 FN | 2267 TP

---

## Plots

| Plot | Description |
|---|---|
| `class_dist.png` | Class imbalance overview |
| `feature_distributions.png` | KDE per feature split by class |
| `correlation.png` | Hillas parameter correlation matrix |
| `pairplot.png` | Pairwise scatter of key features |
| `cm_RandomForest.png` | Confusion matrix — RandomForest |
| `cm_XGBoost.png` | Confusion matrix — XGBoost |
| `roc_curves.png` | ROC curves — both models |
| `pr_curves.png` | Precision-Recall curves — both models |
| `fi_RandomForest.png` | Feature importances — RandomForest |
| `fi_XGBoost.png` | Feature importances — XGBoost |

---
