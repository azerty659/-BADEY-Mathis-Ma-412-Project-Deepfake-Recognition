Deepfake Recognition — Ma 412 Project

This repository contains a notebook and supporting files for a project on AI-generated image detection (deepfake / generative images detection). The goal is to classify images as either **AI-generated (fake)** or **real** using classical image features and modern deep learning methods, compare their accuracy and speed, and provide a reproducible experiment pipeline.

---

## Contents

- `Ma_412_project_BADEY_Mathis.ipynb` — Main notebook with data processing, two methods, timing, and evaluation.
- `requirements.txt` — Python dependencies (see *Environment*).
- `data_2/`  — Expected dataset folders (not included in repo):
  - `fake`
  - `real`
- `models/` — (created by the notebook) saved trained models and artifacts.
- `README.md` — This file.

---

## Project Overview

- **Classical feature pipeline:**
  - Image-level feature extraction implemented in the notebook includes:
    - Color histograms (R, G, B channels) — normalized hist counts
    - Channel means and standard deviations
    - Grayscale mean & std
    - Edge density (Sobel-based)
    - FFT-derived features (magnitude mean, variance, 95th percentile)
    - Blur measure (Variance of Laplacian)
  - After extraction the pipeline performs Standard Scaling and PCA (retain ~95% variance) and trains classical classifiers: RandomForest (with GridSearchCV hyperparameter tuning), MLP (grid search), and SVM (grid search). Training/prediction times and evaluation metrics are computed and saved.

The notebook reports training/prediction times, accuracy, classification reports, confusion matrices, and ROC AUC for whichever methods you run so you can compare accuracy and speed quantitatively.

---

## Quick Start

1. Clone or copy the project folder to your machine.
2. Install dependencies in a Python environment (recommend Python 3.8–3.10):

```powershell
cd "c:\Users\mathi\Desktop\outils_mathematiques_avances\Project"
pip install -r .\requirements.txt
```

3. Prepare your dataset (place it in the project root):

- `data/` or `data_reduced/` with the structure:
  - `train/fake`, `train/real`
  - `test/fake`, `test/real`

4. Open the notebook with Jupyter or VS Code and run cells top-to-bottom:

```powershell
jupyter notebook
# or open `Ma_412_project_BADEY_Mathis.ipynb` in VS Code
```

Notes:
- The notebook automatically prefers `data_2` if present, which allows quicker iteration during development.

---

## Reproducibility & Outputs

- Models, scalers, and PCA objects are saved to `./models/` by the notebook (`scaler.joblib`, `pca.joblib`, `random_forest.joblib`, `mlp.joblib`, `svm.joblib`.
- Plots and evaluation artifacts (confusion matrices, ROC plots, text reports) are saved to `./models report/` as PNG/TXT files.

The notebook also includes a `predict_image(image_path, model='rf')` helper to preprocess a single image (feature extraction, scaling, PCA) and run a saved model; a short demo at the end of the notebook shows how to call it for single-image testing.

When comparing methods, record at minimum:

- **Accuracy**, **Precision**, **Recall**, **F1-score** (from classification reports)
- **Confusion matrix** — to inspect false positives/negatives
- **ROC AUC**
- **Training time** and **Prediction time** (measured in the notebook)

---

## Where to Find Key Code Sections in the Notebook

- Data loading and exploration — top of the notebook (dataset discovery / counts)
- Feature extraction functions (LBP, histograms, FFT features) — Data processing cells
- Classical models (SVM, RandomForest, MLP) — model training sections
- Evaluation & comparison — final sections that print metrics and save reports

---

## How to Improve Performance (Suggestions)

- Increase training data or use stronger augmentation
- Tune hyperparameters: grid search for SVM, RandomForest, and MLP
- For CNN: unfreeze part of the pretrained backbone and fine-tune with a low learning rate
- Try different backbones (ResNet, EfficientNet) depending on compute
- Use stratified cross-validation for robust estimates

---

## Dependencies

All required packages are listed in `requirements.txt`. Key packages include:

- Python 3.8+ (recommended)
- numpy, scipy, scikit-image, scikit-learn
- imageio, opencv-python, pillow
- matplotlib, joblib

---

## License & Acknowledgements

This project is provided for educational purposes. If you share or extend this work, please include attribution. Feel free to add a license (e.g., MIT) if you want reuse permission.

---

## Contact

If you want help running experiments, reproducing results, or extending the notebook (e.g., adding more augmentations, different backbones, cross-validation loops), open an issue or contact me via the project workspace.

Happy experimenting! 🚀
