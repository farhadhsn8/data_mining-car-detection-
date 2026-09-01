# Car Type Detection with Classical ML

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Classify vehicle images (**bus**, **microbus**, **sedan**, **minivan**, **SUV**, **truck**) from bounding-box geometry using classical machine learning.

---

## About

> Classify six types of vehicles from bounding-box geometry using classical machine learning.

A hands-on **data-mining / machine-learning** project that engineers geometric features (width, height, perimeter, area) from raw bounding-box annotations, then benchmarks **Linear Regression**, **Logistic Regression**, **Decision Tree**, and **Multi-Layer Perceptron (MLP)** classifiers on the same 80/20 train/test split. All models are evaluated with accuracy and Jaccard similarity, and results are reproducible via a fixed random seed.

> **GitHub description (copy-paste into repo settings):**
> `Classify car types (bus, sedan, SUV, truck, ...) from bounding-box geometry using logistic regression, decision trees, and MLP — a scikit-learn data-mining exercise.`

> **GitHub topics (copy-paste into repo settings):**
> `data-mining` `machine-learning` `scikit-learn` `classification` `logistic-regression` `decision-tree` `mlp` `jupyter-notebook` `python` `numpy` `pandas` `scipy`

---

## Overview

A data-mining course project that answers a simple, practical question:

> Can we tell what kind of vehicle is in a photo using only the geometry of its bounding box?

Starting from a table of annotated bounding-box coordinates and vehicle-type labels, we engineer geometric features (width, height, perimeter, area), one-hot encode the six vehicle classes, and benchmark three classical scikit-learn models on an 80/20 train/test split.

| Model | Type | Target |
|---|---|---|
| Linear Regression | multi-output regression | one-hot classes |
| Logistic Regression | binary classifier | `bus` vs rest |
| Decision Tree | binary classifier | `bus` vs rest |
| MLP (multi-layer perceptron) | binary classifier | `bus` vs rest |

Evaluation is reported as **accuracy** (`score`) and **Jaccard similarity**.

---

## Features

- Feature engineering from raw bounding-box coordinates (width, height, perimeter, area)
- One-hot encoding of the six vehicle classes
- Four models compared side by side on the same split
- Reproducible results via a fixed `random_state`
- Clean, modular package layout with unit tests

---

## Tech Stack

- Python 3.8+
- scikit-learn (`LinearRegression`, `LogisticRegression`, `DecisionTreeClassifier`, `MLPClassifier`)
- NumPy, Pandas, SciPy

---

## Installation

```bash
git clone https://github.com/<your-username>/car-type-classification.git
cd car-type-classification
pip install -r requirements.txt
```

> **Note:** the dataset `cars.xlsx` is not included in the repository.
> Drop it into `data/` (i.e. `data/cars.xlsx`) before running.

---

## Usage

Run the full pipeline (data prep → feature engineering → training → evaluation):

```bash
python main.py
```

Run the unit tests:

```bash
pytest
```

Run the original notebook:

```bash
jupyter notebook DM_prj2.ipynb
```

---

## Project Structure

```
.
├── DM_prj2.ipynb          # original notebook (kept as reference)
├── main.py                # entry point for the full pipeline
├── config.py              # central configuration & constants
├── requirements.txt
├── src/
│   ├── data/              # data loading, feature engineering, dataset prep
│   │   ├── loader.py
│   │   ├── features.py
│   │   └── dataset.py
│   ├── models/            # one module per model + shared metrics
│   │   ├── metrics.py
│   │   ├── linear_regression.py
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   └── mlp.py
│   └── pipeline.py        # end-to-end orchestration
└── tests/                 # unit tests
    └── test_data.py
```

---

## How It Works

1. **Load** the annotated dataset from `cars.xlsx`.
2. **Feature engineering** — from each bounding box `(x1, y1, x2, y2)` compute `width`, `height`, `perimeter`, `area`, plus a bias column.
3. **Encode labels** — one-hot encode the six vehicle classes.
4. **Split** — shuffle the combined matrix and reserve 80% for training, 20% for testing.
5. **Train & evaluate** — fit each model on the same split and report train/test score and Jaccard similarity.

---

## Roadmap

- Include the original dataset so the project runs out of the box
- Add a confusion matrix and per-class precision/recall
- Compare against a convolutional neural network baseline
- Expose per-class classification reports for all models

---

## License

[MIT](LICENSE)

## Author

[Seyed Farhad Hosseini](https://github.com/<your-username>)
