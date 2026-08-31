# Car Type Detection with Classical ML

Classify vehicle images (bus, microbus, sedan, minivan, SUV, truck) from bounding-box features using logistic regression, decision trees, and a multi-layer perceptron.

## Overview

A data-mining course project that answers a simple but practical question: can we tell what kind of vehicle is in a photo using only the geometry of its bounding box? The dataset (a table of annotated bounding-box coordinates and vehicle-type labels) is transformed into width, height, perimeter, and area features, then fed into three classical scikit-learn classifiers. Each model is trained on 80% of the data and evaluated on the held-out 20%.

## Features

- Feature engineering from raw bounding-box coordinates (width, height, perimeter, area)
- One-hot encoding of the six vehicle classes
- Logistic regression, decision tree, and MLP classifiers compared side by side
- Train/test evaluation using accuracy and Jaccard similarity

## Tech Stack

- Python 3
- scikit-learn (LinearRegression, LogisticRegression, DecisionTreeClassifier, MLPClassifier)
- NumPy, Pandas, SciPy

## Installation

```bash
pip install numpy pandas scikit-learn scipy
```

## Usage

The project is a Jupyter notebook.

1. Open `DM_prj2.ipynb` in Jupyter or Google Colab.
2. The dataset `cars.xlsx` must be provided — it is not included in the repository. In Colab, upload it via `files.upload()`.

```python
# After loading the data, the notebook:
#   1. builds geometric features from bounding boxes
#   2. one-hot encodes the vehicle type
#   3. trains logistic regression, decision tree and MLP
#   4. reports train/test accuracy for each model
```

## Project Structure

```
DM_prj2.ipynb   # main notebook: data prep -> features -> 3 classifiers -> evaluation
```

## Examples

The three classifiers are evaluated on the same 80/20 split, so their scores are directly comparable. All metrics are computed at the end of the notebook.

## Roadmap

- Include the original dataset so the notebook runs out of the box
- Add a confusion matrix and per-class precision/recall
- Compare against a convolutional neural network baseline

## License

MIT

## Author

Seyed Farhad Hosseini
