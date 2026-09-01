from pathlib import Path

DATA_DIR = Path("data")
CARS_FILE = DATA_DIR / "cars.xlsx"

CLASS_LABELS = ["bus", "microbus", "sedan", "minivan", "suv", "truck"]

# Feature matrix built by src.data.features.build_features (9 columns):
# index 0        -> bias
# index 1..4     -> bounding-box coordinates (x1, y1, x2, y2)
# index 5..6     -> width, height
# index 7..8     -> perimeter, area
# The models only consume the first 7 columns, matching the notebook.
FEATURE_SLICE = slice(0, 7)
ONE_HOT_SLICE = slice(9, 15)
BUS_CLASS_COL = 9

TRAIN_RATIO = 0.8
RANDOM_STATE = 0
