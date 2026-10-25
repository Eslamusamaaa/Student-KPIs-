"""Central configuration: paths, feature names and constants shared by every module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
IMG_DIR = ROOT / "img" / "Student-KPIs-"

ORIGINAL_PATH = DATA_DIR / "Elara_Student_Features.csv"
AUGMENTED_PATH = DATA_DIR / "Student_KPIs_Elara_Augmented.csv"
MODEL_PATH = MODEL_DIR / "student_level_rf_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

SEED = 42
ID_COLUMN = "Student_ID"
TARGET = "Label"
TARGET_TOTAL = 90_000  # size of the augmented dataset

# Regularisation by noise: stronger feature noise in augmentation + a few neighbouring-level label flips,
# so the model cannot memorise the exact rules that produced the 300 labels.
NOISE_MULTIPLIER = 1.5
LABEL_NOISE = 0.03

FEATURES = [
    "Overall_Accuracy", "First_Try_Success_Rate", "Average_Time_Per_Question",
    "Hard_Question_Accuracy", "Attempts_Per_Question", "Hint_Usage_Rate",
    "Hint_Efficiency", "Time_Before_First_Hint", "Post_Hint_Improvement_Score",
    "Topic_Weakness_Count", "Score_Trend",
]

LEVELS = {1: "Beginner", 2: "Developing", 3: "Progressing", 4: "Proficient", 5: "Outstanding"}
