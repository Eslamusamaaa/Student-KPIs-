"""Synthetic augmentation of the 300 real student records.

Every augmented record is a noisy copy of a real record (same Student_ID and Label) with:
Gaussian noise, shocks on Score_Trend, ~12% contradictory profiles and ~2% outliers.
Because augmented rows are copies of real students, they must never be split between train and test:
augment AFTER splitting (see train.py), otherwise the model is tested on near-duplicates of its training rows.
"""
import numpy as np
import pandas as pd

import config

SCALES = {
    "Overall_Accuracy": 0.07, "First_Try_Success_Rate": 0.12, "Average_Time_Per_Question": 15.0,
    "Hard_Question_Accuracy": 0.07, "Attempts_Per_Question": 0.7, "Hint_Usage_Rate": 0.12,
    "Hint_Efficiency": 0.12, "Time_Before_First_Hint": 15.0, "Post_Hint_Improvement_Score": 0.12,
    "Topic_Weakness_Count": 3, "Score_Trend": 0.08,
}
RATIO_COLUMNS = ["Overall_Accuracy", "First_Try_Success_Rate", "Hard_Question_Accuracy",
                 "Hint_Usage_Rate", "Hint_Efficiency", "Post_Hint_Improvement_Score"]
CONTRADICTIONS = ["highAcc_highHint", "lowAcc_highFirstTry", "highAcc_manyAttempts", "lowAcc_shortTime"]


def scale(col):
    return SCALES[col] * config.NOISE_MULTIPLIER


def truncated_normal(rng, mean, sd, low, high):
    val = rng.normal(mean, sd)
    for _ in range(10):
        if low <= val <= high:
            break
        val = rng.normal(mean, sd)
    return float(np.clip(val, low, high))


def generate_record(base, rng, label_noise=0.0):
    rec = dict(base)
    for col in RATIO_COLUMNS:
        rec[col] = round(truncated_normal(rng, float(rec[col]), scale(col), 0.0, 1.0), 3)
    for col in ("Average_Time_Per_Question", "Time_Before_First_Hint"):
        rec[col] = round(max(float(rec[col]) + rng.normal(0, scale(col)), 1.0), 2)
    rec["Attempts_Per_Question"] = round(max(float(rec["Attempts_Per_Question"]) + rng.normal(0, scale("Attempts_Per_Question")), 0.1), 2)
    spread = int(round(SCALES["Topic_Weakness_Count"] * config.NOISE_MULTIPLIER))
    shift = int(rng.integers(-spread, spread + 1))
    rec["Topic_Weakness_Count"] = int(max(int(rec["Topic_Weakness_Count"]) + shift, 0))

    p = rng.random()
    if p < 0.05:
        trend = -abs(rng.uniform(0.1, 0.5))
    elif p < 0.10:
        trend = abs(rng.uniform(0.1, 0.5))
    else:
        trend = float(rec["Score_Trend"]) + rng.normal(0, scale("Score_Trend"))
    rec["Score_Trend"] = round(float(np.clip(trend, -0.6, 0.6)), 3)

    if rng.random() < 0.12:  # contradictory profiles so the model does not rely on one KPI
        kind = rng.choice(CONTRADICTIONS)
        if kind == "highAcc_highHint":
            rec["Overall_Accuracy"] = round(min(1.0, max(rec["Overall_Accuracy"], rng.uniform(0.8, 0.98))), 3)
            rec["Hint_Usage_Rate"] = round(min(1.0, max(rec["Hint_Usage_Rate"], rng.uniform(0.3, 0.8))), 3)
            rec["First_Try_Success_Rate"] = round(min(1.0, rec["First_Try_Success_Rate"] * rng.uniform(0.6, 0.95)), 3)
        elif kind == "lowAcc_highFirstTry":
            rec["Overall_Accuracy"] = round(min(1.0, rec["Overall_Accuracy"] * rng.uniform(0.3, 0.6)), 3)
            rec["First_Try_Success_Rate"] = round(min(1.0, max(rec["First_Try_Success_Rate"], rng.uniform(0.6, 0.9))), 3)
        elif kind == "highAcc_manyAttempts":
            rec["Overall_Accuracy"] = round(min(1.0, max(rec["Overall_Accuracy"], rng.uniform(0.8, 0.98))), 3)
            rec["Attempts_Per_Question"] = round(max(rec["Attempts_Per_Question"], rng.uniform(1.5, 4.0)), 3)
        else:
            rec["Overall_Accuracy"] = round(min(1.0, rec["Overall_Accuracy"] * rng.uniform(0.2, 0.6)), 3)
            rec["Average_Time_Per_Question"] = round(max(1.0, rec["Average_Time_Per_Question"] * rng.uniform(0.2, 0.6)), 2)

    if rng.random() < 0.02:  # rare outliers
        col = rng.choice(["Average_Time_Per_Question", "Topic_Weakness_Count", "Hint_Usage_Rate"])
        if col == "Average_Time_Per_Question":
            rec[col] = round(rec[col] * rng.uniform(2.0, 5.0), 2)
        elif col == "Topic_Weakness_Count":
            rec[col] = int(rec[col] + rng.integers(5, 15))
        else:
            rec[col] = round(min(1.0, rec[col] + rng.uniform(0.3, 0.7)), 3)

    for col in RATIO_COLUMNS:
        rec[col] = float(np.clip(rec[col], 0.0, 1.0))
    rec["Attempts_Per_Question"] = float(max(rec["Attempts_Per_Question"], 0.1))
    if rng.random() < label_noise:  # boundary noise: the label moves to a neighbouring level
        level = int(str(rec[config.TARGET]).split()[-1])
        level = int(np.clip(level + rng.choice([-1, 1]), 1, 5))
        rec[config.TARGET] = f"Level {level}"
    return rec


def augment_dataset(df, target_total=config.TARGET_TOTAL, seed=config.SEED, label_noise=config.LABEL_NOISE):
    """Return `df` plus noisy copies of its rows until it has `target_total` rows."""
    rng = np.random.default_rng(seed)
    existing = df.to_dict(orient="records")
    new = [generate_record(existing[rng.integers(len(existing))], rng, label_noise) for _ in range(target_total - len(existing))]
    return pd.concat([df, pd.DataFrame(new)], ignore_index=True)


def main():
    original = pd.read_csv(config.ORIGINAL_PATH)
    augmented = augment_dataset(original)
    augmented.to_csv(config.AUGMENTED_PATH, index=False)
    print(f"Saved {config.AUGMENTED_PATH.name}: {augmented.shape}")
    print(augmented[config.TARGET].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
