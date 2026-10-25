"""Train and evaluate the Random Forest student-level classifier.

The augmented data (90,000 rows) is made of noisy copies of only 300 real students. A random split of those rows
puts copies of the same student in train and test, so accuracy measures memorisation. Here the 300 real students
are split first, only the training students are augmented, and the model is tested on real students it never saw
(repeated with several seeds). The "leaky" random-split number is printed for comparison.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import config
from src.data_augmentation import augment_dataset

N_REPEATS = 5
TEST_SIZE = 0.2


def new_model():
    return RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_split=10, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced", random_state=config.SEED, n_jobs=-1,
    )


def level_number(series):
    return series.str.extract(r"(\d+)")[0].astype(int)


def load_original():
    df = pd.read_csv(config.ORIGINAL_PATH)
    df["Level"] = level_number(df[config.TARGET])
    return df


def evaluate(original):
    rows, pooled_true, pooled_pred = [], [], []
    for seed in range(N_REPEATS):
        tr, te = train_test_split(original, test_size=TEST_SIZE, random_state=seed, stratify=original["Level"])
        aug = augment_dataset(tr.drop(columns="Level"), target_total=int(config.TARGET_TOTAL * (1 - TEST_SIZE)), seed=seed)
        aug["Level"] = level_number(aug[config.TARGET])

        with_aug = new_model().fit(aug[config.FEATURES], aug["Level"])
        train_acc = accuracy_score(aug["Level"], with_aug.predict(aug[config.FEATURES]))
        no_aug = new_model().fit(tr[config.FEATURES], tr["Level"])
        pred = with_aug.predict(te[config.FEATURES])
        pooled_true.extend(te["Level"]), pooled_pred.extend(pred)

        # robustness: noisy / contradictory profiles generated from the held-out students only
        noisy = augment_dataset(te.drop(columns="Level"), target_total=len(te) + 6000, seed=seed + 100, label_noise=0.0).iloc[len(te):]
        noisy_level = level_number(noisy[config.TARGET])
        rows.append({
            "train_acc_on_noisy_data": train_acc,
            "noisy_unseen_augmented_acc": accuracy_score(noisy_level, with_aug.predict(noisy[config.FEATURES])),
            "noisy_unseen_no_augmentation_acc": accuracy_score(noisy_level, no_aug.predict(noisy[config.FEATURES])),
            "augmented_acc": accuracy_score(te["Level"], pred),
            "augmented_macro_f1": f1_score(te["Level"], pred, average="macro"),
            "augmented_within_1": float((np.abs(pred - te["Level"]) <= 1).mean()),
            "no_augmentation_acc": accuracy_score(te["Level"], no_aug.predict(te[config.FEATURES])),
        })
    summary = pd.DataFrame(rows).mean().round(4).to_dict()

    # the number a random split of the augmented file gives (leaky: copies of the same student on both sides)
    full = pd.read_csv(config.AUGMENTED_PATH)
    full["Level"] = level_number(full[config.TARGET])
    Xtr, Xte, ytr, yte = train_test_split(full[config.FEATURES], full["Level"], test_size=TEST_SIZE,
                                          random_state=config.SEED, stratify=full["Level"])
    summary["leaky_random_split_acc"] = round(accuracy_score(yte, new_model().fit(Xtr, ytr).predict(Xte)), 4)
    return summary, pooled_true, pooled_pred


def save_plots(model, y_true, y_pred):
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)
    names = [f"{i} {config.LEVELS[i]}" for i in sorted(config.LEVELS)]
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted"), plt.ylabel("Actual")
    plt.title(f"Confusion matrix on unseen real students ({N_REPEATS} repeated hold-outs)")
    plt.xticks(rotation=30, ha="right"), plt.tight_layout()
    plt.savefig(config.IMG_DIR / "confusion_matrix.png", dpi=130), plt.close()

    imp = pd.Series(model.feature_importances_, index=config.FEATURES).sort_values()
    plt.figure(figsize=(7.5, 5))
    imp.plot.barh(color="#6c63ff")
    plt.title("Feature importance (final model)")
    plt.tight_layout(), plt.savefig(config.IMG_DIR / "feature_importance.png", dpi=130), plt.close()


def main():
    original = load_original()
    summary, y_true, y_pred = evaluate(original)
    print(f"Unseen students (mean of {N_REPEATS} hold-outs of {int(TEST_SIZE * len(original))} students):")
    print(json.dumps(summary, indent=2))
    print(classification_report(y_true, y_pred, zero_division=0))

    # final model: every real student, augmented to the full size
    full = pd.read_csv(config.AUGMENTED_PATH)
    full["Level"] = level_number(full[config.TARGET])
    final = new_model().fit(full[config.FEATURES], full["Level"])
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, config.MODEL_PATH, compress=3)
    config.METRICS_PATH.write_text(json.dumps(summary, indent=2))
    save_plots(final, y_true, y_pred)
    print(f"Saved {config.MODEL_PATH.name} ({config.MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
