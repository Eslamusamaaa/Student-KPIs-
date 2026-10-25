"""Exploratory analysis of the student KPIs: saves figures to img/Student-KPIs-/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

sns.set_theme(style="whitegrid")
ORDER = [f"Level {i}" for i in range(1, 6)]


def save(name):
    plt.tight_layout()
    plt.savefig(config.IMG_DIR / name, dpi=130)
    plt.close()


def main():
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(config.ORIGINAL_PATH)
    augmented = pd.read_csv(config.AUGMENTED_PATH)
    print("Original:", original.shape, "| augmented:", augmented.shape)
    print(original[config.FEATURES].describe().round(2).T.to_string())

    plt.figure(figsize=(7, 4))
    sns.countplot(x=config.TARGET, data=original, order=ORDER, hue=config.TARGET, legend=False, palette="viridis")
    plt.title("Student levels in the 300 real records (balanced)")
    save("eda_level_distribution.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(original[config.FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation between KPIs (real students)")
    save("eda_correlation.png")

    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    for ax, col in zip(axes.ravel(), config.FEATURES):
        sns.boxplot(x=config.TARGET, y=col, data=original, order=ORDER, hue=config.TARGET, legend=False, palette="viridis", ax=ax)
        ax.set_title(col, fontsize=10), ax.set_xlabel("")
    axes.ravel()[-1].axis("off")
    save("eda_boxplots.png")

    cols = ["Overall_Accuracy", "Hint_Usage_Rate", "Hard_Question_Accuracy"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    both = pd.concat([original.assign(Data="Real (300)"), augmented.iloc[len(original):].sample(3000, random_state=config.SEED).assign(Data="Augmented")])
    for ax, col in zip(axes, cols):
        sns.kdeplot(data=both, x=col, hue="Data", fill=True, common_norm=False, ax=ax)
        ax.set_title(f"{col}: real vs augmented")
    save("eda_real_vs_augmented.png")
    print("Saved figures to", config.IMG_DIR)


if __name__ == "__main__":
    main()
