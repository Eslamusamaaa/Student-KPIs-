"""Loads the trained model and turns KPIs into a level plus adaptive-support suggestions."""
import joblib
import pandas as pd

import config

# What the Elara app should do for a student of each level.
SUPPORT = {
    1: "Offer step-by-step hints early, simplify explanations and start with easy questions to build confidence.",
    2: "Keep hints available but nudge the student to try first; revisit the basics with short worked examples.",
    3: "Reduce hint prompts, mix in medium questions and target the weakest topics.",
    4: "Introduce harder questions and timed challenges; hints only on request.",
    5: "Offer advanced and open-ended challenges; the student rarely needs support.",
}


def kpi_observations(kpis):
    """Plain-language flags for the KPIs that explain the level."""
    notes = []
    if kpis["Hint_Usage_Rate"] > 0.5:
        notes.append("Asks for hints on more than half of the questions.")
    if kpis["Topic_Weakness_Count"] >= 5:
        notes.append(f"{int(kpis['Topic_Weakness_Count'])} topics are below 60% accuracy.")
    if kpis["Hard_Question_Accuracy"] < 0.3:
        notes.append("Struggles with hard questions.")
    if kpis["Attempts_Per_Question"] > 2.5:
        notes.append("Needs many attempts per question.")
    if kpis["Score_Trend"] < -0.02:
        notes.append("Scores are trending down.")
    elif kpis["Score_Trend"] > 0.02:
        notes.append("Scores are trending up.")
    return notes


class LevelPredictor:
    def __init__(self):
        self.model = joblib.load(config.MODEL_PATH)

    def predict(self, kpis):
        row = pd.DataFrame([{k: float(kpis[k]) for k in config.FEATURES}], columns=config.FEATURES)
        proba = self.model.predict_proba(row)[0]
        level = int(self.model.classes_[proba.argmax()])
        return {
            "level": level,
            "level_name": config.LEVELS[level],
            "confidence": round(float(proba.max()) * 100, 1),
            "probabilities": {f"{int(c)} {config.LEVELS[int(c)]}": round(float(p) * 100, 1) for c, p in zip(self.model.classes_, proba)},
            "support": SUPPORT[level],
            "observations": kpi_observations(kpis),
        }
