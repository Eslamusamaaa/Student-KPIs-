"""Small Flask demo: enter a student's KPIs, get the level and suggested support.

Run:  python app.py  ->  http://127.0.0.1:5000
"""
import os

from flask import Flask, jsonify, render_template, request

import config
from src.predictor import LevelPredictor

app = Flask(__name__)

try:
    predictor = LevelPredictor()
except FileNotFoundError as exc:
    raise SystemExit(f"{exc}\nTrain the model first:  python run_pipeline.py") from exc

# (min, max) accepted for each KPI
LIMITS = {
    "Overall_Accuracy": (0, 1), "First_Try_Success_Rate": (0, 1), "Average_Time_Per_Question": (0, 3600),
    "Hard_Question_Accuracy": (0, 1), "Attempts_Per_Question": (0, 50), "Hint_Usage_Rate": (0, 1),
    "Hint_Efficiency": (0, 1), "Time_Before_First_Hint": (0, 3600), "Post_Hint_Improvement_Score": (-1, 1),
    "Topic_Weakness_Count": (0, 100), "Score_Trend": (-1, 1),
}
PRESETS = {
    "Beginner": dict(Overall_Accuracy=0.2, First_Try_Success_Rate=0.2, Average_Time_Per_Question=190, Hard_Question_Accuracy=0.1,
                     Attempts_Per_Question=3.2, Hint_Usage_Rate=0.8, Hint_Efficiency=0.25, Time_Before_First_Hint=25,
                     Post_Hint_Improvement_Score=0.1, Topic_Weakness_Count=8, Score_Trend=-0.03),
    "Proficient": dict(Overall_Accuracy=0.85, First_Try_Success_Rate=0.7, Average_Time_Per_Question=70, Hard_Question_Accuracy=0.7,
                       Attempts_Per_Question=1.3, Hint_Usage_Rate=0.15, Hint_Efficiency=0.8, Time_Before_First_Hint=110,
                       Post_Hint_Improvement_Score=0.4, Topic_Weakness_Count=1, Score_Trend=0.06),
}


@app.route("/")
def home():
    return render_template("index.html", features=config.FEATURES, limits=LIMITS, presets=PRESETS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    kpis = {}
    for name, (low, high) in LIMITS.items():
        try:
            value = float(data[name])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": f"'{name}' is required and must be a number."}), 400
        if not low <= value <= high:
            return jsonify({"error": f"'{name}' must be between {low} and {high}."}), 400
        kpis[name] = value
    return jsonify(predictor.predict(kpis))


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
