import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ===============================
# 1) LOAD DATA
# ===============================
data_path = r"C:\Users\Laptop World\Desktop\Student KPIs For Elara\Student_KPIs_Elara_Augmented.csv"
df = pd.read_csv(data_path)

print("Data Loaded Successfully!")
print(df.head())


# ===============================
# 2) LABEL ENCODING (If Needed)
# ===============================
label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])

# Save the encoder
encoder_path = r"C:\Users\Laptop World\Desktop\Student KPIs For Elara\label_encoder.pkl"
joblib.dump(label_encoder, encoder_path)

print("Label Encoding Done & Saved!")


# ===============================
# 3) FEATURES + TARGET
# ===============================
feature_cols = [
    'Overall_Accuracy', 'First_Try_Success_Rate', 'Average_Time_Per_Question',
    'Hard_Question_Accuracy', 'Attempts_Per_Question', 'Hint_Usage_Rate',
    'Hint_Efficiency', 'Time_Before_First_Hint', 'Post_Hint_Improvement_Score',
    'Topic_Weakness_Count', 'Score_Trend'
]

X = df[feature_cols]
y = df["Label"]


# ===============================
# 4) TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ===============================
# 5) TRAIN RANDOM FOREST
# ===============================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\nModel Training Done!")


# ===============================
# 6) PREDICT
# ===============================
y_pred = model.predict(X_test)


# ===============================
# 7) ACCURACY
# ===============================
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc:.4f}")


# ===============================
# 8) CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ===============================
# 9) CLASSIFICATION REPORT
# ===============================
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ===============================
# 10) SAVE MODEL
# ===============================
model_path = r"C:\Users\Laptop World\Desktop\Student KPIs For Elara\student_level_rf_model.pkl"
joblib.dump(model, model_path)

print("\nModel Saved Successfully!")
print("Done!")
