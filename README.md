# Student Performance Classification for Elara Mobile App

## Overview
This project implements a **Random Forest Classifier** to categorize students into **five performance levels** based on their learning KPIs (Key Performance Indicators) in the Elara Mobile App.  
The model analyzes each student’s learning progress and performance patterns to provide adaptive support, such as hints or simplified explanations, according to their level.

**Goals:**  
- Identify students’ performance levels.  
- Measure students’ progress and weaknesses.  
- Enable personalized learning interventions.  

---

## Labels (Student Levels)

| Label | Level Name   | Description                                                                 |
|-------|-------------|-----------------------------------------------------------------------------|
| 1     | Beginner    | New to the topic, low accuracy, high hint usage, low trend (slow/negative progress). |
| 2     | Developing  | Started understanding basics, can solve simple questions, occasional hint usage. |
| 3     | Progressing | Can solve most questions, good accuracy, relies minimally on hints.         |
| 4     | Proficient  | Strong student, high accuracy, capable of handling hard questions confidently. |
| 5     | Outstanding| Excellent student, top performance across all topics, rarely needs help.    |

**Note:** The **Score Trend** shows improvement or decline over time but does not alone determine the label; all KPIs are considered together.

---

## Features (KPIs) and Calculation

| Feature | Meaning | Calculation | Example |
|---------|---------|-------------|---------|
| Overall Accuracy | % of correct answers out of all attempts | `correct_answers / total_attempts` | 50 attempted → 40 correct → 0.8 |
| First-Try Success Rate | % of questions correct on first attempt | `first_attempt_correct / total_attempts` | 50 questions → 30 correct → 0.6 |
| Average Time per Question | Avg time per question, excluding hints | `question_opened_at → answer_submitted_at` minus hint durations | [60s, 90s, 30s] → avg 60s |
| Hard Question Accuracy | Correct % on "Hard" questions | `correct_hard / total_hard_attempted` | 10/15 → 0.667 |
| Attempts per Question (APQ) | Avg attempts per question before correct/abandon | Max attempt per question, average across questions | [1,2,3] → 2 |
| Hint Usage Rate | Fraction of questions where hint requested | `questions_with_hint / total_attempted` | 20/50 → 0.4 |
| Hint Efficiency / Success-after-Hint | Fraction correct after using a hint | `correct_after_hint / total_hint_requests` | 7/10 → 0.7 |
| Time Before First Hint (TBFH) | Avg time before first hint requested | `hint_requested_at - question_opened_at` | [60s,120s,90s] → avg 90s |
| Post-Hint Improvement Score (PHIS) | Improvement after hints | `(correct_after_hint / total_hint) - (correct_before_hint / total_hint)` | Before:3/10, After:6/10 → 0.3 |
| Topic Weakness Count / Accuracy Vector | Number of topics with <60% accuracy | Compute accuracy per topic; count topics <0.6 | 10 topics → 4 weak topics |
| Score Trend / Learning Velocity | Progress over time | Slope of linear regression of `(timestamp, score)` series | Week1:60%, W2:70%, W3:75% → +7.5%/week |

---

## Dataset Construction (300 Records)

1. **Initial Dataset**  
   - 300 student records from Elara Mobile App logs.  
   - Each record includes the 11 KPIs and a Label (1–5).  

2. **Label Assignment**  
   - Based on combined KPIs: accuracy, attempts, hint usage, trend, hard question performance.  
   - Score Trend shows development trajectory but is not used alone.  

3. **Data Augmentation**  
   - Generated additional synthetic records to increase variability (~90,000–100,000 records).  
   - Introduced contradictory / edge cases (~12%) to prevent overfitting.  
   - Added controlled noise to numeric KPIs.  
   - Applied scaling and normalization to features.  
   - Shuffled and combined feature values to generate variations.  
   - Created duplicate variations with slight perturbations.  

4. **Purpose of Augmentation**  
   - Enhance model generalization.  
   - Prevent memorization of patterns.  
   - Simulate diverse real-world student behaviors.  

5. **Purpose of the Model**  
   - Predict student performance level (1–5).  
   - Identify weaknesses and improvement areas.  
   - Enable adaptive interventions (hints, explanations, targeted topics).  

---

## Model Overview

- **Algorithm:** Random Forest Classifier (sklearn)  
- **Input:** 11 KPIs  
- **Output:** Student Level (1–5)  
- **Training Dataset:** Augmented dataset (~90k–100k records)  
- **Purpose:** Predict performance level and guide personalized learning strategies.  

**Notes:**  
- All KPIs contribute to label prediction.  
- Data augmentation ensures robustness and generalization.  
- Random Forest handles numeric features natively; categorical encoding applied for labels.  
- Training and validation accuracy monitored to ensure minimal overfitting.  

---

## References

- Elara Mobile App logs  
- KPIs definitions above  
- Random Forest methodology (sklearn)  
- Data augmentation techniques: noise injection, contradictory examples (~12%), record shuffling, scaling, duplicates
