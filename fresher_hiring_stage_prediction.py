# Indian Fresher Job Market Prediction
# Dataset file: data.csv
# Run: python fresher_hiring_stage_prediction.py

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42
DATA_FILE = "data.csv"
OUTPUT_DIR = "project_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# These are profile details that can reasonably be available before the hiring outcome.
PROFILE_FEATURES = [
    "cgpa", "backlogs", "prior_internship", "projects_count",
    "certifications_count", "top_skills", "linkedin_connections"
]
TARGETS = ["hiring_stage", "job_role", "sector", "work_type", "job_location"]


def make_preprocessor(frame):
    numeric = [c for c in PROFILE_FEATURES if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in PROFILE_FEATURES if c in frame.columns and c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ], remainder="drop")


def make_classifier(model):
    return Pipeline([
        ("preprocessing", make_preprocessor(data)),
        ("model", model)
    ])


def show_model_evaluation(frame):
    if "hiring_stage" not in frame.columns:
        print("No 'hiring_stage' column found; skipping model evaluation.")
        return
    usable = frame.dropna(subset=["hiring_stage"]).copy()
    X = usable[PROFILE_FEATURES]
    y = usable["hiring_stage"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced")
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for name, model in models.items():
        pipe = Pipeline([("preprocessing", make_preprocessor(X_train)), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=1)
        cv_rows.append({"model": name, "mean_5_fold_weighted_f1": scores.mean(), "std_5_fold_weighted_f1": scores.std()})
    cv_table = pd.DataFrame(cv_rows).sort_values("mean_5_fold_weighted_f1", ascending=False)
    cv_table.to_csv(os.path.join(OUTPUT_DIR, "cross_validation_comparison.csv"), index=False)
    print("\n--- Cross-validation comparison (profile features only) ---")
    print(cv_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Select using cross-validation only, then evaluate once on the held-out test set.
    best_name = cv_table.iloc[0]["model"]
    final_pipe = Pipeline([("preprocessing", make_preprocessor(X_train)), ("model", models[best_name])])
    final_pipe.fit(X_train, y_train)
    pred = final_pipe.predict(X_test)
    results = {
        "model": best_name,
        "accuracy": accuracy_score(y_test, pred),
        "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0)
    }
    pd.DataFrame([results]).to_csv(os.path.join(OUTPUT_DIR, "final_test_metrics.csv"), index=False)
    print("\nSelected model by 5-fold CV:", best_name)
    print("Held-out test metrics:")
    for k, v in results.items():
        if k != "model": print(f"  {k}: {v:.4f}")
    report = classification_report(y_test, pred, zero_division=0)
    print("\nClassification report:\n", report)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("Selected model: " + best_name + "\n\n" + report)
    labels = sorted(y.unique())
    pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels).to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"))


def fit_prediction_models(frame):
    fitted = {}
    for target in TARGETS:
        if target not in frame.columns:
            continue
        rows = frame.dropna(subset=[target]).copy()
        rows[target] = rows[target].astype(str).str.strip()
        rows = rows[rows[target] != ""]
        if rows[target].nunique() < 2 or len(rows) < 10:
            continue
        pipe = Pipeline([
            ("preprocessing", make_preprocessor(rows[PROFILE_FEATURES])),
            ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1))
        ])
        pipe.fit(rows[PROFILE_FEATURES], rows[target])
        fitted[target] = pipe
    salary_pipe = None
    if "offered_salary_inr" in frame.columns:
        rows = frame.dropna(subset=["offered_salary_inr"]).copy()
        rows = rows[rows["offered_salary_inr"] > 0]
        if len(rows) >= 20:
            salary_pipe = Pipeline([
                ("preprocessing", make_preprocessor(rows[PROFILE_FEATURES])),
                ("model", RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                    random_state=RANDOM_STATE, n_jobs=-1))
            ])
            salary_pipe.fit(rows[PROFILE_FEATURES], rows["offered_salary_inr"])
    return fitted, salary_pipe


def ask_int(prompt):
    while True:
        try:
            value = int(input(prompt).strip())
            if value < 0: print("Enter zero or a positive whole number."); continue
            return value
        except ValueError:
            print("Enter a whole number, for example 3.")


def ask_float(prompt):
    while True:
        try:
            value = float(input(prompt).strip())
            if not 0 <= value <= 10: print("Enter a CGPA between 0 and 10."); continue
            return value
        except ValueError:
            print("Enter a number, for example 8.7.")


def predict_candidate(models, salary_model):
    print("\nEnter Candidate Details")
    print("------------------------------------------------------------")
    cgpa = ask_float("CGPA: ")
    backlogs = ask_int("Backlogs: ")
    internship_answer = input("Prior Internship (Yes/No): ").strip().lower()
    while internship_answer not in {"y", "yes", "n", "no"}:
        internship_answer = input("Please enter Yes or No: ").strip().lower()
    candidate = {
        "cgpa": cgpa,
        "backlogs": backlogs,
        "prior_internship": "Yes" if internship_answer in {"y", "yes"} else "No",
        "projects_count": ask_int("Projects: "),
        "certifications_count": ask_int("Certifications: "),
        "top_skills": input("Top Skill(s), separated by commas (e.g. Python, Java, CSS): ").strip(),
        "linkedin_connections": ask_int("LinkedIn Connections: ")
    }
    # Important: use exactly the original feature columns; each pipeline applies its own fitted encoder.
    candidate_df = pd.DataFrame([{col: candidate.get(col, np.nan) for col in PROFILE_FEATURES}])
    print("============================================================")
    print("            INDIAN FRESHER JOB MARKET PREDICTION            ")
    print("============================================================")
    print("Candidate Profile")
    print("------------------------------------------------------------")
    for label, key in [("CGPA", "cgpa"), ("Backlogs", "backlogs"), ("Prior Internship", "prior_internship"),
                       ("Projects", "projects_count"), ("Certifications", "certifications_count"),
                       ("Top Skill", "top_skills"), ("LinkedIn Connections", "linkedin_connections")]:
        val = candidate[key]
        if key == "prior_internship": val = val.upper()
        print(f"{label:22}: {val}")
    print("------------------------------------------------------------")
    print("                         PREDICTION                         ")
    print("------------------------------------------------------------")
    if salary_model is not None:
        salary = max(0, float(salary_model.predict(candidate_df)[0]))
        print("Predicted Salary")
        print(f"    Rs. {salary:,.0f}")
    for target, label in [("hiring_stage", "Predicted Hiring Stage"), ("job_role", "Predicted Job Role"),
                          ("sector", "Predicted Sector"), ("work_type", "Predicted Work Type"),
                          ("job_location", "Predicted Job Location")]:
        if target in models:
            print(label)
            print("    " + str(models[target].predict(candidate_df)[0]))
    print("============================================================")


def main():
    if not os.path.exists(DATA_FILE):
        print(f"Dataset not found: {DATA_FILE}")
        print("Place data.csv in the same folder as this Python file.")
        return
    frame = pd.read_csv(DATA_FILE)
    missing = [c for c in PROFILE_FEATURES + ["hiring_stage"] if c not in frame.columns]
    if missing:
        print("Dataset is missing required columns:", ", ".join(missing))
        return
    print("Dataset shape:", frame.shape)
    print("Duplicate rows:", int(frame.duplicated().sum()))
    print("\nMissing values (largest counts):")
    print(frame.isna().sum().sort_values(ascending=False).head(10).to_string())
    print("\nTarget distribution: hiring_stage")
    print(frame["hiring_stage"].value_counts(dropna=False).to_string())
    show_model_evaluation(frame)
    print("\nTraining prediction models using candidate profile features...")
    models, salary_model = fit_prediction_models(frame)
    if not models:
        print("Could not train prediction models. Check target columns in data.csv.")
        return
    while True:
        predict_candidate(models, salary_model)
        again = input("\nPredict another? (Yes/No): ").strip().lower()
        if again not in {"y", "yes"}:
            break
    print("\nProject completed. Reports are saved in the project_outputs folder.")
    print("Note: predictions are estimates. Validate with independent real-world data before relying on them.")

if __name__ == "__main__":
    main()
