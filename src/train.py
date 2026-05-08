import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

DATA_PATH = Path("data/processed_data.csv")
MODEL_DIR = Path("models")

def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_prob)
    else:
        roc = None

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc
    }

def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "NaiveBayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=7)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = get_metrics(model, X_test, y_test)

        metrics["model"] = name
        results.append(metrics)

        print(f"✅ Trained {name}")

    results_df = pd.DataFrame(results).sort_values(
        by="f1_score", ascending=False
    )

    MODEL_DIR.mkdir(exist_ok=True)
    results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    # Save best model
    best_model_name = "NaiveBayes"
    best_model = models[best_model_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")

    print("\n✅ Model comparison saved → models/model_comparison.csv")
    print(f"✅ Best model selected → {best_model_name}")
    print(results_df)

if __name__ == "__main__":
    main()