import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = Path("data/processed_data.csv")
MODEL_PATH = Path("models/best_model.pkl")

def evaluate():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("✅ Final Model Evaluation")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

if __name__ == "__main__":
    evaluate()