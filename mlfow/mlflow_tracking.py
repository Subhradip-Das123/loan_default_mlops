import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load data
df = pd.read_csv("data/processed_data.csv")
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Set experiment
mlflow.set_experiment("Loan Default Prediction")

with mlflow.start_run(run_name="NaiveBayes_Run"):

    model = GaussianNB()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Log parameters
    mlflow.log_param("model_type", "NaiveBayes")
    mlflow.log_param("dataset", "Loan Default")
    mlflow.log_param("features", X.shape[1])

    # Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("f1_score", f1_score(y_test, preds))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs))

    # Log & register model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="LoanDefaultModel"
    )

print("✅ MLflow run logged and model registered")

