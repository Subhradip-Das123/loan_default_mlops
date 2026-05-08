import mlflow
import mlflow.sklearn
import joblib
from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

mlflow.set_experiment("Loan Default Prediction")

X, y = preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load("models/best_model.pkl")

with mlflow.start_run():
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.sklearn.log_model(model, "model")

    print("✅ MLflow run logged")
``