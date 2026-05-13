# 02_train_register.py

import os
import json
import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

from dotenv import load_dotenv
import os

load_dotenv()


SUBSCRIPTION_ID = os.getenv('SUBSCRIPTION_ID')
RESOURCE_GROUP = os.getenv('RESOURCE_GROUP')
WORKSPACE_NAME = os.getenv('WORKSPACE_NAME')
LOCATION = os.getenv('LOCATION')
COMPUTE_NAME = os.getenv('COMPUTE_NAME')
VM_SIZE = os.getenv('VM_SIZE')
MODEL_NAME = os.getenv('MODEL_NAME')
ENDPOINT_NAME = os.getenv('ENDPOINT_NAME')
DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME')
DATA_FILE = os.getenv('DATA_FILE')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME')

# =========================================================
# AUTH
# =========================================================

def get_credential():
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
        return credential

    except Exception:
        print("DefaultAzureCredential failed.")
        print("Opening browser login...")

        return InteractiveBrowserCredential()


def get_ml_client():

    credential = get_credential()

    return MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


# =========================================================
# DATA LOADING
# =========================================================

def load_data():

    print(f"Loading dataset: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    if "Status" not in df.columns:
        raise ValueError("Target column 'Status' not found.")

    # remove rows where target missing
    df = df.dropna(subset=["Status"])

    # target to integer
    df["Status"] = df["Status"].astype(int)

    # remove leakage / useless columns
    drop_cols = [
        "Interest_rate_spread",
        "rate_of_interest",
        "Upfront_charges",
        "property_value",
        "LTV",
        "age",
        "submission_of_application",
        "ID",
    ]

    existing_drop_cols = [
        col for col in drop_cols
        if col in df.columns
    ]

    df = df.drop(columns=existing_drop_cols)

    X = df.drop(columns=["Status"])
    y = df["Status"]

    print("Feature shape:", X.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    return X, y


# =========================================================
# PREPROCESSOR
# =========================================================

def build_preprocessor(X):

    numeric_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\nNumeric columns:")
    print(numeric_cols)

    print("\nCategorical columns:")
    print(categorical_cols)

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),

            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_cols
            ),

            (
                "cat",
                categorical_pipeline,
                categorical_cols
            ),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):

        y_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            y_proba
        )

    else:
        roc_auc = 0.0

    metrics = {

        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "roc_auc": roc_auc,
    }

    return metrics, y_pred


# =========================================================
# CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(
    y_test,
    y_pred,
    output_path
):

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred
    )

    plt.title("Confusion Matrix")

    plt.savefig(output_path)

    plt.close()


# =========================================================
# MAIN
# =========================================================

def main():

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/best_model", exist_ok=True)

    # Azure ML Client
    ml_client = get_ml_client()

    # MLflow
    workspace = ml_client.workspaces.get(WORKSPACE_NAME)

    mlflow.set_tracking_uri(
        workspace.mlflow_tracking_uri
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\nMLflow Tracking URI:")
    print(mlflow.get_tracking_uri())

    print("\nExperiment:")
    print(EXPERIMENT_NAME)

    # Load Data
    X, y = load_data()

    # Preprocessor
    preprocessor, numeric_cols, categorical_cols = (
        build_preprocessor(X)
    )

    # Train test split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    )

    # Cross Validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # =====================================================
    # MODELS + GRIDS
    # =====================================================

    model_grids = {

        "logistic_regression": (

            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),

            {
                "model__C": [0.01, 0.1, 1, 10],

                "model__solver": [
                    "liblinear",
                    "lbfgs"
                ],
            },
        ),

        "random_forest": (

            RandomForestClassifier(
                random_state=42,
                class_weight="balanced"
            ),

            {
                "model__n_estimators": [
                    50
                ],

                "model__max_depth": [
                    5,
                    10
                ],

                "model__min_samples_split": [
                    5,
                    10
                ],

                "model__min_samples_leaf": [
                    5,
                    10
                ],

                "model__max_features": [
                    "sqrt",
                    "log2"
                ],
            },
        ),

        "decision_tree": (

            DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced"
            ),

            {
                "model__max_depth": [
                    5,
                    10,
                    
                ],

                "model__min_samples_split": [
                    10,
                    15
                ],

                "model__min_samples_leaf": [
                    5,
                    10
                    
                ],
            },
        ),

        "naive_bayes": (

            GaussianNB(),

            {
                "model__var_smoothing": [
                    1e-9,
                    1e-8
                ],
            },
        ),
    }

    # =====================================================
    # BEST MODEL TRACKING
    # =====================================================

    best_model = None
    best_model_name = None
    best_metrics = None
    best_score = -1
    best_params = None

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    for model_name, (
        estimator,
        param_grid
    ) in model_grids.items():

        print("\n====================================")
        print(f"Tuning model: {model_name}")
        print("====================================")

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    estimator
                ),
            ]
        )

        grid_search = GridSearchCV(

            estimator=pipeline,

            param_grid=param_grid,

            scoring="roc_auc",

            cv=cv,

            n_jobs=-1,

            verbose=2,

            refit=True,
        )

        with mlflow.start_run(run_name=model_name):

            # ---------------------------------------------
            # LOG PARAMETERS
            # ---------------------------------------------

            mlflow.log_param(
                "model_name",
                model_name
            )

            mlflow.log_param(
                "target_column",
                "Status"
            )

            mlflow.log_param(
                "num_features",
                len(numeric_cols)
            )

            mlflow.log_param(
                "cat_features",
                len(categorical_cols)
            )

            mlflow.log_param(
                "train_rows",
                len(X_train)
            )

            mlflow.log_param(
                "test_rows",
                len(X_test)
            )

            mlflow.log_param(
                "cv_folds",
                5
            )

            # ---------------------------------------------
            # TRAIN
            # ---------------------------------------------

            grid_search.fit(
                X_train,
                y_train
            )

            best_pipeline = (
                grid_search.best_estimator_
            )

            # ---------------------------------------------
            # EVALUATE
            # ---------------------------------------------

            metrics, y_pred = evaluate_model(
                best_pipeline,
                X_test,
                y_test
            )

            # ---------------------------------------------
            # LOG METRICS
            # ---------------------------------------------

            mlflow.log_metric(
                "cv_best_roc_auc",
                grid_search.best_score_
            )

            for metric_name, metric_value in metrics.items():

                mlflow.log_metric(
                    metric_name,
                    metric_value
                )

            # ---------------------------------------------
            # LOG BEST PARAMS
            # ---------------------------------------------

            for k, v in grid_search.best_params_.items():

                mlflow.log_param(k, v)

            # ---------------------------------------------
            # SAVE CONFUSION MATRIX
            # ---------------------------------------------

            cm_path = (
                f"outputs/confusion_matrix_{model_name}.png"
            )

            save_confusion_matrix(
                y_test,
                y_pred,
                cm_path
            )

            mlflow.log_artifact(cm_path)

            # ---------------------------------------------
            # SAVE MODEL
            # ---------------------------------------------

            temp_model_path = (
                f"outputs/model_{model_name}.pkl"
            )

            joblib.dump(
                best_pipeline,
                temp_model_path
            )

            mlflow.log_artifact(
                temp_model_path
            )

            # ---------------------------------------------
            # PRINT RESULTS
            # ---------------------------------------------

            print("\nBest CV ROC-AUC:")
            print(grid_search.best_score_)

            print("\nBest Parameters:")
            print(grid_search.best_params_)

            print("\nTest Metrics:")
            print(metrics)

            # ---------------------------------------------
            # STORE BEST MODEL
            # ---------------------------------------------

            if grid_search.best_score_ > best_score:

                best_score = (
                    grid_search.best_score_
                )

                best_model = best_pipeline

                best_model_name = model_name

                best_metrics = metrics

                best_params = (
                    grid_search.best_params_
                )

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print("\n====================================")
    print("FINAL BEST MODEL")
    print("====================================")

    print("Best Model:")
    print(best_model_name)

    print("\nBest CV ROC-AUC:")
    print(best_score)

    print("\nBest Params:")
    print(best_params)

    print("\nBest Metrics:")
    print(best_metrics)

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    best_model_file = (
        "outputs/best_model/model.pkl"
    )

    joblib.dump(
        best_model,
        best_model_file
    )

    # =====================================================
    # SAVE METADATA
    # =====================================================

    metadata = {

        "best_model_name": best_model_name,

        "best_metrics": best_metrics,

        "best_params": best_params,

        "target_column": "Status",

        "data_file": DATA_FILE,
    }

    metadata_file = (
        "outputs/best_model/metadata.json"
    )

    with open(metadata_file, "w") as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # =====================================================
    # REGISTER MODEL
    # =====================================================

    print("\nRegistering model to Azure ML...")

    registered_model = Model(

        name=MODEL_NAME,

        path="outputs/best_model",

        type=AssetTypes.CUSTOM_MODEL,

        description=(
            "Best loan default classification "
            "model with GridSearchCV + MLflow"
        ),

        tags={

            "best_model": best_model_name,

            "roc_auc": str(
                best_metrics["roc_auc"]
            ),

            "accuracy": str(
                best_metrics["accuracy"]
            ),
        },
    )

    registered_model = (
        ml_client.models.create_or_update(
            registered_model
        )
    )

    print("\nModel registered successfully.")

    print("Model Name:")
    print(registered_model.name)

    print("Model Version:")
    print(registered_model.version)


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()
