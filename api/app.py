from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "data/processed_data.csv"

model = joblib.load(MODEL_PATH)

# -----------------------------
# Load feature columns dynamically
# -----------------------------
processed_df = pd.read_csv(DATA_PATH)
FEATURE_COLUMNS = processed_df.drop("loan_status", axis=1).columns.tolist()

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Loan Default Prediction API",
    description="Predict loan default probability",
    version="1.0.0"
)

# -----------------------------
# Input schema
# -----------------------------
class LoanApplication(BaseModel):
    Credit_Score: int
    Applicant_Income: float
    Loan_Amount: float
    Loan_Term: int
    Property_Age: int
    Employment_Type: int
    Loan_Purpose: int


# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}


# -----------------------------
# Prediction endpoint ✅
# -----------------------------
@app.post("/predict")
def predict(data: LoanApplication):

    # Convert input to DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Create full feature DataFrame with zeros
    full_df = pd.DataFrame(0, columns=FEATURE_COLUMNS, index=[0])

    # Fill user-provided fields
    for col in input_df.columns:
        if col in full_df.columns:
            full_df[col] = input_df[col].values

    # Predict
    prediction = model.predict(full_df)[0]
    probs = model.predict_proba(full_df)[0]

    return {
        "loan_default_prediction": int(prediction),
        "probability_no_default": round(float(probs[0]), 6),
        "probability_default": round(float(probs[1]), 6)
    }
