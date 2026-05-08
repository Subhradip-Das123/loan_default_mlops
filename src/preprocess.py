import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

DATA_PATH = Path("data/Loan_Default.csv")
OUTPUT_PATH = Path("data/processed_data.csv")

def preprocess_data():
    df = pd.read_csv(DATA_PATH)

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Rename target column
    if "Status" in df.columns:
        df = df.rename(columns={"Status": "loan_status"})

    # Separate numeric & categorical columns SAFELY
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "string"]).columns

    # Fill missing values (NO inplace, NO chaining)
    df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))
    df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna("Unknown"))

    # Encode categorical columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Preprocessing completed")
    print("✅ Output saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    preprocess_data()
