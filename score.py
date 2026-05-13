# score.py

import os
import json
import logging
import joblib
import pandas as pd

model = None

logging.basicConfig(level=logging.INFO)


def init():
    global model

    model_dir = os.environ.get("AZUREML_MODEL_DIR")
    model_name = os.environ.get("MODEL_NAME")
    logging.info(f"AZUREML_MODEL_DIR: {model_dir}")

    for root, dirs, files in os.walk(model_dir):
        logging.info(f"ROOT: {root}")
        logging.info(f"DIRS: {dirs}")
        logging.info(f"FILES: {files}")

    model_path = os.path.join(model_dir, "best_model", "model.pkl")

    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"model.pkl not found at: {model_path}")

    model = joblib.load(model_path)
    logging.info("Model loaded successfully.") 


def run(raw_data):
    try:
        logging.info("Received request.")

        data = json.loads(raw_data)

        # Expected format:
        # {
        #   "data": [
        #       {"col1": value, "col2": value}
        #   ]
        # }

        if "data" not in data:
            return {"error": "Input JSON must contain key: data"}

        input_df = pd.DataFrame(data["data"])

        predictions = model.predict(input_df).tolist()

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[:, 1].tolist()
        else:
            probabilities = [None] * len(predictions)

        response = {
            "predictions": predictions,
            "probability_default": probabilities,
        }

        logging.info(f"Prediction response: {response}")
        return response

    except Exception as e:
        logging.exception("Scoring failed.")
        return {"error": str(e)}
