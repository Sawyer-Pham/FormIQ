from pathlib import Path
import joblib

from preprocess_live import preprocess_live
from features_live import extract_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load saved model and feature order
model = joblib.load(DATA_DIR / "formiq_model.joblib")
feature_columns = joblib.load(DATA_DIR / "feature_columns.joblib")


def predict_trial(filename):
    # Process recording
    processed_df = preprocess_live(filename)

    # Extract 22 features per window
    features_df = extract_features(processed_df)

    if features_df.empty:
        raise ValueError("Recording is too short for prediction.")

    # Match training feature order
    X = features_df[feature_columns]

    # Predict each window
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    print("Window predictions:", predictions)
    print("Class probabilities:", probabilities)

    return predictions, probabilities


# Test using latest recording
if __name__ == "__main__":
    recordings = list((DATA_DIR / "recordings").glob("trial_*.csv"))

    if not recordings:
        raise FileNotFoundError("No recordings found.")

    latest = max(recordings, key=lambda p: p.stat().st_mtime)

    print("Predicting:", latest)
    predict_trial(latest)