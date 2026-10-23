from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
features_df = pd.read_csv(DATA_DIR / "features.csv")

feature_columns = features_df.columns.drop(
    ["recording_id", "participant_id", "label"]
).tolist()

# Split whole trials, keeping both labels in training and testing.
# With four trials, this gives two training trials and two test trials.
trials = features_df[["recording_id", "label"]].drop_duplicates()
train_trials, test_trials = train_test_split(
    trials, test_size=0.5, stratify=trials["label"], random_state=42
)
train_df = features_df[
    features_df["recording_id"].isin(train_trials["recording_id"])
]
test_df = features_df[
    features_df["recording_id"].isin(test_trials["recording_id"])
]

X_train = train_df[feature_columns]
X_test = test_df[feature_columns]
y_train = train_df["label"]
y_test = test_df["label"]

# Scaling is learned only from the training data.
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

joblib.dump(model, DATA_DIR / "formiq_model.joblib")
joblib.dump(feature_columns, DATA_DIR / "feature_columns.joblib")

predictions = model.predict(X_test)



print("\nModel coefficients (scaled features):")
for feature, coefficient in zip(
    feature_columns, model.named_steps["logisticregression"].coef_[0]
):
    print(f"{feature}: {coefficient:.4f}")


print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")