
import pandas as pd
import numpy as np

def extract_features(df):
    # Make a copy to avoid modifying the original data
    trial_df = df.copy().reset_index(drop=True)

    features_rows = []

    # Calculate gyroscope magnitudes
    trial_df["gyro_mag_arm"] = np.sqrt(
        trial_df["gx_arm"]**2 +
        trial_df["gy_arm"]**2 +
        trial_df["gz_arm"]**2
    )

    trial_df["gyro_mag_forearm"] = np.sqrt(
        trial_df["gx_forearm"]**2 +
        trial_df["gy_forearm"]**2 +
        trial_df["gz_forearm"]**2
    )

    # Calculate accelerometer magnitudes
    trial_df["accel_mag_arm"] = np.sqrt(
        trial_df["ax_arm"]**2 +
        trial_df["ay_arm"]**2 +
        trial_df["az_arm"]**2
    )

    trial_df["accel_mag_forearm"] = np.sqrt(
        trial_df["ax_forearm"]**2 +
        trial_df["ay_forearm"]**2 +
        trial_df["az_forearm"]**2
    )

    # Calculate arm and forearm differences
    for channel in ["gx", "gy", "gz", "ax", "ay", "az"]:
        trial_df[f"{channel}_diff"] = (
            trial_df[f"{channel}_forearm"] -
            trial_df[f"{channel}_arm"]
        )

    # Split recording into 200-sample windows
    for start in range(0, len(trial_df) - 200 + 1, 200):
        window = trial_df.iloc[start:start + 200]

        features = {
            "gyro_mag_arm_mean": window["gyro_mag_arm"].mean(),
            "gyro_mag_forearm_mean": window["gyro_mag_forearm"].mean(),
            "accel_mag_arm_mean": window["accel_mag_arm"].mean(),
            "accel_mag_forearm_mean": window["accel_mag_forearm"].mean(),
        }

        # Calculate difference statistics
        for channel in ["gx", "gy", "gz", "ax", "ay", "az"]:
            values = window[f"{channel}_diff"]

            features[f"{channel}_diff_mean"] = values.mean()
            features[f"{channel}_diff_std"] = values.std()
            features[f"{channel}_diff_abs_mean"] = values.abs().mean()

        features_rows.append(features)

    # Return features as a DataFrame
    feature_df = pd.DataFrame(features_rows)

    return feature_df
