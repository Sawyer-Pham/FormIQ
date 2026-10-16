import pandas as pd
import numpy as np

from preprocess import combined_df


features = []

for recording_id, trial_df in combined_df["recording_id"].unique():
    trial_df = trial_df.reset_index(drop=True)

    # calulate the magnitude of the gyroscope readings for the arm
    trial_df["gyro_mag_arm"] = np.sqrt(
        trial_df["gx_arm"] ** 2
        + trial_df["gy_arm"] ** 2
        + trial_df["gz_arm"] ** 2
    )
    # calulate the magnitude of the gyroscope readings for the forearm
    trial_df["gyro_mag_forearm"] = np.sqrt(
        trial_df["gx_forearm"] ** 2
        + trial_df["gy_forearm"] ** 2
        + trial_df["gz_forearm"] ** 2
    )

    # calulate the magnitude of the accelerometer readings for the arm
    trial_df["accel_mag_arm"] = np.sqrt(
        trial_df["ax_arm"] ** 2
        + trial_df["ay_arm"] ** 2
        + trial_df["az_arm"] ** 2
    )
    # calulate the magnitude of the accelerometer readings for the forearm
    trial_df["accel_mag_forearm"] = np.sqrt(
        trial_df["ax_forearm"] ** 2
        + trial_df["ay_forearm"] ** 2
        + trial_df["az_forearm"] ** 2
    )

    #splitting each trial into chunks of 200 rows 
    for start in range(0, len(trial_df) - 200 + 1, 200):
        window = trial_df.iloc[start:start + 200]

        features = {
            "recording_id": recording_id,
            "participant_id": window["participant_id"].iloc[0],
            "label": window["label"].iloc[0],
            "gyro_mag_arm_mean": window["gyro_mag_arm"].mean(),
            "gyro_mag_forearm_mean": window["gyro_mag_forearm"].mean(),
            "accel_mag_arm_mean": window["accel_mag_arm"].mean(),
            "accel_mag_forearm_mean": window["accel_mag_forearm"].mean(),
        }

        features.append(features)



feature_df =  pd.DataFrame(features)