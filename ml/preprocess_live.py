
import pandas as pd
import numpy as np

def preprocess_live(filename):
    # Load sensor data from the selected trial
    df = pd.read_csv(filename)

    # Gyroscope columns
    gyro_cols = [
        "gx_arm", "gy_arm", "gz_arm",
        "gx_forearm", "gy_forearm", "gz_forearm"
    ]

    # Accelerometer columns
    accel_cols = [
        "ax_arm", "ay_arm", "az_arm",
        "ax_forearm", "ay_forearm", "az_forearm"
    ]

    # Convert gyroscope from rad/s to deg/s
    df[gyro_cols] = np.degrees(df[gyro_cols])

    # Convert accelerometer from m/s² to g
    df[accel_cols] = df[accel_cols] / 9.80665

    # Return processed data for feature extraction
    return df
