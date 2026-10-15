import pandas as pd
import pathlib as path

Base_DIR = path.Path(__file__).resolve().parent.parent
DATA_DIR = Base_DIR / "data"

arm = pd.read_csv(DATA_DIR / "A01EFE0_1_Rarm.csv")
forearm = pd.read_csv(DATA_DIR / "A01EFE0_1_Rforearm.csv")

columns = [
    "Time (s)",
    "Gyroscope X (deg/s)",
    "Gyroscope Y (deg/s)",
    "Gyroscope Z (deg/s)",
    "Accelerometer X (g)",
    "Accelerometer Y (g)",
    "Accelerometer Z (g)"
]

arm = arm[columns].copy()
forearm = forearm[columns].copy()

arm["Time (s)"] -= arm["Time (s)"].iloc[0]
forearm["Time (s)"] -= forearm["Time (s)"].iloc[0]

print("Upper arm:")
print(arm.head())

print("Forearm")
print(forearm.head())

arm.columns = [
    "time",
    "gx_arm", "gy_arm", "gz_arm",
    "ax_arm", "ay_arm", "az_arm"
]

forearm.columns = [
    "time",
    "gx_forearm", "gy_forearm", "gz_forearm",
    "ax_forearm", "ay_forearm", "az_forearm"
]

combined = pd.concat(
    [
        arm.reset_index(drop=True),
        forearm.drop(columns=["time"]).reset_index(drop=True)
    ],
    axis=1
)

combined["recording_id"] = "A01EFE0_1"
combined["participant_id"] = "A01"
combined["label"] = 0

print(combined.shape)
print(combined[["recording_id", "participant_id", "time", "label"]].head())

metadata = ["recording_id", "participant_id", "time"]
sensor_columns = [
    col for col in combined.columns
    if col not in metadata + ["label"]
]

combined = combined[metadata + sensor_columns + ["label"]]

print(combined.head())

print(combined.head())
print(combined.shape)