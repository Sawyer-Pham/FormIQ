import pandas as pd
import pathlib as path

Base_DIR = path.Path(__file__).resolve().parent.parent
DATA_DIR = Base_DIR / "data"


alldfs = []


for file in DATA_DIR.glob("*_Rarm.csv"):
    
    recording_id = file.stem[0:9]
    forearm_path = DATA_DIR / f"{recording_id}_Rforearm.csv"

    if not forearm_path.exists():
        continue

    arm = pd.read_csv(file)
    forearm = pd.read_csv(forearm_path)

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

    combined["recording_id"] = recording_id
    combined["participant_id"] = recording_id[0:3]
    combined["label"] = int(recording_id[6])

    print(combined.shape)
    

    metadata = ["recording_id", "participant_id", "time"]
    sensor_columns = [
        col for col in combined.columns
        if col not in metadata + ["label"]
    ]

    combined = combined[metadata + sensor_columns + ["label"]]

    alldfs.append(combined)


combined_df = pd.concat(alldfs, axis=0, ignore_index=True)

print(combined_df.head())
print(combined_df.tail())
print("Shape:", combined_df.shape)

