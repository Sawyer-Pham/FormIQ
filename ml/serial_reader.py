import serial
import time
import csv
from datetime import datetime
from pathlib import Path

# Make connection to ESP
ser = serial.Serial("COM4", 115200, timeout=2)

print("Port opened:", ser.is_open)

time.sleep(3)

duration = 10
start_time = time.monotonic()

try:
    # Open CSV file
    recordings_dir = Path("data/recordings")
    recordings_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = recordings_dir / f"trial_{timestamp}.csv"

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "time",
            "gx_arm", "gy_arm", "gz_arm",
            "ax_arm", "ay_arm", "az_arm",
            "gx_forearm", "gy_forearm", "gz_forearm",
            "ax_forearm", "ay_forearm", "az_forearm"
        ])

        # Read data from ESP to CSV
        while time.monotonic() - start_time < duration:

            line = ser.readline().decode("utf-8").strip()

            if line:
                values = line.split(",")

                if len(values) == 13:
                    try:
                        [float(value) for value in values]
                        writer.writerow(values)
                    except ValueError:
                        continue

except KeyboardInterrupt:
    print("Stopped reading")


finally:
    ser.close()

print(f"Recording saved to: {filename}")
