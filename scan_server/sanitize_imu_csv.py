"""
Stopgap cleanup for imu_data.csv files recorded before the ImuSensor.kt fix
(client/android/app/src/main/java/com/tracking/client/sensors/ImuSensor.kt) —
that bug emitted a reading on every onSensorChanged from *either* the
accelerometer or the gyroscope, interleaving two independently-clocked
timestamp streams and producing ~50% non-monotonic (mostly exact-duplicate)
consecutive timestamps. Anything doing IMU preintegration (ORB-SLAM3, Kalibr)
assumes strictly increasing timestamps, so this drops every row that doesn't
strictly increase over the last kept row, in original file order.

This does not recover the lost samples — it just makes existing recordings
safe to feed into ORB-SLAM3/Kalibr again. Re-record with the fixed app for
full-rate, non-lossy data.

Usage: python sanitize_imu_csv.py path/to/imu_data.csv [more paths...]
Writes a .bak backup alongside each file before overwriting it in place.
"""

import shutil
import sys
from pathlib import Path


def sanitize(path: Path) -> None:
    rows = path.read_text().splitlines()
    kept: list[str] = []
    last_ts: int | None = None
    for line in rows:
        line = line.strip()
        if not line:
            continue
        ts = int(line.split(",", 1)[0])
        if last_ts is not None and ts <= last_ts:
            continue
        kept.append(line)
        last_ts = ts

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text("\n".join(kept) + "\n")
    print(f"{path}: {len(rows)} -> {len(kept)} rows ({len(rows) - len(kept)} dropped), backup at {backup}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        sanitize(Path(arg))
