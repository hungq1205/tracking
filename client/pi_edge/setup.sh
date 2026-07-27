#!/usr/bin/env bash
# One-shot setup for the Pi edge server. Run this ON THE PI, from inside
# this directory (client/pi_edge/), after the code has already been copied
# over (e.g. via scp). Does NOT touch raspi-config/camera-interface/reboot
# -- that's a manual, interactive step (see README.md) since it differs by
# OS version and a script shouldn't silently reboot the device.
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Pi edge server setup ==="
echo "--- apt packages (python3-picamera2, python3-venv) ---"
sudo apt update
sudo apt install -y python3-picamera2 python3-venv --no-install-recommends

echo "--- python venv (--system-site-packages so apt's picamera2 is visible inside it) ---"
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Confirm the camera is detected:        libcamera-hello --list-cameras"
echo "  2. Confirm ALSA sees your mic/speaker:    arecord -l ; aplay -l"
echo "  3. List what PortAudio sees:              .venv/bin/python main.py --list-audio-devices"
echo "  4. Run it:                                .venv/bin/python main.py"
echo "  5. Verify from another machine:           python test_client.py --host <this-pi-ip>"
