# Pi edge server

Raspberry Pi Zero 2 W-side server: streams camera (JPEG + raw luma) and mic
audio to the Android app over ZMQ (4 fixed media sockets + 1 small control
socket the phone uses to report its current mode), and plays back whatever
audio the app sends. The wire protocol is fixed to match the already-built
Android client
(`client/android/app/src/main/java/com/tracking/client/edge/RemoteEdgeDevice.kt`
+ `AudioMixer.kt`) — see `main.py`'s module docstring for the full
byte-level format and rationale.

## Hardware assumed

- Raspberry Pi Zero 2 W
- Raspberry Pi Camera Module 3 (12MP, Sony IMX708, connected via CSI ribbon)
- A USB audio adapter (Pi Zero 2 W has no analog audio input, and its only
  USB port is the OTG data port) with a 3.5mm earphone-with-mic plugged in

## Install (on the Pi, Raspberry Pi OS Bookworm or later)

### 1. Get onto the Pi

```bash
ssh pi@raspberrypi.local   # or ssh pi@<its-ip>
```

### 2. Enable the camera + update the system

```bash
sudo raspi-config
# Interface Options -> Camera -> Enable (only needed on older OS versions;
# Bookworm+ usually auto-detects the Camera Module 3 via CSI with no toggle)
sudo reboot

sudo apt update && sudo apt full-upgrade -y
```

Confirm the camera is actually detected before installing any Python at all
(the binary is named `rpicam-hello` on current Raspberry Pi OS / Debian
Trixie; older OS versions call it `libcamera-hello` instead — try whichever
exists):

```bash
rpicam-hello --list-cameras   # should list the IMX708 (Camera Module 3)
rpicam-hello -t 2000          # quick 2s preview, confirms the sensor really streams
```

### 3. Install picamera2 + venv tooling (apt, not pip)

```bash
sudo apt install -y python3-picamera2 python3-venv libportaudio2 --no-install-recommends
```

`libportaudio2` is the native shared library the `sounddevice` pip package
binds to -- pip alone doesn't provide it on ARM, and importing sounddevice
fails with `OSError: PortAudio library not found` without it.

### 4. Get this code onto the Pi

If the Pi has network access to your git remote:

```bash
git clone <your-repo-url> ~/tracking
cd ~/tracking/client/pi_edge
```

Otherwise, copy just this folder over from your dev machine:

```bash
# run on your dev machine, not the Pi
scp -r client/pi_edge pi@raspberrypi.local:~/tracking-pi-edge
# then on the Pi: cd ~/tracking-pi-edge
```

### 5. Create the venv and install deps

```bash
python3 -m venv --system-site-packages .venv   # --system-site-packages so the apt-installed picamera2 is visible
source .venv/bin/activate
pip install -r requirements.txt
```

### 6. Plug in the USB audio adapter + earphone, check ALSA sees it

```bash
arecord -l    # should list a capture device (the USB adapter's mic)
aplay -l      # should list a playback device (same adapter, or separate)
python main.py --list-audio-devices   # PortAudio's view of the same devices
```

If nothing shows up: it's almost certainly not plugged into the Pi's single
USB-OTG port (Pi Zero 2 W has no analog audio in/out at all), or the
adapter needs its own driver — check `dmesg | tail -30` right after
plugging it in.

## Run

```bash
python main.py --list-audio-devices     # see what ALSA/PortAudio sees
python main.py                          # binds 0.0.0.0:5601-5604, auto-picks mic/speaker
python main.py --input-device 1 --output-device 1 --log-level DEBUG
python main.py --audio-in-jitter-ms 150   # bump the speaker's prebuffer if playback still stutters
```

Then point the Android app's Settings screen "Use Remote Edge Device"
toggle at this Pi's IP address.

## Verify without the Android app

From the Pi itself, or any other machine on the same network:

```bash
python test_client.py --host <pi-ip-or-127.0.0.1>
```

Prints PASS/FAIL per socket and exits non-zero if anything failed.

## Auto-start on boot

```bash
sudo cp tracking-edge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tracking-edge.service
journalctl -u tracking-edge -f   # follow logs
```

Edit the `WorkingDirectory`/`ExecStart` paths in `tracking-edge.service`
first if this repo isn't checked out at `/home/pi/tracking`.

## Notes / known limitations

- `frame_out` JPEGs are physically rotated (not just tagged) before
  encoding — unlike `luma_out`, this channel carries no rotation metadata
  field at all, and the Android side decodes it straight into a Bitmap with
  no rotation compensation applied anywhere downstream. `--rotation-degrees`
  (default 90) drives both channels from one value; if the image looks
  sideways/upside-down on first real test, try 90/180/270 here rather than
  editing code.
- Mic capture tries to open the USB audio adapter directly at 16kHz; if it
  only offers e.g. 44.1/48kHz (common on cheap dongles), falls back to the
  device's native rate and resamples down in software.
- `luma_out` is only actually needed by AngleTracker during walking/guiding
  (see `ToolDispatcher.feedAngleLumaFrame()` on the Android side) — every
  other mode was paying the full 15fps camera/network cost for data the
  phone just discarded. The phone now reports its current mode over a
  small 5th control socket (port 5605), and `main.py` skips capturing/
  sending `luma_out` unless the mode is walking or guiding. Defaults to
  enabled if nothing's ever reported (an older Android build), so this
  degrades safely rather than silently losing data. Note this does NOT
  reduce the camera's own dual-stream capture cost (picamera2/libcamera
  still produce both streams every request regardless) — only the
  per-frame buffer copy and the network send.
- `audio_in` playback uses an adaptive jitter buffer (`--audio-in-jitter-ms`,
  default 100ms): it waits for that much audio to be queued before the
  speaker starts pulling real audio, absorbing normal network jitter that
  would otherwise sound like stutter. It grows automatically if underruns
  keep happening (logged as `audio_in jitter buffer grew to ...`) and
  shrinks back down after a long stable stretch — a single isolated pause
  (Gemini just stopped talking) does NOT grow it, only repeated underruns
  in a short window do.
- Video (`frame_out`/`luma_out`) queues drop the oldest frame under
  backlog — a fresher frame is strictly better than a stale one. The mic
  queue deliberately does **not** drop under normal load (dropping mid
  utterance corrupts what Gemini hears); it's sized for ~30s and only
  drops-oldest as a last resort if something has been broken for a long
  time.
- No auth/TLS/discovery/control channel, by design — bind to 0.0.0.0 and
  type the IP into the Android app's Settings screen.
- Not verified against real Pi Zero 2 W / Camera Module 3 / ALSA hardware
  from this environment — reviewed for correctness against the actual
  Android-side `RemoteEdgeDevice.kt`/`AudioMixer.kt` wire format (ports,
  header layout, luma sub-header, audio_in's stereo/44.1kHz format were all
  cross-checked against that real code, not assumed), but picamera2's exact
  API surface and real microphone/speaker behavior need a live device to
  confirm.
