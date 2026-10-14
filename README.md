# Smart Study Monitor

A small local webcam project inspired by the supplied 48-second demo. It displays a live camera view and gives reminders when it sees prolonged eye closure, a phone in the camera frame, or no visible face.

## Full-screen website version

The responsive browser app is in the [`web`](web) folder. For the public website build, use [`web/README.md`](web/README.md), or double-click `run-website.bat` and open `http://localhost:8000` to preview it on this PC. The site uses browser-based face and phone models, includes the two supplied voice clips, and has a privacy notice. For publishing, upload `SmartStudyMonitor-Website.zip` to Netlify Drop or publish the contents of `web` with GitHub Pages.

## What the video actually shows

- A desktop recording with a VS Code window behind a webcam app titled **Smart Study Monitor**.
- The visible source imports OpenCV, cvzone's `FaceMeshDetector`, Ultralytics `YOLO`, and pygame. Tabs for three MP3 files are visible, but the frames do not show enough source code to prove which sound plays for which event.
- An eye-ratio value changes while the student reads. Around 9 seconds, **WAKE UP & STUDY!** appears.
- Around 14–15 seconds, hiding the face behind a book triggers **DON'T COVER YOUR FACE!**.
- Around 22–23 seconds, YOLO draws a phone box and a confidence label (65% in the sampled frame), with **PUT THE PHONE AWAY!**.
- The phone warning remains visible for several later seconds and clears by roughly 39 seconds. That persistence could come from repeated detections, temporal smoothing, or inference delay; the recording alone does not identify which.
- The final seconds show the student smiling/thumbs-up and then an Instagram-style outro.

The video demonstrates visual signals and feedback. It does not establish that the app can measure comprehension or a person's true attention.

## How this version works

1. OpenCV reads webcam frames and mirrors the view.
2. cvzone/MediaPipe returns facial landmarks. The program computes a normalized Eye Aspect Ratio (EAR) from six landmarks around each eye. Both eyes are averaged.
3. The eye panel shows `OPEN`, `CLOSING`, or `CLOSED - ALERT`, both per-eye EAR values, and a timer/progress bar for the close duration. If EAR stays below the configured threshold for `--closed-seconds`, the app shows a drowsiness reminder.
4. A pretrained Ultralytics YOLO model checks for the COCO `cell phone` class every 0.35 seconds by default. A detection produces a box and confidence label.
5. If no face is detected for the configured time, the app shows the face-visibility reminder.
6. pygame plays the supplied eye-closed voice for drowsiness and the phone-detected voice for phone alerts. Face-visibility alerts use a short beep; if a voice file cannot load, that alert falls back to the beep. A voice already playing is allowed to finish if the detection briefly flickers off. `--no-sound` disables all audio.
7. Pressing Q closes the app and prints a session summary. Frames are not saved.

Alert priority is phone, face visibility, then prolonged eye closure. Phone detection means the model sees a phone in the camera view; it cannot establish whether the person is using it.

## Requirements

- Windows 10/11, 64-bit
- Python 3.10–3.12 (64-bit). Python 3.10.11 is already installed on this PC and works with this project.
- A working webcam
- Internet for the initial package installation; the YOLO model is bundled in the project

The project uses cvzone's Face Mesh wrapper, which depends on MediaPipe's legacy `solutions` API. MediaPipe 0.10.31 and later removed that legacy API, so this project pins MediaPipe 0.10.21 for compatibility. The matching OpenCV 4.11 packages keep NumPy versions compatible across MediaPipe and Ultralytics.

## Setup on Windows PowerShell

Open PowerShell in the `SmartStudyMonitor` folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If PowerShell blocks virtual-environment activation, or `py` cannot find the installed Python, run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

After setup is complete, you can also double-click `run.bat` in this folder to start the app.

The project folder and ZIP include `yolo26n.pt`, so the phone-detection model is ready locally. If that file is missing, Ultralytics may try to download it on launch. The model and Python packages run locally after installation.

## Controls and calibration

- `Q`: quit
- `C`: calibrate while facing the camera with both eyes naturally open. The threshold is set to 65% of the current smoothed EAR; calibration no longer depends on the previous threshold.
- `D`: toggle eye-landmark debug points on/off
- `[`: lower the eye threshold by 0.01 (stricter closed-eye detection)
- `]`: raise the eye threshold by 0.01 (more sensitive detection)
- `--camera 1`: choose a different camera index
- `--no-sound`: disable voice clips and beeps

Example with a different webcam and a more conservative phone threshold:

```powershell
python app.py --camera 1 --phone-confidence 0.55 --closed-seconds 1.1
```

Useful settings:

| Option | Default | Meaning |
|---|---:|---|
| `--eye-threshold` | `0.20` | EAR below this is treated as closed |
| `--closed-seconds` | `0.9` | Continuous low EAR required for the sleepy alert |
| `--face-lost-seconds` | `1.5` | Missing-face duration before the face alert |
| `--phone-confidence` | `0.45` | Minimum phone detection confidence |
| `--phone-interval` | `0.35` | Time between YOLO inference passes |

## Suggested build and tuning procedure

1. First confirm the webcam opens and the live preview is stable. Press D to show the eye landmarks; points should sit on the eye corners and lids, not the glasses frame.
2. Sit at the distance and angle you expect during study. Keep the face well-lit and unobstructed.
3. Press C with eyes open. The display shows the averaged EAR and separate left/right readings; it uses a five-frame median to reduce jitter. Blink normally and check that short blinks do not trigger an alert.
4. Close your eyes for about one second. Wait until the panel says `CLOSED - ALERT`; the eye-close voice starts at that point. Adjust with `[` and `]` if needed. At launch, the console prints whether `eye-closed.mpeg` and `phone-detected.mpeg` loaded successfully, and prints when an alert voice starts.
5. Put a phone inside the camera view and check the box/confidence. Raise `--phone-confidence` if books or screens trigger false detections; lower it if a real phone is missed.
6. Cover your face briefly and check the face-visibility warning.
7. Try `--phone-interval 0.5` on a slower CPU. A larger interval reduces compute use but can delay phone warnings.

## Troubleshooting

- **Camera will not open:** close Zoom/Teams/Camera, allow desktop apps to use the camera in Windows Privacy settings, then try `--camera 1`.
- **No phone class in model:** run with the default `yolo26n.pt` model. A custom model must include a class named `cell phone`, `mobile phone`, or `phone`.
- **Slow video:** close other heavy apps, reduce the camera resolution in `app.py`, or increase `--phone-interval`.
- **No alert audio:** wait for `EYE STATUS: CLOSED - ALERT` (the eye must stay below the displayed limit for the configured duration), check the console for `Loaded alert voice: eye-closed.mpeg` and `Playing eye-closed voice alert`, confirm the Windows output device, and ensure `--no-sound` was not used. If the loaded message is missing, check that `assets\eye-closed.mpeg` exists; if the playing message is missing, the eye-close condition has not triggered or a higher-priority phone/face alert is active.
- **Eye reading looks wrong or unstable:** the panel shows EAR as a decimal (for example, `0.26`) plus separate left/right readings. Press D to check that landmarks sit on your eyelids and eye corners. Add light in front of your face, reduce glare on glasses, keep the camera near eye level, and press C while looking at the camera with both eyes open. The progress bar shows time below the threshold; tune with `[` and `]` if needed.

## Limits and next steps

This is a prototype reminder, not a reliable measure of study focus, sleepiness, or learning. Looking at a book does not prove reading or comprehension, and the app does not estimate gaze direction. A phone outside the camera view cannot be detected; objects that resemble a phone can cause false alerts. Face absence can mean the student moved away or the camera is blocked, so the wording is only a reminder.

For a next version, add a local CSV/SQLite session log with event durations, a dashboard, head-pose or gaze signals, and a short calibration sequence that records several open-eye frames. Test thresholds in the room and with the camera setup where the app will be used.

