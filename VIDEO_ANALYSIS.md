# Video analysis: Smart Study Monitor

## Recording facts

- Duration: approximately **48.1 seconds**
- Frame size: **476 × 848**, portrait
- Frame rate: **30 fps**
- Content: a desktop screen recording showing a webcam app over VS Code, followed by a social-media outro

## Time-coded observations

| Approx. time | What is visible | What it supports |
|---|---|---|
| 0–6 s | Window title **Smart Study Monitor**; webcam preview of a student reading; `Eye Ratio` label and changing number. VS Code above it shows imports for `cv2`, `cvzone`, `FaceMeshDetector`, `YOLO`, and `pygame`. Tabs named `faudio.mp3`, `alarm.mp3`, and `paudio.mp3` are visible. | Webcam + face-landmark + object-detection + audio stack. The exact eye formula and sound-to-event mapping are not visible. |
| About 9 s | Red **WAKE UP & STUDY!** banner appears while the student looks down with eyes mostly closed. | A drowsiness-style reminder based on an eye signal is shown. The frame alone does not reveal its threshold or duration filter. |
| About 12 s | The face is visible again and the warning is absent in the sampled frame. | The visual warning can clear when the state changes. |
| 14–15 s | A book is raised over the face; red **DON'T COVER YOUR FACE!** banner appears. | There is a separate face-visibility/occlusion warning. |
| 21–23 s | A phone is held in front of the camera. A magenta box encloses it; label reads **Phone detected: 65%** in the sampled 23-second frame; orange **PUT THE PHONE AWAY!** banner is visible. | Phone object detection and a confidence display are visibly working in this moment. |
| 24–36 s | The orange phone warning stays on in the sampled frames; the magenta phone box is not visible in every sample. | The warning state persists across multiple frames. The video does not tell us whether this is temporal smoothing, delayed inference, or another state rule. |
| About 39 s | Phone warning is gone; the student is reading again. | The distraction state eventually clears. |
| About 42 s | Student smiles and gives a thumbs-up; no warning is visible in the sampled frame. | Closing demo moment. |
| 45–48 s | The recording fades into an Instagram-style overlay/outro. | This is social-media branding, not another app feature. |

## Evidence versus interpretation

**Directly visible:** app title, changing eye-ratio display, the three warning texts, phone bounding box and confidence, the listed Python imports, and the audio-file tabs.

**Likely implementation:** MediaPipe face landmarks are used for an eye-opening ratio, and YOLO detects a phone. The visible imports and outputs support this interpretation, but the recording does not show the relevant function bodies, model filename, model class list, exact thresholds, or alert timing logic.

**Not demonstrated:** gaze direction, head-pose tracking, a focus score, study-time dashboard, persistent event history, or a model trained specifically for this project. These can be future additions, not facts about the recorded demo.

## Practical interpretation

The project is a local, real-time reminder loop: estimate whether the eyes are open, detect whether a phone is in camera view, and show an alert. That is a useful computer-vision prototype. It cannot determine whether someone understands the book or is mentally focused, and a phone detector cannot tell whether the detected phone is being used.
