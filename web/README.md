# Smart Study Monitor website

This is a static, full-screen-friendly website version of the desktop prototype. It uses the browser camera and runs face/eye and phone models in the browser. There is no app server and the camera frames are not uploaded.

## Run it on your PC

1. Open PowerShell in this `web` folder.
2. Start a local web server using the Python environment already installed for the desktop app:

   ```powershell
   ..\.venv\Scripts\python.exe -m http.server 8000
   ```

   If PowerShell says that path is missing, run this from the `SmartStudyMonitor` project folder instead:

   ```powershell
   .\.venv\Scripts\python.exe -m http.server 8000 --directory web
   ```
3. Open `http://localhost:8000` in Chrome or Edge.
4. Choose **Start study**, allow camera access, and wait for the status to say **Detection ready**. The first model load needs an internet connection.
5. Use the square-corners button in the header for full screen. Press `Esc` to leave full screen.
6. Keep the camera at eye level, face the camera, and light both eyes well. Use **Calibrate eyes** while looking at the camera with your eyes open. Eye alerts wait for both eyes to look closed for 0.45 seconds, which filters normal blinks. If it misses a real closure, lower **Eye sensitivity** or shorten **Eye alert delay**.

## Publish it as a public website

The `web` folder contains the whole static site. It is ready to upload to a static hosting provider; it does not require Python, a database, an API key, or a server in production.

Quick path with Netlify Drop:

1. Open [Netlify Drop](https://app.netlify.com/drop).
2. Drag the `web` folder (or `SmartStudyMonitor-Website.zip`) onto the page.
3. Netlify will show a public `netlify.app` URL. Open that URL in Chrome/Edge and allow camera access.
4. Share that HTTPS URL. Anyone visiting it can use the app with their own webcam; their video remains in their own browser.

For a GitHub Pages site, put the contents of `web` at the root of a public repository, then choose **Settings → Pages → Deploy from a branch → main → /(root)** and save. GitHub will provide the public URL after deployment.

Camera access in browsers requires HTTPS, except for `localhost`. Model libraries and model weights are downloaded from their documented hosts on first use. This site does not upload webcam frames, save video, or send session scores to a server. Read [PRIVACY.md](PRIVACY.md) before publishing so you understand the external library/CDN requests.

### If the camera light is on but the page has no preview

The camera light only means the camera is active; the browser page still needs its own live video permission. Press **Start study** and allow camera access for this site. If the page reports that the camera is busy, close the Windows Camera app, Zoom, Teams, and other browser tabs using the camera, then retry. If it reports permission blocked, use the site icon beside the address bar → **Camera** → **Allow**, reload the page, and press **Start study** again. On a public site, open its `https://` address; camera access does not work on an ordinary `http://` site.

## Eye and sound behavior

- The eye detector reads MediaPipe `eyeBlinkLeft` and `eyeBlinkRight` scores. It smooths five readings and waits through short score flickers.
- The default eye level is 0.38 and the alert delay is 0.45 seconds. Both eyes must show a close score. Lower the eye level or shorten the delay if it misses a closure; raise the level or lengthen the delay if open eyes trigger it.
- The live **EYES** label shows `OPEN`, `CLOSING`, or `CLOSED`; the alert counter is the total for this session. When eyes reopen, the eye voice stops.
- A phone alert plays the phone voice. An eye alert plays the eye voice and interrupts a phone voice. If both happen together, the page says **BOTH ALERTS** and the eye voice plays first. The phone voice follows after the eyes open, if the phone is still visible.
- The eye boxes are drawn around the actual face landmarks. If a box is misplaced, improve lighting, reduce glasses glare, face the camera, and calibrate again.
- Phone detection uses COCO-SSD's `cell phone` class at a 30% default confidence and checks about every 0.4 seconds. The lower confidence and faster checks catch more phones, but can mistake similar objects for one. Raise the phone sensitivity level to reduce false alerts.

## Files

- `index.html` — page structure and model entry points
- `styles.css` — responsive white interface and full-screen layout
- `app.js` — webcam, eye/phone detection, calibration, metrics, and voice alerts
- `assets/` — the two supplied voice clips
- `PRIVACY.md` — camera and third-party model privacy notes
