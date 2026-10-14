# Privacy notes

## Camera and study session

The camera is opened only after the visitor presses **Start study** and grants browser permission. Eye and phone inference run in the visitor's browser. This site does not send camera frames, microphone audio, detected objects, or study-session metrics to a Smart Study Monitor server. It does not save the webcam stream or session statistics.

The page loads its JavaScript libraries from jsDelivr and model files from Google-hosted URLs. MediaPipe documents that task input is processed on-device, while its APIs may send library performance and usage metrics to Google: [MediaPipe Tasks privacy notice](https://developers.google.com/edge/mediapipe/solutions/tasks#mediapipe_tasks_privacy_notice).

The site includes two voice clips in its public `assets` folder. Those files are served to every visitor's browser so the local reminders can play.

## Hosting

If you publish the site, your hosting provider may keep ordinary web-server records such as visitor IP address, requested files, and request time. Review the provider's privacy terms. This page does not include an account system, analytics pixel, advertising, or a database.

## Permissions

Camera permission can be revoked in the browser's site settings. The camera stream is stopped when the visitor ends the session or leaves the page.
