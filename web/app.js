const $ = (id) => document.getElementById(id);
const ui = {
  shell: $('appShell'), video: $('cameraVideo'), canvas: $('overlayCanvas'), stage: $('cameraStage'),
  placeholder: $('cameraPlaceholder'), start: $('startButton'), placeholderStart: $('placeholderStart'),
  stop: $('stopButton'), cameraSelect: $('cameraSelect'), calibrate: $('calibrateButton'),
  fullscreen: $('fullscreenButton'), modelStatus: $('modelStatus'), live: $('liveIndicator'),
  cameraLabel: $('cameraLabel'), stateIcon: $('cameraStateIcon'), stateTitle: $('cameraStateTitle'),
  stateSubtitle: $('cameraStateSubtitle'), clock: $('sessionClock'), badge: $('statusBadge'),
  score: $('focusScore'), meter: $('focusMeter'), headline: $('focusHeadline'), description: $('focusDescription'),
  eyeScore: $('eyeScore'), eyeState: $('eyeStatusText'), eyeFill: $('eyeMeterFill'), leftEye: $('leftEyeState'),
  rightEye: $('rightEyeState'), eyeThresholdLabel: $('eyeThresholdLabel'), eyeTime: $('eyeTimeValue'),
  eyeAlerts: $('eyeAlertCount'), phoneAlerts: $('phoneAlertCount'),
  eyeSensitivity: $('eyeSensitivity'), eyeSensitivityValue: $('eyeSensitivityValue'),
  closedDuration: $('closedDuration'), closedDurationValue: $('closedDurationValue'),
  phoneSensitivity: $('phoneSensitivity'), phoneSensitivityValue: $('phoneSensitivityValue'),
  audioToggle: $('audioToggle'), eyeAudio: $('eyeAudio'), phoneAudio: $('phoneAudio'), toast: $('toast'),
};
const ctx = ui.canvas.getContext('2d');

const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm';
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const LEFT_EYE_IDS = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE_IDS = [33, 160, 158, 133, 153, 144];
// Smooth short blinks and score flickers so an open eye does not trigger a voice.
const MEDIAN_WINDOW = 5;
const CLOSED_GRACE_MS = 160;
const PHONE_HOLD_MS = 900;

let stream = null;
let faceTask = null;
let phoneModel = null;
let running = false;
let modelLoading = false;
let lastVideoTime = -1;
let lastFaceInference = 0;
let phoneInferenceRunning = false;
let lastPhoneInference = 0;
let lastPhoneAt = 0;
let lastPhoneBoxes = [];
let phoneIsActive = false;
let phoneVoicePending = false;
let eyeThreshold = Number(ui.eyeSensitivity.value);
let closedDuration = Number(ui.closedDuration.value);
let phoneConfidence = Number(ui.phoneSensitivity.value);
let closedSince = null;
let lastClosedAt = 0;
let eyeAlertFired = false;
let currentBlend = null;
let eyeHistory = { left: [], right: [] };
let calibration = null;
let sessionStartedAt = 0;
let lastTick = 0;
let focusedSeconds = 0;
let eyeAlertCount = 0;
let phoneAlertCount = 0;
let lastAlertAudio = null;
let lastEyeVoiceAt = 0;
let toastTimer = 0;
let audioContext = null;
let voiceBuffers = {};
let activeVoiceSource = null;
let activeVoiceType = null;
let displayedState = null;

function showToast(message) {
  ui.toast.textContent = message;
  ui.toast.classList.add('show');
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => ui.toast.classList.remove('show'), 3300);
}

function setModelStatus(message, kind = '') {
  ui.modelStatus.textContent = message;
  ui.modelStatus.className = `model-status ${kind}`.trim();
}

function formatClock(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600).toString().padStart(2, '0');
  const minutes = Math.floor((total % 3600) / 60).toString().padStart(2, '0');
  const remainder = (total % 60).toString().padStart(2, '0');
  return `${hours}:${minutes}:${remainder}`;
}

function setCurrentState(state) {
  if (state === displayedState) return;
  displayedState = state;
  const values = {
    READY: { badge: 'READY', icon: '✦', title: 'Ready to start', subtitle: '', headline: 'Ready to start', description: 'Start studying when you are ready.', badgeClass: 'status-idle' },
    FOCUSED: { badge: 'STUDYING', icon: '✦', title: 'No alert right now', subtitle: '', headline: 'Keep going', description: 'Eyes are open. No phone is detected.', badgeClass: 'status-focused' },
    EYES: { badge: 'EYES CLOSED', icon: '◌', title: 'Eyes closed', subtitle: 'Eye voice', headline: 'Open your eyes', description: 'Your eyes stayed closed. Open them to continue.', badgeClass: 'status-alert' },
    PHONE: { badge: 'PHONE SEEN', icon: '▣', title: 'Phone detected', subtitle: 'Phone voice', headline: 'Put your phone away', description: 'Move the phone out of camera view.', badgeClass: 'status-alert' },
    BOTH: { badge: 'BOTH ALERTS', icon: '!', title: 'Eyes closed + phone detected', subtitle: 'Eye voice first · phone voice next', headline: 'Two reminders active', description: 'Open your eyes. The phone reminder plays next if the phone is still visible.', badgeClass: 'status-alert' },
    NO_FACE: { badge: 'FACE NOT SEEN', icon: '◉', title: 'Face not in view', subtitle: '', headline: 'Move into view', description: 'Sit where the camera can see your face and eyes.', badgeClass: 'status-alert' },
    MODELS: {
      badge: 'LOADING', icon: '✳',
      title: faceTask ? 'Eyes ready' : phoneModel ? 'Phone ready' : 'Getting ready',
      subtitle: faceTask ? 'Phone detector is loading…' : phoneModel ? 'Eye detector is loading…' : 'First load may take a little while',
      headline: faceTask ? 'Eye alerts are active' : phoneModel ? 'Phone alerts are active' : 'Preparing detection',
      description: faceTask ? 'Phone reminders will start when its model finishes loading.' : phoneModel ? 'Eye reminders will start when its model finishes loading.' : 'Keep this page open while the eye and phone models download.',
      badgeClass: 'status-idle',
    },
  }[state];
  if (!values) return;
  ui.badge.className = `status-badge ${values.badgeClass}`;
  ui.badge.innerHTML = `<i></i>${values.badge}`;
  ui.stateIcon.textContent = values.icon;
  ui.stateTitle.textContent = values.title;
  ui.stateSubtitle.textContent = values.subtitle;
  ui.headline.textContent = values.headline;
  ui.description.textContent = values.description;
}

function eyeScoreMap(result) {
  const categories = result?.faceBlendshapes?.[0]?.categories;
  if (!categories) return null;
  const map = new Map(categories.map((item) => [item.categoryName, item.score]));
  const left = map.get('eyeBlinkLeft');
  const right = map.get('eyeBlinkRight');
  if (!Number.isFinite(left) || !Number.isFinite(right)) return null;
  return { left: clamp(left, 0, 1), right: clamp(right, 0, 1), average: (left + right) / 2 };
}

function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }

function smoothScore(side, score) {
  const values = eyeHistory[side];
  values.push(score);
  if (values.length > MEDIAN_WINDOW) values.shift();
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.floor(ordered.length / 2)];
}

function playQueuedPhoneVoice() {
  if (!phoneVoicePending) return;
  if (!phoneIsActive) {
    phoneVoicePending = false;
    return;
  }
  if (eyeAlertFired || !ui.audioToggle.checked) return;
  phoneVoicePending = false;
  playVoice('phone');
}

function onVoiceFinished(which) {
  if (activeVoiceType === which) activeVoiceType = null;
  if (which === 'eye') playQueuedPhoneVoice();
}

function stopVoice(which) {
  if (activeVoiceType !== which) return;
  if (activeVoiceSource) {
    const source = activeVoiceSource;
    activeVoiceSource = null;
    try { source.stop(); } catch { /* Already finished. */ }
  }
  if (lastAlertAudio) {
    lastAlertAudio.pause();
    lastAlertAudio.currentTime = 0;
    lastAlertAudio = null;
  }
  activeVoiceType = null;
}

function playVoice(which) {
  if (!ui.audioToggle.checked) return;
  const audio = which === 'eye' ? ui.eyeAudio : ui.phoneAudio;
  const currentIsPlaying = activeVoiceSource || (lastAlertAudio && !lastAlertAudio.paused);
  if (currentIsPlaying) {
    if (activeVoiceType === which || lastAlertAudio === audio) return;
    // Eye closure has priority; queue the phone voice until the eye voice finishes.
    if (which !== 'eye') {
      if (activeVoiceType === 'eye') phoneVoicePending = true;
      return;
    }
    if (phoneIsActive) phoneVoicePending = true;
    if (activeVoiceSource) {
      try { activeVoiceSource.stop(); } catch { /* Already finished. */ }
      activeVoiceSource = null;
    }
    if (lastAlertAudio) lastAlertAudio.pause();
  }
  if (which === 'eye') lastEyeVoiceAt = performance.now();
  if (audioContext?.state === 'running' && voiceBuffers[which]) {
    const source = audioContext.createBufferSource();
    source.buffer = voiceBuffers[which];
    source.connect(audioContext.destination);
    source.onended = () => {
      if (activeVoiceSource === source) {
        activeVoiceSource = null;
        onVoiceFinished(which);
      }
    };
    activeVoiceSource = source;
    activeVoiceType = which;
    source.start();
  } else {
    audio.currentTime = 0;
    lastAlertAudio = audio;
    activeVoiceType = which;
    audio.play().catch(() => {
      if (lastAlertAudio === audio) lastAlertAudio = null;
      if (activeVoiceType === which) activeVoiceType = null;
      showToast('Sound allow nahi hua. Browser tab ka sound unmute karein.');
    });
  }
}

function resetAudio() {
  phoneVoicePending = false;
  if (activeVoiceSource) {
    try { activeVoiceSource.stop(); } catch { /* Already finished. */ }
  }
  activeVoiceSource = null;
  activeVoiceType = null;
  for (const audio of [ui.eyeAudio, ui.phoneAudio]) {
    audio.pause();
    audio.currentTime = 0;
  }
  lastAlertAudio = null;
}

ui.eyeAudio.addEventListener('ended', () => {
  if (lastAlertAudio === ui.eyeAudio) lastAlertAudio = null;
  onVoiceFinished('eye');
});
ui.phoneAudio.addEventListener('ended', () => {
  if (lastAlertAudio === ui.phoneAudio) lastAlertAudio = null;
  onVoiceFinished('phone');
});

async function loadVoiceBuffers() {
  if (!audioContext || !audioContext.decodeAudioData) return;
  try {
    const [eyeResponse, phoneResponse] = await Promise.all([
      fetch(ui.eyeAudio.getAttribute('src')),
      fetch(ui.phoneAudio.getAttribute('src')),
    ]);
    if (!eyeResponse.ok || !phoneResponse.ok) throw new Error('Audio files are unavailable.');
    const [eyeData, phoneData] = await Promise.all([eyeResponse.arrayBuffer(), phoneResponse.arrayBuffer()]);
    [voiceBuffers.eye, voiceBuffers.phone] = await Promise.all([
      audioContext.decodeAudioData(eyeData),
      audioContext.decodeAudioData(phoneData),
    ]);
  } catch (error) {
    console.warn('Web Audio clip load failed; using HTML audio fallback:', error);
    voiceBuffers = {};
  }
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    const error = new Error('Camera access needs Chrome/Edge on HTTPS or localhost.');
    error.name = 'CameraApiUnavailableError';
    throw error;
  }
  const deviceId = ui.cameraSelect.value;
  const videoOptions = deviceId
    ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
    : { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } };
  stream = await navigator.mediaDevices.getUserMedia({ video: videoOptions, audio: false });
  ui.video.srcObject = stream;
  await ui.video.play();
  await new Promise((resolve) => {
    if (ui.video.videoWidth) return resolve();
    ui.video.addEventListener('loadedmetadata', resolve, { once: true });
  });
  ui.canvas.width = ui.video.videoWidth;
  ui.canvas.height = ui.video.videoHeight;
  ui.stage.classList.add('has-video');
  ui.live.classList.add('is-live');
  ui.cameraLabel.textContent = 'CAMERA LIVE';
  ui.calibrate.disabled = false;
  await refreshCameras();
}

function explainCameraError(error) {
  const messages = {
    CameraApiUnavailableError: 'Camera access ke liye Chrome/Edge mein HTTPS ya localhost page kholiye.',
    NotAllowedError: 'Browser ne website ki camera permission block ki. Address-bar ke site settings mein Camera ko Allow karke reload karein.',
    PermissionDeniedError: 'Browser ne website ki camera permission block ki. Address-bar ke site settings mein Camera ko Allow karke reload karein.',
    NotFoundError: 'Koi camera nahi mila. Camera connect karke ya camera list se doosra camera chun kar dobara try karein.',
    DevicesNotFoundError: 'Koi camera nahi mila. Camera connect karke ya camera list se doosra camera chun kar dobara try karein.',
    NotReadableError: 'Camera kisi aur app ya browser tab mein busy ho sakta hai. Camera, Zoom, Teams aur doosre tabs band karke dobara try karein.',
    TrackStartError: 'Camera start nahi ho paya; woh kisi aur app mein busy ho sakta hai. Camera, Zoom, Teams aur doosre tabs band karke dobara try karein.',
    AbortError: 'Camera start ruk gaya. Doosre camera apps/tabs band karke dobara try karein.',
    OverconstrainedError: 'Chuna hua camera available nahi. Camera list se Default/available camera chun kar dobara try karein.',
    SecurityError: 'Browser camera access rok raha hai. Site ko HTTPS ya localhost par kholiye.',
  };
  return messages[error?.name] || error?.message || 'Camera nahi khula. Site permission aur camera selection check karke dobara try karein.';
}

async function refreshCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter((item) => item.kind === 'videoinput');
    const activeId = stream?.getVideoTracks()[0]?.getSettings().deviceId || ui.cameraSelect.value;
    ui.cameraSelect.replaceChildren();
    for (const [index, camera] of cameras.entries()) {
      const option = document.createElement('option');
      option.value = camera.deviceId;
      option.textContent = camera.label || `Camera ${index + 1}`;
      ui.cameraSelect.append(option);
    }
    if (activeId && cameras.some((item) => item.deviceId === activeId)) ui.cameraSelect.value = activeId;
  } catch { /* Camera selection remains on the browser default. */ }
}

async function createFaceTask() {
  const { FaceLandmarker, FilesetResolver } = await import(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs'
  );
  const files = await FilesetResolver.forVisionTasks(WASM_ROOT);
  let lastError;
  for (const delegate of ['GPU', 'CPU']) {
    try {
      return await FaceLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate },
        runningMode: 'VIDEO', numFaces: 1,
        minFaceDetectionConfidence: 0.35,
        minFacePresenceConfidence: 0.35,
        minTrackingConfidence: 0.4,
        outputFaceBlendshapes: true,
      });
    } catch (error) { lastError = error; }
  }
  throw lastError || new Error('Face model could not start.');
}

async function createPhoneModel() {
  if (!window.tf || !window.cocoSsd) throw new Error('Phone detection library did not load.');
  const webglReady = await window.tf.setBackend('webgl').catch(() => false);
  if (!webglReady) await window.tf.setBackend('cpu');
  await window.tf.ready();
  return window.cocoSsd.load({ base: 'lite_mobilenet_v2' });
}

async function loadModels() {
  if (modelLoading) return;
  if (faceTask && phoneModel) {
    setModelStatus('Detection ready', 'good');
    setCurrentState('FOCUSED');
    return;
  }
  modelLoading = true;
  setCurrentState('MODELS');
  setModelStatus(faceTask || phoneModel ? 'Loading remaining detector…' : 'First load: downloading detection models…');
  const updateProgress = () => {
    if (faceTask && phoneModel) {
      setModelStatus('Detection ready', 'good');
    } else if (faceTask) {
      setModelStatus('Eyes ready · phone detector loading…');
    } else if (phoneModel) {
      setModelStatus('Phone ready · eye detector loading…');
    }
    if (displayedState === 'MODELS') {
      displayedState = null;
      setCurrentState('MODELS');
    }
  };
  const facePromise = faceTask
    ? Promise.resolve(faceTask)
    : createFaceTask().then((model) => { faceTask = model; updateProgress(); return model; });
  const phonePromise = phoneModel
    ? Promise.resolve(phoneModel)
    : createPhoneModel().then((model) => { phoneModel = model; updateProgress(); return model; });
  const [faceResult, phoneResult] = await Promise.allSettled([
    facePromise,
    phonePromise,
  ]);
  if (faceResult.status === 'fulfilled') faceTask = faceResult.value;
  if (phoneResult.status === 'fulfilled') phoneModel = phoneResult.value;
  modelLoading = false;
  if (faceTask && phoneModel) setModelStatus('Detection ready', 'good');
  else if (faceTask) setModelStatus('Eye alerts ready · phone unavailable', 'bad');
  else if (phoneModel) setModelStatus('Phone alerts ready · eyes unavailable', 'bad');
  else setModelStatus('Detection could not load', 'bad');
  if (!faceTask || !phoneModel) {
    const detail = [faceResult, phoneResult].filter((item) => item.status === 'rejected').map((item) => item.reason?.message).join(' ');
    showToast(`${!faceTask ? 'Eye' : 'Phone'} model load nahi hua. Internet connection check karein.`);
    console.warn('Study Monitor model load:', detail);
  }
  if (running) setCurrentState('FOCUSED');
}

function beginSession() {
  if (running) return;
  running = true;
  sessionStartedAt = performance.now();
  lastTick = sessionStartedAt;
  focusedSeconds = 0;
  eyeAlertCount = 0;
  phoneAlertCount = 0;
  phoneVoicePending = false;
  closedSince = null;
  lastClosedAt = 0;
  eyeAlertFired = false;
  currentBlend = null;
  eyeHistory = { left: [], right: [] };
  lastPhoneAt = 0;
  phoneIsActive = false;
  lastPhoneBoxes = [];
  ui.eyeAlerts.textContent = '0';
  ui.phoneAlerts.textContent = '0';
  ui.score.textContent = '100';
  ui.meter.style.setProperty('--score', '100%');
  ui.start.classList.add('hidden');
  ui.placeholderStart.classList.add('hidden');
  ui.stop.classList.remove('hidden');
  setCurrentState('MODELS');
  setModelStatus('Opening camera…');
  animationFrame = requestAnimationFrame(renderLoop);
  if (faceTask && phoneModel) {
    setCurrentState('FOCUSED');
    setModelStatus('Detection ready', 'good');
  } else void loadModels();
}

let animationFrame = 0;
async function startSession() {
  if (running) return;
  ui.start.disabled = true;
  ui.start.innerHTML = '<span class="button-dot"></span> Starting…';
  try {
    // Resume audio from the user's button click so later voice alerts are allowed.
    if (window.AudioContext) {
      audioContext ||= new AudioContext();
      if (audioContext.state === 'suspended') await audioContext.resume();
    }
    await openCamera();
    await loadVoiceBuffers();
    beginSession();
  } catch (error) {
    const message = explainCameraError(error);
    showToast(message);
    setModelStatus(message, 'bad');
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
  } finally {
    ui.start.disabled = false;
    ui.start.innerHTML = '<span class="button-dot"></span> Start study';
  }
}

function stopSession() {
  if (!running && !stream) return;
  running = false;
  cancelAnimationFrame(animationFrame);
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  ui.video.pause();
  ui.video.srcObject = null;
  ui.stage.classList.remove('has-video');
  ui.live.classList.remove('is-live');
  ui.cameraLabel.textContent = 'CAMERA OFF';
  ui.calibrate.disabled = true;
  ui.start.classList.remove('hidden');
  ui.placeholderStart.classList.remove('hidden');
  ui.stop.classList.add('hidden');
  ctx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
  resetAudio();
  setCurrentState('READY');
  setModelStatus('Start camera to begin');
  ui.score.textContent = '—';
  ui.eyeScore.textContent = '—';
  ui.eyeState.textContent = '—';
  ui.eyeState.className = 'eye-status-text';
  ui.eyeFill.style.width = '0%';
  ui.eyeTime.textContent = `CLOSED 0.00 / ${closedDuration.toFixed(2)}s`;
  ui.leftEye.textContent = 'LEFT · —';
  ui.rightEye.textContent = 'RIGHT · —';
  ui.clock.textContent = '00:00:00';
}

function startCalibration() {
  calibration = { startedAt: performance.now(), values: [] };
  showToast('Eyes open rakhkar camera ko dekhein · calibrating…');
  ui.calibrate.textContent = 'Calibrating…';
  ui.calibrate.disabled = true;
}

function finishCalibration(now) {
  if (!calibration || now - calibration.startedAt < 1600) return;
  const { values } = calibration;
  calibration = null;
  ui.calibrate.textContent = 'Calibrate eyes';
  ui.calibrate.disabled = false;
  if (values.length < 5) {
    showToast('Face clear nahi tha. Light badha kar dobara calibrate karein.');
    return;
  }
  values.sort((a, b) => a - b);
  const baseline = values[Math.floor(values.length / 2)];
  eyeThreshold = clamp(Math.max(0.34, baseline + 0.22), 0.34, 0.72);
  ui.eyeSensitivity.value = eyeThreshold.toFixed(2);
  ui.eyeSensitivityValue.value = eyeThreshold.toFixed(2);
  updateThresholdLabel();
  showToast(`Eyes calibrated · alert level ${eyeThreshold.toFixed(2)}`);
}

function updateThresholdLabel() {
  ui.eyeThresholdLabel.textContent = `CLOSE LEVEL ${Math.round(eyeThreshold * 100)}%`;
}

function updateEyeUI(scores) {
  if (!scores) {
    ui.eyeScore.textContent = 'No face';
    ui.eyeState.textContent = 'NOT SEEN';
    ui.eyeState.className = 'eye-status-text not-seen';
    ui.eyeFill.style.width = '0%';
    ui.leftEye.textContent = 'LEFT · —';
    ui.rightEye.textContent = 'RIGHT · —';
    ui.eyeTime.textContent = `CLOSED 0.00 / ${closedDuration.toFixed(2)}s`;
    if (lastClosedAt === 0 || performance.now() - lastClosedAt > CLOSED_GRACE_MS) {
      const wasAlert = eyeAlertFired;
      closedSince = null;
      eyeAlertFired = false;
      if (wasAlert) {
        stopVoice('eye');
        playQueuedPhoneVoice();
      }
    }
    return eyeAlertFired;
  }
  const pct = Math.round(scores.average * 100);
  ui.eyeScore.textContent = `${pct}% closed`;
  ui.eyeFill.style.width = `${pct}%`;
  ui.eyeFill.style.background = pct / 100 >= eyeThreshold ? 'var(--amber)' : 'var(--green)';
  ui.leftEye.textContent = `LEFT EYE · ${Math.round(scores.left * 100)}%`;
  ui.rightEye.textContent = `RIGHT EYE · ${Math.round(scores.right * 100)}%`;
  if (calibration) {
    calibration.values.push(scores.average);
    finishCalibration(performance.now());
  }
  const candidate = scores.average >= eyeThreshold
    && scores.left >= eyeThreshold * 0.8
    && scores.right >= eyeThreshold * 0.8;
  const now = performance.now();
  const wasAlert = eyeAlertFired;
  if (candidate) {
    if (closedSince === null) closedSince = now;
    lastClosedAt = now;
  } else if (now - lastClosedAt > CLOSED_GRACE_MS) {
    closedSince = null;
    eyeAlertFired = false;
  }
  const elapsed = closedSince === null ? 0 : now - closedSince;
  ui.eyeTime.textContent = `CLOSED ${(elapsed / 1000).toFixed(2)} / ${closedDuration.toFixed(2)}s`;
  if (closedSince !== null && elapsed >= closedDuration * 1000 && !eyeAlertFired) {
    eyeAlertFired = true;
    eyeAlertCount += 1;
    ui.eyeAlerts.textContent = String(eyeAlertCount);
    if (phoneIsActive) phoneVoicePending = true;
    playVoice('eye');
  } else if (eyeAlertFired && now - lastEyeVoiceAt > 15000 && ui.eyeAudio.paused) {
    playVoice('eye');
  }
  if (wasAlert && !eyeAlertFired) {
    stopVoice('eye');
    playQueuedPhoneVoice();
  }
  const eyeStatus = eyeAlertFired ? 'CLOSED' : candidate ? 'CLOSING' : 'OPEN';
  ui.eyeState.textContent = eyeStatus;
  ui.eyeState.className = `eye-status-text ${eyeStatus.toLowerCase()}`;
  const progress = clamp(elapsed / (closedDuration * 1000), 0, 1);
  ui.eyeFill.style.boxShadow = progress > 0 && progress < 1 ? `0 0 ${3 + progress * 8}px var(--amber)` : 'none';
  ui.eyeFill.style.opacity = closedSince === null ? '.68' : String(.68 + progress * .32);
  return eyeAlertFired;
}

function drawEyeBoxes(landmarks, scores) {
  for (const [side, ids, score] of [
    ['L', LEFT_EYE_IDS, scores?.left ?? 0],
    ['R', RIGHT_EYE_IDS, scores?.right ?? 0],
  ]) {
    const points = ids.map((id) => ({ x: (1 - landmarks[id].x) * ui.canvas.width, y: landmarks[id].y * ui.canvas.height }));
    const xs = points.map((point) => point.x);
    const ys = points.map((point) => point.y);
    const x = Math.min(...xs), y = Math.min(...ys), width = Math.max(...xs) - x, height = Math.max(...ys) - y;
    const padX = Math.max(9, width * .24), padY = Math.max(8, height * .7);
    const isClosed = score >= eyeThreshold;
    const color = isClosed ? '#ffbd73' : '#c8f27a';
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, ui.canvas.width / 480);
    ctx.beginPath();
    ctx.roundRect(x - padX, y - padY, width + padX * 2, height + padY * 2, Math.max(7, width * .18));
    ctx.stroke();
    ctx.fillStyle = 'rgba(8, 12, 9, .78)';
    const label = `${side} EYE  ${Math.round(score * 100)}%`;
    ctx.font = `600 ${Math.max(12, ui.canvas.width / 65)}px sans-serif`;
    const labelWidth = ctx.measureText(label).width + 14;
    const labelY = Math.max(4, y - padY - Math.max(18, ui.canvas.width / 42));
    ctx.fillRect(x - padX, labelY, labelWidth, Math.max(18, ui.canvas.width / 42));
    ctx.fillStyle = color;
    ctx.fillText(label, x - padX + 7, labelY + Math.max(13, ui.canvas.width / 78));
  }
}

function drawPhoneBoxes() {
  for (const prediction of lastPhoneBoxes) {
    const [rawX, y, width, height] = prediction.bbox;
    const x = ui.canvas.width - rawX - width;
    ctx.strokeStyle = '#7650b5';
    ctx.lineWidth = Math.max(2, ui.canvas.width / 450);
    ctx.strokeRect(x, y, width, height);
    const label = `PHONE ${Math.round(prediction.score * 100)}%`;
    ctx.font = `600 ${Math.max(13, ui.canvas.width / 62)}px sans-serif`;
    const labelWidth = ctx.measureText(label).width + 16;
    ctx.fillStyle = '#7650b5';
    ctx.fillRect(x, Math.max(0, y - 27), labelWidth, 25);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + 8, Math.max(17, y - 9));
  }
}

function onPhonePredictions(predictions, now) {
  const boxes = predictions.filter((item) => item.class === 'cell phone' && item.score >= phoneConfidence);
  if (boxes.length) {
    lastPhoneBoxes = boxes;
    lastPhoneAt = now;
    if (!phoneIsActive) {
      phoneIsActive = true;
      phoneAlertCount += 1;
      ui.phoneAlerts.textContent = String(phoneAlertCount);
      if (eyeAlertFired) phoneVoicePending = true;
      else playVoice('phone');
    }
  } else if (now - lastPhoneAt > PHONE_HOLD_MS) {
    stopVoice('phone');
    phoneIsActive = false;
    phoneVoicePending = false;
    lastPhoneBoxes = [];
  }
}

function updateSessionUI(now, delta, eyeAlert, faceFound) {
  const phoneActive = phoneIsActive && now - lastPhoneAt <= PHONE_HOLD_MS;
  const state = eyeAlert && phoneActive ? 'BOTH' : eyeAlert ? 'EYES' : phoneActive ? 'PHONE' : (!faceTask || !phoneModel) ? 'MODELS' : !faceFound ? 'NO_FACE' : 'FOCUSED';
  setCurrentState(state);

  const distracted = phoneActive || eyeAlert;
  if (!distracted) focusedSeconds += delta;

  const elapsed = (now - sessionStartedAt) / 1000;
  const score = elapsed > 1 ? Math.round(clamp((focusedSeconds / elapsed) * 100, 0, 100)) : 100;
  ui.score.textContent = String(score);
  ui.meter.style.setProperty('--score', `${score}%`);
  ui.meter.style.background = `conic-gradient(${score < 60 ? 'var(--amber)' : 'var(--green)'} ${score}%, #edf1ed ${score}%)`;
  ui.clock.textContent = formatClock(elapsed);
  return state;
}

function renderLoop() {
  if (!running) return;
  animationFrame = requestAnimationFrame(renderLoop);
  if (ui.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  const now = performance.now();
  const delta = clamp((now - lastTick) / 1000, 0, .20);
  lastTick = now;
  let faceFound = false;
  let eyeAlert = eyeAlertFired;

  if (faceTask && ui.video.currentTime !== lastVideoTime && now - lastFaceInference > 70) {
    lastVideoTime = ui.video.currentTime;
    lastFaceInference = now;
    try {
      const result = faceTask.detectForVideo(ui.video, now);
      const landmarks = result.faceLandmarks?.[0];
      faceFound = Boolean(landmarks);
      if (landmarks) {
        lastFaceLandmarks = landmarks;
        const raw = eyeScoreMap(result);
        if (raw) {
          const scores = {
            left: smoothScore('left', raw.left),
            right: smoothScore('right', raw.right),
          };
          scores.average = (scores.left + scores.right) / 2;
          currentBlend = scores;
          eyeAlert = Boolean(updateEyeUI(scores));
        } else {
          currentBlend = null;
          lastFaceLandmarks = null;
          updateEyeUI(null);
          ctx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
        }
      } else {
        currentBlend = null;
        lastFaceLandmarks = null;
        updateEyeUI(null);
        ctx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
      }
    } catch (error) {
      console.warn('Eye detection frame failed:', error);
    }
  } else {
    faceFound = currentBlend !== null;
    eyeAlert = eyeAlertFired;
  }

  // A phone pass runs less often than face tracking to keep the page responsive.
  if (phoneModel && !phoneInferenceRunning && now - lastPhoneInference > 400) {
    lastPhoneInference = now;
    phoneInferenceRunning = true;
    phoneModel.detect(ui.video, 6, phoneConfidence)
      .then((predictions) => onPhonePredictions(predictions, performance.now()))
      .catch((error) => console.warn('Phone detection frame failed:', error))
      .finally(() => { phoneInferenceRunning = false; });
  }
  if (now - lastPhoneAt > PHONE_HOLD_MS) {
    stopVoice('phone');
    phoneIsActive = false;
    phoneVoicePending = false;
    lastPhoneBoxes = [];
  }

  ctx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
  if (faceTask && currentBlend && lastFaceLandmarks) drawEyeBoxes(lastFaceLandmarks, currentBlend);
  drawPhoneBoxes();
  updateSessionUI(now, delta, eyeAlertFired, faceFound);
}

let lastFaceLandmarks = null;

ui.start.addEventListener('click', startSession);
ui.placeholderStart.addEventListener('click', startSession);
ui.stop.addEventListener('click', stopSession);
ui.calibrate.addEventListener('click', startCalibration);
ui.fullscreen.addEventListener('click', async () => {
  try {
    if (!document.fullscreenElement) await ui.shell.requestFullscreen();
    else await document.exitFullscreen();
  } catch { showToast('Is browser mein fullscreen available nahi hai. F11 use karein.'); }
});
ui.eyeSensitivity.addEventListener('input', () => {
  eyeThreshold = Number(ui.eyeSensitivity.value);
  ui.eyeSensitivityValue.value = eyeThreshold.toFixed(2);
  updateThresholdLabel();
});
ui.closedDuration.addEventListener('input', () => {
  closedDuration = Number(ui.closedDuration.value);
  ui.closedDurationValue.value = `${closedDuration.toFixed(2)}s`;
});
ui.phoneSensitivity.addEventListener('input', () => {
  phoneConfidence = Number(ui.phoneSensitivity.value);
  ui.phoneSensitivityValue.value = `${Math.round(phoneConfidence * 100)}%`;
});
ui.cameraSelect.addEventListener('change', async () => {
  if (!running) return;
  try {
    stream?.getTracks().forEach((track) => track.stop());
    await openCamera();
    lastVideoTime = -1;
    eyeHistory = { left: [], right: [] };
  } catch (error) { showToast(error.message || 'Selected camera did not open.'); }
});
ui.audioToggle.addEventListener('change', () => {
  if (!ui.audioToggle.checked) resetAudio();
});
window.addEventListener('pagehide', () => {
  if (stream) stream.getTracks().forEach((track) => track.stop());
});

updateThresholdLabel();
setCurrentState('READY');
