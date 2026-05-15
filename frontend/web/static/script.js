/* ══════════════════════════════════════════════════════════════════════════════
   SoundSense — Main Script
   Handles: SPA routing, live detection, waveform, history, analytics,
            custom sound recording, feedback, ISL guide, settings
══════════════════════════════════════════════════════════════════════════════ */

'use strict';

// ── Constants ──────────────────────────────────────────────────────────────────
const EMERGENCY_SOUNDS = new Set([
  'fire_alarm','siren','glass_breaking','thunderstorm'
]);
const IMPORTANT_SOUNDS = new Set([
  'door_wood_knock','traditional_bell','crying_baby','dog',
  'car_horn','motorcycle_horn','clock_alarm','train'
]);

const ISL_MAP = {
  'dog':              { icon:'🐕', msg:'A dog is barking nearby!',         sub:'Stay calm, keep distance' },
  'crying_baby':      { icon:'👶', msg:'A baby is crying!',                sub:'Baby needs attention immediately' },
  'fire_alarm':       { icon:'🔥', msg:'FIRE ALARM! Evacuate now!',        sub:'Leave the building immediately' },
  'siren':            { icon:'🚨', msg:'Emergency siren detected!',         sub:'Move aside, emergency vehicle approaching' },
  'car_horn':         { icon:'🚗', msg:'Car horn — be careful!',            sub:'Vehicle approaching' },
  'glass_breaking':   { icon:'💥', msg:'Glass breaking sound!',            sub:'Possible accident or intrusion' },
  'door_wood_knock':  { icon:'🚪', msg:'Someone at the door',              sub:'Check who is knocking' },
  'clock_alarm':      { icon:'⏰', msg:'Alarm is ringing!',                sub:'Time alert' },
  'clapping':         { icon:'👏', msg:'Applause detected',                sub:'Celebration or performance nearby' },
  'laughing':         { icon:'😄', msg:'Laughter nearby',                  sub:'Joyful moment' },
  'rain':             { icon:'🌧️', msg:'It is raining',                    sub:'Carry an umbrella' },
  'thunderstorm':     { icon:'⛈️', msg:'Thunderstorm!',                    sub:'Stay indoors' },
  'footsteps':        { icon:'👣', msg:'Footsteps nearby',                 sub:'Someone is walking close' },
  'washing_machine':  { icon:'🫧', msg:'Washing machine running',          sub:'Laundry cycle in progress' },
  'toilet_flush':     { icon:'🚽', msg:'Toilet flushed',                   sub:'Bathroom in use' },
  'auto_rickshaw_horn':{ icon:'🛺', msg:'Auto-rickshaw horn!',             sub:'Indian traffic nearby' },
  'motorcycle_horn':  { icon:'🏍️', msg:'Motorcycle horn!',                sub:'Two-wheeler approaching' },
  'temple_bells':     { icon:'🔔', msg:'Temple bells ringing',             sub:'Religious ceremony nearby' },
  'traditional_bell': { icon:'🔔', msg:'Doorbell ringing!',               sub:'Someone at the door' },
  'train':            { icon:'🚂', msg:'Train approaching!',               sub:'Stay away from tracks' },
  'engine':           { icon:'🚘', msg:'Engine sound detected',            sub:'Vehicle nearby' },
  'cat':              { icon:'🐈', msg:'Cat meowing',                      sub:'Your cat needs attention' },
  'rooster':          { icon:'🐓', msg:'Rooster crowing — morning!',       sub:'Early morning signal' },
  'coughing':         { icon:'😷', msg:'Someone is coughing',             sub:'Check if help is needed' },
  'sneezing':         { icon:'🤧', msg:'Sneeze detected',                  sub:'Bless you!' },
  'chirping_birds':   { icon:'🐦', msg:'Birds chirping',                   sub:'Peaceful outdoor sounds' },
  'pouring_water':    { icon:'💧', msg:'Water pouring',                    sub:'Tap or jug may be open' },
  'wind':             { icon:'💨', msg:'Wind blowing',                     sub:'Windy conditions' },
  'crackling_fire':   { icon:'🔥', msg:'Fire crackling',                   sub:'Open fire or gas stove' },
  'fireworks':        { icon:'🎆', msg:'Fireworks!',                       sub:'Celebration in the area' },
  'glass_breaking':   { icon:'💥', msg:'Glass breaking!',                  sub:'Careful of broken glass' },
  'snoring':          { icon:'😴', msg:'Snoring detected',                 sub:'Someone is sleeping' },
};

const SETTINGS_DEFAULTS = {
  sensitivity: 3, emergency: true, vibration: true,
  isl_auto: false, battery_saver: false
};

// ── State ──────────────────────────────────────────────────────────────────────
let state = {
  isListening:       false,
  lastDetection:     null,
  historyData:       [],
  historyFilter:     'all',
  mediaRecorder:     null,
  audioStream:       null,
  pollingTimer:      null,
  audioChunks:       [],
  charts:            {},
  customSamplesDone: 0,
  customBlobs:       [],
  isRecording:       false,
  waveAnimId:        null,
  analyser:          null,
  audioCtx:          null,
};

// ── Settings ───────────────────────────────────────────────────────────────────
function loadSettings() {
  const s = {};
  for (const [k, v] of Object.entries(SETTINGS_DEFAULTS)) {
    const stored = localStorage.getItem('sed_' + k);
    s[k] = stored === null ? v
         : typeof v === 'boolean' ? stored === 'true'
         : Number(stored);
  }
  return s;
}
function saveSetting(key, value) {
  localStorage.setItem('sed_' + key, value);
}
function applySettings() {
  const s = loadSettings();
  document.getElementById('slider-sensitivity').value = s.sensitivity;
  document.getElementById('sens-val').textContent       = s.sensitivity;
  document.getElementById('toggle-emergency').checked   = s.emergency;
  document.getElementById('toggle-vibration').checked   = s.vibration;
  document.getElementById('toggle-isl').checked         = s.isl_auto;
  document.getElementById('toggle-battery').checked     = s.battery_saver;
}
function getPollingInterval() {
  return loadSettings().battery_saver ? 5000 : 2000;
}
// Sensitivity 1-5 → confidence threshold 0.12 → 0.40
function getThreshold() {
  const sens = loadSettings().sensitivity;
  return 0.12 + (sens - 1) * 0.07;
}

// ── SPA Routing ────────────────────────────────────────────────────────────────
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  document.getElementById('nav-' + name).classList.add('active');
  // Load page data
  const loaders = {
    history:   loadHistory,
    analytics: loadAnalytics,
    feedback:  refreshFeedbackPage,
    custom:    loadCustomSounds,
    isl:       renderISLGrid,
    settings:  () => { applySettings(); renderEmergencySoundsList(); }
  };
  if (loaders[name]) loaders[name]();
}

// ── Waveform Canvas ────────────────────────────────────────────────────────────
const canvas = document.getElementById('waveform-canvas');
const ctx2d  = canvas.getContext('2d');
let   waveData = new Array(60).fill(0);
let   wavePhase = 0;

function resizeCanvas() {
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function drawWave(live) {
  const W = canvas.width, H = canvas.height;
  ctx2d.clearRect(0, 0, W, H);

  const bars = 60;
  const bw   = W / bars - 2;

  if (live && state.analyser) {
    const buf = new Uint8Array(state.analyser.frequencyBinCount);
    state.analyser.getByteFrequencyData(buf);
    for (let i = 0; i < bars; i++) {
      waveData[i] = buf[Math.floor(i * buf.length / bars)] / 255;
    }
  } else if (!live) {
    wavePhase += 0.04;
    for (let i = 0; i < bars; i++) {
      waveData[i] = (Math.sin(i * 0.3 + wavePhase) * 0.15 + 0.15);
    }
  }

  for (let i = 0; i < bars; i++) {
    const h = waveData[i] * H * 0.85 + 2;
    const x = i * (bw + 2);
    const isEmergency = state.lastDetection
      && EMERGENCY_SOUNDS.has(state.lastDetection.class);
    const alpha = 0.5 + waveData[i] * 0.5;
    ctx2d.fillStyle = isEmergency
      ? `rgba(255,68,68,${alpha})`
      : `rgba(79,142,247,${alpha})`;
    const y = (H - h) / 2;
    ctx2d.beginPath();
    ctx2d.roundRect(x, y, bw, h, 3);
    ctx2d.fill();
  }
  state.waveAnimId = requestAnimationFrame(() => drawWave(live));
}

function startWaveIdle() {
  if (state.waveAnimId) cancelAnimationFrame(state.waveAnimId);
  drawWave(false);
}
function startWaveLive() {
  if (state.waveAnimId) cancelAnimationFrame(state.waveAnimId);
  drawWave(true);
}

// ── Live Listening ─────────────────────────────────────────────────────────────
async function toggleListen() {
  if (state.isListening) { stopListening(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    state.audioStream = stream;

    // Hook up analyser for waveform
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    state.analyser = state.audioCtx.createAnalyser();
    state.analyser.fftSize = 256;
    state.audioCtx.createMediaStreamSource(stream).connect(state.analyser);

    state.isListening = true;
    document.getElementById('btn-listen').textContent    = '⏹ Stop Listening';
    document.getElementById('btn-listen').className      = 'btn btn-danger';
    document.getElementById('listen-dot').classList.add('on');
    document.getElementById('listen-label').textContent  = 'Listening…';
    document.getElementById('listen-label').classList.add('on');
    setStatus('🎙 Capturing audio every 3 seconds…');

    startWaveLive();
    scheduleRecord();
  } catch(e) {
    setStatus('❌ Microphone access denied: ' + e.message);
  }
}

function stopListening() {
  state.isListening = false;
  if (state.audioStream) {
    state.audioStream.getTracks().forEach(t => t.stop());
    state.audioStream = null;
  }
  if (state.audioCtx) { state.audioCtx.close(); state.audioCtx = null; state.analyser = null; }
  if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') state.mediaRecorder.stop();
  clearTimeout(state.pollingTimer);
  document.getElementById('btn-listen').textContent   = '🎙 Start Listening';
  document.getElementById('btn-listen').className     = 'btn btn-primary';
  document.getElementById('listen-dot').classList.remove('on');
  document.getElementById('listen-label').textContent = 'Not Listening';
  document.getElementById('listen-label').classList.remove('on');
  setStatus('💡 Press Start Listening or upload an audio file');
  startWaveIdle();
}

function scheduleRecord() {
  if (!state.isListening) return;
  const chunks = [];
  let mr;
  try {
    mr = new MediaRecorder(state.audioStream, { mimeType: 'audio/webm' });
  } catch(e) {
    mr = new MediaRecorder(state.audioStream);
  }
  mr.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  mr.onstop = () => {
    if (!state.isListening) return;
    const blob = new Blob(chunks, { type: mr.mimeType });
    sendAudioBlob(blob, mr.mimeType.includes('webm') ? 'audio.webm' : 'audio.wav');
  };
  mr.start();
  setTimeout(() => {
    if (mr.state !== 'inactive') mr.stop();
    // Chain next recording
    state.pollingTimer = setTimeout(scheduleRecord, 300);
  }, 3000);
}

// ── Send audio for prediction ──────────────────────────────────────────────────
async function sendAudioBlob(blob, filename) {
  const fd = new FormData();
  fd.append('audio', blob, filename);
  try {
    const res  = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) { setStatus('❌ ' + (data.error || 'Prediction failed')); return; }
    handleDetection(data);
  } catch(e) {
    setStatus('❌ Server error: ' + e.message);
  }
}

async function uploadFile(input) {
  const file = input.files[0];
  if (!file) return;
  setStatus('⏳ Analysing ' + file.name + '…');
  const fd = new FormData();
  fd.append('audio', file, file.name);
  try {
    const res  = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) { setStatus('❌ ' + (data.error || 'Prediction failed')); return; }
    handleDetection(data);
  } catch(e) {
    setStatus('❌ Error: ' + e.message);
  }
  input.value = '';
}

// ── Handle detection result ────────────────────────────────────────────────────
function handleDetection(data) {
  state.lastDetection = data;
  let label        = data.class;
  const confidence = Math.round(data.confidence * 100);
  const direction  = data.direction || 'CENTER';
  
  if (data.confidence < getThreshold()) {
      label = 'Uncertain';
  }
  
  const isEmg       = EMERGENCY_SOUNDS.has(label);
  const isImp       = IMPORTANT_SOUNDS.has(label);
  const isUncertain = label === 'Uncertain' || label === 'Unknown';
  let severity      = isUncertain ? 'idle' : isEmg ? 'emergency' : isImp ? 'important' : 'normal';

  // Extract Icon
  const mapData = ISL_MAP[label] || { icon:'🔍' };
  const icon = isUncertain ? '❓' : mapData.icon;
  const bigLabel = isUncertain ? 'UNCERTAIN' : `${icon} ${label.replace(/_/g, ' ')} detected`;

  // Detection card
  const card = document.getElementById('detection-card');
  card.className = `detection-card ${severity}`;
  document.getElementById('detected-label').textContent = bigLabel;

  // Confidence bar
  const fill = document.getElementById('conf-fill');
  fill.style.width = confidence + '%';
  fill.className   = `conf-fill ${severity}`;
  document.getElementById('conf-text').textContent = `Confidence: ${confidence}%`;

  // Priority badge
  const badge = document.getElementById('priority-badge');
  let badgeHtml = '';
  if (!isUncertain) {
      if (isEmg) badgeHtml = `<span class="badge badge-emergency">🔴 EMERGENCY</span>`;
      else if (isImp) badgeHtml = `<span class="badge badge-important">🟡 IMPORTANT</span>`;
      else badgeHtml = `<span class="badge badge-normal">🟢 NORMAL</span>`;
  }
  badge.innerHTML = badgeHtml;

  // Direction
  updateDirection(direction);

  // Status
  setStatus(isUncertain ? '🔇 Low confidence — no reliable match' : `✅ Detected: ${label.replace(/_/g,' ')} (${confidence}%)`);

  // Emergency & Important effects
  const settings = loadSettings();
  document.body.className = ''; // remove old flashes
  
  if (!isUncertain && (isEmg || isImp) && settings.emergency) {
    if (settings.vibration && navigator.vibrate) {
        if (isEmg) navigator.vibrate([300, 100, 300, 100, 300]); // heavy panic vibration
        else navigator.vibrate([200, 100, 200]);                 // standard important alert
    }
    
    // Visual screen flash sync
    document.body.classList.add(isEmg ? 'flash-emergency' : 'flash-important');
    
    document.getElementById('vibrate-indicator').style.display = 'flex';
    setTimeout(() => {
        document.getElementById('vibrate-indicator').style.display = 'none';
        document.body.className = ''; 
    }, 2000);
    
    if (settings.isl_auto) showISLOverlay(label);
  } else {
    document.getElementById('vibrate-indicator').style.display = 'none';
  }

  // Mini history
  if (!isUncertain) addToMiniHistory(label, confidence, severity);
}

function updateDirection(dir) {
  const l = document.getElementById('dir-left');
  const c = document.getElementById('dir-center');
  const r = document.getElementById('dir-right');
  l.className = c.className = r.className = '';
  if (dir === 'LEFT')   l.className = 'd-active';
  else if (dir === 'RIGHT') r.className = 'd-active';
  else                    c.className = 'd-active';
  document.getElementById('dir-label').textContent = '📍 Sound from ' + dir;
}

function addToMiniHistory(label, confidence, severity) {
  const list = document.getElementById('mini-history');
  const time = new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' });
  const item = document.createElement('div');
  item.className = 'history-item';
  const icon = ISL_MAP[label] ? ISL_MAP[label].icon : '🔊';
  item.innerHTML = `
    <div class="history-dot ${severity}"></div>
    <div class="history-name">${icon} ${label.replace(/_/g,' ')}</div>
    <div class="history-time">${time}</div>
    <span class="history-conf">${confidence}%</span>`;
  // Prepend and keep max 3
  list.insertBefore(item, list.firstChild);
  while (list.children.length > 3) list.removeChild(list.lastChild);
  if (list.querySelector('.history-item') === null) {
    list.innerHTML = '<div style="color:var(--muted);font-size:13px;text-align:center;padding:16px;">No detections yet</div>';
  }
}

function setStatus(msg) { document.getElementById('status-text').textContent = msg; }

// ── ISL Overlay ────────────────────────────────────────────────────────────────
function showISLOverlay(label) {
  const info = ISL_MAP[label] || { icon: '🔊', msg: label, sub: 'Sound detected' };
  document.getElementById('overlay-icon').textContent    = info.icon;
  document.getElementById('overlay-message').textContent = info.msg;
  document.getElementById('overlay-sub').textContent     = info.sub;
  const overlay = document.getElementById('isl-overlay');
  overlay.style.display = 'flex';
  setTimeout(() => { overlay.style.display = 'none'; }, 8000);
}

// ── ISL Guide Page ─────────────────────────────────────────────────────────────
function renderISLGrid() {
  // Active card
  const active = document.getElementById('isl-active-card');
  if (state.lastDetection && state.lastDetection.class !== 'Uncertain') {
    const label = state.lastDetection.class;
    const isEmg = EMERGENCY_SOUNDS.has(label);
    const isImp = IMPORTANT_SOUNDS.has(label);
    const info  = ISL_MAP[label] || { icon: '🔊', msg: label, sub: 'Sound detected' };
    document.getElementById('isl-icon').textContent    = info.icon;
    document.getElementById('isl-message').textContent = info.msg;
    document.getElementById('isl-sub').textContent     = info.sub;
    
    const pBadge = document.getElementById('isl-priority-badge');
    if (isEmg) {
        pBadge.className = 'badge badge-emergency';
        pBadge.textContent = '🔴 EMERGENCY';
    } else if (isImp) {
        pBadge.className = 'badge badge-important';
        pBadge.textContent = '🟡 IMPORTANT';
    } else {
        pBadge.className = 'badge badge-normal';
        pBadge.textContent = '🟢 NORMAL';
    }
    active.style.display = 'block';
  } else {
    active.style.display = 'none';
  }

  // Grid of all ISL references
  const grid = document.getElementById('isl-grid');
  grid.innerHTML = '';
  for (const [sound, info] of Object.entries(ISL_MAP)) {
    const isEmg = EMERGENCY_SOUNDS.has(sound);
    const isImp = IMPORTANT_SOUNDS.has(sound);
    const tile  = document.createElement('div');
    tile.className = 'isl-tile';
    
    let innerBadge = '';
    if (isEmg) innerBadge = '<div style="margin-top:4px;"><span class="badge badge-emergency" style="font-size:9px">🔴 EMERGENCY</span></div>';
    else if (isImp) innerBadge = '<div style="margin-top:4px;"><span class="badge badge-important" style="font-size:9px">🟡 IMPORTANT</span></div>';

    tile.innerHTML = `
      <div class="isl-tile-icon">${info.icon}</div>
      <div class="isl-tile-name">${sound.replace(/_/g,' ')}</div>
      ${innerBadge}`;
      
    tile.onclick = () => {
      document.getElementById('isl-icon').textContent    = info.icon;
      document.getElementById('isl-message').textContent = info.msg;
      document.getElementById('isl-sub').textContent     = info.sub;
      
      const pBadge = document.getElementById('isl-priority-badge');
      if (isEmg) {
          pBadge.className = 'badge badge-emergency';
          pBadge.textContent = '🔴 EMERGENCY';
      } else if (isImp) {
          pBadge.className = 'badge badge-important';
          pBadge.textContent = '🟡 IMPORTANT';
      } else {
          pBadge.className = 'badge badge-normal';
          pBadge.textContent = '🟢 NORMAL';
      }
      active.style.display = 'block';
    };
    grid.appendChild(tile);
  }
}

// ── Alert History Page ─────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const res = await fetch('/history');
    state.historyData = await res.json();
    renderHistory();
  } catch(e) {
    document.getElementById('history-list').innerHTML =
      '<div style="color:var(--muted);text-align:center;padding:32px;">Could not load history</div>';
  }
}

function setFilter(filter, btn) {
  state.historyFilter = filter;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderHistory();
}

function renderHistory() {
  const search = (document.getElementById('history-search').value || '').toLowerCase();
  let data = state.historyData;
  if (state.historyFilter !== 'all') data = data.filter(r => r.priority === state.historyFilter);
  if (search) data = data.filter(r => r.class.toLowerCase().includes(search));

  const list = document.getElementById('history-list');
  if (!data.length) {
    list.innerHTML = '<div style="color:var(--muted);text-align:center;padding:32px;">No records found</div>';
    return;
  }
  list.innerHTML = data.map(r => {
    const conf = Math.round(r.confidence * 100);
    const isEmg = r.priority === 'emergency';
    const time  = new Date(r.timestamp).toLocaleString();
    return `
      <div class="history-item">
        <div class="history-dot ${isEmg ? 'emergency' : 'normal'}"></div>
        <span class="history-name">${r.class}</span>
        <span class="badge ${isEmg ? 'badge-emergency' : 'badge-normal'}" style="font-size:10px;">${isEmg ? '🔴' : '🟡'}</span>
        <span class="history-time">${time}</span>
        <span class="history-conf">${conf}%</span>
      </div>`;
  }).join('');
}

// ── Feedback Page ──────────────────────────────────────────────────────────────
async function refreshFeedbackPage() {
  if (state.lastDetection) {
    document.getElementById('feedback-sound-name').textContent = state.lastDetection.class;
    document.getElementById('feedback-conf-text').textContent  =
      `Confidence: ${Math.round(state.lastDetection.confidence * 100)}%`;
  }

  // Populate dropdown with all known class names
  const select = document.getElementById('feedback-correct-label');
  select.innerHTML = '<option value="">— Select the correct sound —</option>';
  const labels = Object.keys(ISL_MAP).sort();
  labels.forEach(l => {
    const opt = document.createElement('option');
    opt.value = l; opt.textContent = l.replace(/_/g,' ');
    select.appendChild(opt);
  });

  // Load stats from analytics
  try {
    const res  = await fetch('/analytics');
    const data = await res.json();
    const stats = { total: 0, correct: 0, wrong: 0 };
    // Approximate from accuracy
    if (data.accuracy !== null) {
      document.getElementById('stat-accuracy').textContent = data.accuracy + '%';
    } else {
      document.getElementById('stat-accuracy').textContent = '–';
    }
    // Feedback count from analytics response isn't there yet; just show from history
    document.getElementById('stat-total-fb').textContent  = '–';
    document.getElementById('stat-wrong').textContent     = '–';
  } catch(e) {}
}

function showWrongDropdown() {
  document.getElementById('wrong-section').style.display = 'block';
  document.getElementById('learning-msg').classList.remove('show');
}

async function sendFeedback(isCorrect) {
  if (!state.lastDetection) return;
  const actual = isCorrect ? '' : document.getElementById('feedback-correct-label').value;
  try {
    await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        detected: state.lastDetection.class,
        correct:  isCorrect,
        actual:   actual
      })
    });
    document.getElementById('wrong-section').style.display = 'none';
    document.getElementById('learning-msg').classList.add('show');
    setTimeout(() => document.getElementById('learning-msg').classList.remove('show'), 4000);
  } catch(e) {}
}

// ── Analytics Page ─────────────────────────────────────────────────────────────
async function loadAnalytics() {
  try {
    const res = await fetch('/analytics');
    const d   = await res.json();
    document.getElementById('an-total').textContent     = d.total;
    document.getElementById('an-emergency').textContent = d.emergency_count;
    document.getElementById('an-conf').textContent      = d.avg_confidence + '%';
    document.getElementById('an-accuracy').textContent  = d.accuracy !== null ? d.accuracy + '%' : '–';

    renderTopChart(d.top10);
    renderDailyChart(d.daily);
  } catch(e) {
    console.warn('Could not load analytics', e);
  }
}

function renderTopChart(top10) {
  const el = document.getElementById('chart-top');
  if (state.charts.top) state.charts.top.destroy();
  state.charts.top = new Chart(el, {
    type: 'bar',
    data: {
      labels:   top10.map(r => r.class.replace(/_/g,' ')),
      datasets: [{
        label: 'Detections',
        data:  top10.map(r => r.count),
        backgroundColor: top10.map(r =>
          EMERGENCY_SOUNDS.has(r.class) ? 'rgba(255,68,68,0.7)' : 'rgba(79,142,247,0.7)'
        ),
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color:'#8888AA', font:{size:11} }, grid:{color:'rgba(255,255,255,0.05)'} },
        y: { ticks: { color:'#8888AA' }, grid:{color:'rgba(255,255,255,0.05)'}, beginAtZero:true }
      }
    }
  });
}

function renderDailyChart(daily) {
  const el = document.getElementById('chart-daily');
  if (state.charts.daily) state.charts.daily.destroy();
  state.charts.daily = new Chart(el, {
    type: 'line',
    data: {
      labels:   daily.map(r => r.date.slice(5)),   // MM-DD
      datasets: [{
        label: 'Detections',
        data:  daily.map(r => r.count),
        borderColor:     '#4F8EF7',
        backgroundColor: 'rgba(79,142,247,0.12)',
        fill: true, tension: 0.4, borderWidth: 2,
        pointBackgroundColor: '#4F8EF7', pointRadius: 4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color:'#8888AA' }, grid:{color:'rgba(255,255,255,0.05)'} },
        y: { ticks: { color:'#8888AA' }, grid:{color:'rgba(255,255,255,0.05)'}, beginAtZero:true }
      }
    }
  });
}

// ── Custom Sound Page ──────────────────────────────────────────────────────────
let customStream = null;

async function toggleRecord() {
  if (state.isRecording) return;  // prevent double-tap
  const name = document.getElementById('custom-sound-name').value.trim();
  if (!name) { alert('Please enter a sound name first!'); return; }
  if (state.customSamplesDone >= 3) return;

  state.isRecording = true;
  const btn = document.getElementById('record-btn');
  btn.classList.add('recording');
  const sampleNum = state.customSamplesDone + 1;
  setDotState(sampleNum, 'active');
  document.getElementById('record-status-text').textContent = `Recording sample ${sampleNum}/3… (3 sec)`;

  try {
    if (!customStream) {
      customStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    }
    const chunks = [];
    let mimeType = 'audio/webm';
    const mr = MediaRecorder.isTypeSupported('audio/webm')
      ? new MediaRecorder(customStream, { mimeType: 'audio/webm' })
      : new MediaRecorder(customStream);
    mimeType = mr.mimeType;
    mr.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    mr.onstop = () => {
      const blob = new Blob(chunks, { type: mimeType });
      state.customBlobs.push(blob);
      state.customSamplesDone++;
      setDotState(sampleNum, 'done');
      btn.classList.remove('recording');
      state.isRecording = false;
      if (state.customSamplesDone >= 3) {
        document.getElementById('record-status-text').textContent = '✅ 3 samples recorded. Click Save!';
        document.getElementById('btn-save-sound').disabled = false;
      } else {
        document.getElementById('record-status-text').textContent =
          `Sample ${sampleNum}/3 done. Tap to record next.`;
      }
    };
    mr.start();
    setTimeout(() => mr.stop(), 3000);
  } catch(e) {
    state.isRecording = false;
    btn.classList.remove('recording');
    document.getElementById('record-status-text').textContent = '❌ Mic access denied';
  }
}

function setDotState(n, st) {
  const dot = document.getElementById('dot-' + n);
  if (!dot) return;
  dot.className = 'sample-dot ' + st;
}

async function saveCustomSound() {
  const name = document.getElementById('custom-sound-name').value.trim();
  if (!name || state.customBlobs.length < 3) return;

  const fd = new FormData();
  fd.append('name', name);
  state.customBlobs.forEach((blob, i) => {
    fd.append(`sample_${i+1}`, blob, `sample_${i+1}.webm`);
  });

  try {
    const res = await fetch('/learn', { method: 'POST', body: fd });
    const d   = await res.json();
    if (res.ok) {
      document.getElementById('save-success').style.display = 'block';
      setTimeout(() => document.getElementById('save-success').style.display = 'none', 4000);
      // Reset
      state.customBlobs = []; state.customSamplesDone = 0;
      document.getElementById('custom-sound-name').value = '';
      document.getElementById('btn-save-sound').disabled = true;
      [1,2,3].forEach(n => setDotState(n, ''));
      document.getElementById('record-status-text').textContent = 'Tap to record';
      loadCustomSounds();
    }
  } catch(e) { alert('Save failed: ' + e.message); }
}

async function loadCustomSounds() {
  try {
    const res   = await fetch('/custom_sounds');
    const sounds = await res.json();
    const list  = document.getElementById('custom-sounds-list');
    if (!sounds.length) {
      list.innerHTML = '<div style="color:var(--muted);font-size:13px;text-align:center;padding:24px;">No custom sounds yet</div>';
      return;
    }
    list.innerHTML = sounds.map(s => `
      <div class="custom-sound-item">
        <span class="nav-icon">🎵</span>
        <span class="custom-sound-name">${s.name}</span>
        <span class="custom-sound-samples badge badge-accent">${s.samples} samples</span>
      </div>`).join('');
  } catch(e) {}
}

// ── Settings Page ──────────────────────────────────────────────────────────────
function renderEmergencySoundsList() {
  const el = document.getElementById('emergency-sounds-list');
  el.innerHTML = [...EMERGENCY_SOUNDS].map(s =>
    `<span class="badge badge-emergency">${s.replace(/_/g,' ')}</span>`
  ).join('');
}

// ── Init ───────────────────────────────────────────────────────────────────────
(function init() {
  applySettings();
  renderEmergencySoundsList();
  startWaveIdle();
  // Preload mini-history from server
  fetch('/history').then(r => r.json()).then(data => {
    state.historyData = data;
    const list = document.getElementById('mini-history');
    const recent = data.slice(0, 3);
    if (!recent.length) return;
    list.innerHTML = recent.map(r => `
      <div class="history-item">
        <div class="history-dot ${r.priority === 'emergency' ? 'emergency' : 'normal'}"></div>
        <span class="history-name">${r.class}</span>
        <span class="history-time">${new Date(r.timestamp).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</span>
        <span class="history-conf">${Math.round(r.confidence*100)}%</span>
      </div>`).join('');
  }).catch(() => {});
})();
