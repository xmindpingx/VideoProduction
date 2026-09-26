// Shorts Lab page: resumable chunked uploads, direction, and job tracking. No build step, no external scripts.
const $ = (s, el = document) => el.querySelector(s);
const state = { uploads: new Map(), jobs: new Map(), health: null, queue: [], uploading: false, musicId: null, wake: null };
const RESUME_KEY = 'shorts.resume';

// ------------------------------------------------------------------ API
async function api(path, opts = {}) {
  const init = { credentials: 'same-origin', headers: { 'X-Shorts': '1' }, ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || 'POST';
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(opts.json);
  }
  const res = await fetch('api/' + path, init);
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (res.status === 401 && path !== 'login') showLogin();
  if (!res.ok) {
    const err = new Error((data && (data.detail || data.error)) || `Error ${res.status}`);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

function fmtBytes(n) {
  if (n >= 1024 ** 3) return (n / 1024 ** 3).toFixed(2) + ' GB';
  if (n >= 1024 ** 2) return (n / 1024 ** 2).toFixed(0) + ' MB';
  return Math.max(1, Math.round(n / 1024)) + ' KB';
}
function fmtTime(s) {
  s = Math.round(s || 0);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
function readResume() {
  try { return JSON.parse(localStorage.getItem(RESUME_KEY) || '{}'); } catch { return {}; }
}
function writeResume(map) {
  try { localStorage.setItem(RESUME_KEY, JSON.stringify(map)); } catch { /* private mode */ }
}
const fileKey = (f, kind) => `${kind}|${f.name}|${f.size}|${f.lastModified}`;

async function keepAwake(on) {
  try {
    if (on && !state.wake && 'wakeLock' in navigator) state.wake = await navigator.wakeLock.request('screen');
    if (!on && state.wake) { await state.wake.release(); state.wake = null; }
  } catch { state.wake = null; }
}

// ------------------------------------------------------------------ views
function showLogin() {
  $('#app').hidden = true;
  $('#login').hidden = false;
  $('#token').focus();
}
async function showApp() {
  $('#login').hidden = true;
  $('#app').hidden = false;
  await Promise.all([loadUploads(), loadJobs(), loadHealth()]);
}

$('#login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  $('#login-error').textContent = '';
  try {
    await api('login', { json: { token: $('#token').value.trim() } });
    $('#token').value = '';
    showApp();
  } catch (err) {
    $('#login-error').textContent = err.message;
  }
});
$('#logout').addEventListener('click', async () => {
  await api('logout', { method: 'POST' }).catch(() => {});
  showLogin();
});

// ------------------------------------------------------------------ health
function dot(label, cls, title) {
  const s = document.createElement('span');
  s.className = 'dot ' + cls;
  s.textContent = label;
  s.title = title;
  return s;
}
async function loadHealth() {
  let h;
  try { h = await api('health'); } catch { return; }
  state.health = h;
  const box = $('#status');
  box.replaceChildren();
  const o = h.ollama;
  if (!o.ok) box.append(dot('AI', 'off', `Ollama not reachable at ${o.url}: the rule-based editor will be used`));
  else if (!o.text_ok || !o.vision_ok) box.append(dot('AI', 'warn', `Ollama is up, but pull: ${[!o.text_ok && o.text_model, !o.vision_ok && o.vision_model].filter(Boolean).join(', ')}`));
  else box.append(dot('AI', 'on', `Ollama ${o.via_openwebui ? 'via Open WebUI ' : ''}· ${o.text_model} + ${o.vision_model}`));
  const c = h.comfyui;
  if (!c.enabled) box.append(dot('Gen', 'warn', 'No video generator set up: transitions use still-frame motion'));
  else if (!c.ok || !c.workflow_ok) box.append(dot('Gen', 'off', c.error || c.workflow_error || 'ComfyUI problem'));
  else box.append(dot('Gen', 'on', 'ComfyUI ready' + (c.first_last_frame ? ' (first + last frame)' : '')));
  const g = h.gpu || {};
  if (g.torch) {
    const gpuOk = !/no GPU|not installed/.test(g.torch);
    box.append(dot('GPU', gpuOk && g.video_enc ? 'on' : gpuOk || g.video_enc ? 'warn' : 'off', `AI: ${g.torch} · Video: ${g.video}`));
  }
  if (!h.worker.ok) box.append(dot('Worker', 'off', h.worker.error || 'Worker is not running'));
  if (h.disk.free_gb < 10) box.append(dot(`${h.disk.free_gb} GB`, 'warn', 'Low disk space on the server'));
  const genOk = c.enabled && c.ok && c.workflow_ok;
  $('#gen-note').textContent = genOk
    ? 'The last frame of a shot goes to your ComfyUI image-to-video workflow with a camera-only prompt, so the next shot flows in.'
    : 'No ComfyUI generator is connected, so the last frame of a shot is animated with a camera move on the server (still-frame motion).';
}

// ------------------------------------------------------------------ uploads
function clipCard(u) {
  let li = document.getElementById('u-' + u.id);
  if (!li) {
    li = $('#clip-tpl').content.firstElementChild.cloneNode(true);
    li.id = 'u-' + u.id;
    $('.x', li).addEventListener('click', () => removeUpload(u.id));
    $('.use input', li).addEventListener('change', updateMake);
    $('select', li).addEventListener('change', (e) => api(`uploads/${u.id}`, { method: 'PATCH', json: { slowmo: e.target.value } }).catch(() => {}));
    let t;
    $('.note', li).addEventListener('input', (e) => {
      clearTimeout(t);
      t = setTimeout(() => api(`uploads/${u.id}`, { method: 'PATCH', json: { note: e.target.value } }).catch(() => {}), 600);
    });
    (u.kind === 'music' ? $('#music') : $('#clips')).prepend(li);
  }
  return li;
}

function renderUpload(u, extra = {}) {
  state.uploads.set(u.id, { ...state.uploads.get(u.id), ...u });
  const li = clipCard(u);
  $('.name', li).textContent = u.name;
  const m = u.meta || {};
  const bits = [fmtBytes(u.size)];
  if (m.duration) bits.push(fmtTime(m.duration));
  if (m.width) bits.push(`${m.width}×${m.height}`);
  if (m.fps) bits.push(`${Math.round(m.fps)} fps`);
  if (m.hdr) bits.push('HDR');
  $('.meta', li).textContent = bits.join(' · ');
  $('.badge', li).textContent = m.high_fps ? `SLO-MO ${Math.round(m.fps)}` : m.fps >= 50 ? `${Math.round(m.fps)} FPS` : '';
  const pct = u.status === 'ready' ? 100 : Math.floor((100 * (extra.sent ?? u.received)) / u.size);
  $('.bar i', li).style.width = pct + '%';
  const st = $('.state', li);
  st.classList.toggle('err', u.status === 'error' || !!extra.error);
  if (extra.error) st.textContent = extra.error;
  else if (u.status === 'uploading') st.textContent = extra.active ? `Uploading ${pct}%` : `Paused at ${pct}%: pick the same file again to resume`;
  else if (u.status === 'error') st.textContent = u.error || 'Upload failed';
  else if (u.kind === 'music') st.textContent = 'Ready';
  else st.textContent = { none: 'Uploaded · waiting for analysis', running: 'Analyzing…', done: 'Analyzed ✓', error: 'Uploaded (analysis will retry)' }[u.analysis] || 'Uploaded';
  if (u.status === 'ready' && u.kind === 'video') {
    const img = $('img', li);
    if (!img.src) img.src = `api/uploads/${u.id}/thumb.jpg`;
    $('.opts', li).hidden = false;
    $('select', li).value = u.slowmo || 'auto';
    $('.slow', li).hidden = !(m.fps >= 50);
    if (document.activeElement !== $('.note', li)) $('.note', li).value = u.note || '';
  }
  if (u.kind === 'music') {
    $('.thumb', li).hidden = true;
    li.style.gridTemplateColumns = '1fr';
    $('.use input', li).checked = state.musicId === u.id;
    $('.use input', li).type = 'radio';
    $('.use input', li).name = 'music';
    $('.use input', li).onchange = () => { state.musicId = u.id; updateMake(); };
  }
  updateMake();
}

async function loadUploads() {
  const list = await api('uploads');
  for (const u of list.reverse()) {
    if (u.kind === 'music' && u.status === 'ready' && !state.musicId) state.musicId = u.id;
    renderUpload(u);
  }
}

async function removeUpload(id) {
  const u = state.uploads.get(id);
  if (!u || !confirm(`Remove ${u.name}?`)) return;
  try {
    await api(`uploads/${id}`, { method: 'DELETE' });
  } catch (err) {
    alert(err.message);
    return;
  }
  state.uploads.delete(id);
  document.getElementById('u-' + id)?.remove();
  if (state.musicId === id) state.musicId = null;
  const map = readResume();
  for (const k of Object.keys(map)) if (map[k] === id) delete map[k];
  writeResume(map);
  updateMake();
}

function enqueue(files, kind) {
  for (const f of files) state.queue.push({ file: f, kind });
  pump();
}

async function pump() {
  if (state.uploading) return;
  state.uploading = true;
  keepAwake(true);
  while (state.queue.length) {
    const { file, kind } = state.queue.shift();
    try {
      await uploadFile(file, kind);
    } catch (err) {
      console.error(err);
    }
  }
  state.uploading = false;
  keepAwake(false);
  updateMake();
}

function putChunk(id, offset, blob, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', `api/uploads/${id}?offset=${offset}`);
    xhr.setRequestHeader('X-Shorts', '1');
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = (e) => onProgress(e.loaded);
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* ignore */ }
      if (xhr.status === 200) resolve(data);
      else reject(Object.assign(new Error((data && (data.detail || data.error)) || `HTTP ${xhr.status}`), { status: xhr.status, data }));
    };
    xhr.onerror = () => reject(Object.assign(new Error('Network error'), { status: 0 }));
    xhr.ontimeout = () => reject(Object.assign(new Error('Timed out'), { status: 0 }));
    xhr.timeout = 180000;
    xhr.send(blob);
  });
}

async function uploadFile(file, kind) {
  const key = fileKey(file, kind);
  const map = readResume();
  let u = null;
  if (map[key]) {
    try { u = await api(`uploads/${map[key]}`); } catch { u = null; }
    if (u && u.status === 'ready') { renderUpload(u); return; }
    if (u && u.status !== 'uploading') u = null;
  }
  if (!u) {
    try {
      u = await api('uploads', { json: { name: file.name || 'clip.mov', size: file.size, kind } });
    } catch (err) {
      const tmp = { id: 'tmp' + Date.now(), kind, name: file.name, size: file.size, received: 0, status: 'error', error: err.message };
      renderUpload(tmp);
      return;
    }
    map[key] = u.id;
    writeResume(map);
  }
  const chunk = u.chunk_size || state.health?.limits?.chunk_bytes || 16 * 1024 * 1024;
  let offset = u.received || 0;
  let tries = 0;
  renderUpload(u, { active: true, sent: offset });
  while (offset < file.size) {
    const end = Math.min(file.size, offset + chunk);
    try {
      const r = await putChunk(u.id, offset, file.slice(offset, end), (loaded) => renderUpload(u, { active: true, sent: offset + loaded }));
      offset = r.received;
      u.received = offset;
      tries = 0;
    } catch (err) {
      if (err.status === 409 && err.data && typeof err.data.received === 'number') { offset = err.data.received; continue; }
      if (err.status === 401 || err.status === 404 || err.status === 413 || tries >= 8) {
        renderUpload(u, { error: `Upload stopped: ${err.message}. Pick the file again to resume.` });
        return;
      }
      tries += 1;
      renderUpload(u, { active: true, sent: offset, error: `Connection dropped, retrying (${tries}/8)…` });
      await new Promise((r) => setTimeout(r, Math.min(30000, 1000 * 2 ** tries)));
    }
  }
  try {
    const done = await api(`uploads/${u.id}/complete`, { method: 'POST' });
    delete map[key];
    writeResume(map);
    if (kind === 'music') state.musicId = done.id;
    renderUpload(done);
  } catch (err) {
    renderUpload({ ...u, status: 'error', error: err.message });
  }
}

$('#pick-video').addEventListener('change', (e) => { enqueue([...e.target.files], 'video'); e.target.value = ''; });
$('#pick-music').addEventListener('change', (e) => { enqueue([...e.target.files].slice(0, 1), 'music'); e.target.value = ''; });
window.addEventListener('beforeunload', (e) => { if (state.uploading) { e.preventDefault(); e.returnValue = ''; } });

// ------------------------------------------------------------------ direction
document.querySelectorAll('.group').forEach((g) => {
  g.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-value]');
    if (!b || b.disabled) return;
    g.querySelectorAll('button[data-value]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  });
});
function brief() {
  const pick = (name) => $(`.group[data-name="${name}"] button[aria-pressed="true"]`)?.dataset.value;
  return {
    vibe: pick('vibe'), length: Number(pick('length')), prompt: $('#prompt').value.trim(), captions: pick('captions'),
    look: pick('look'), fps: Number(pick('fps')), effects: pick('effects'), music_mode: pick('music_mode'),
    ai_transitions: $('#ai_transitions').checked, clean_audio: $('#clean_audio').checked,
    punch_in: $('#punch_in').checked, hook_title: $('#hook_title').checked,
  };
}
function selectedClips() {
  return [...state.uploads.values()]
    .filter((u) => u.kind === 'video' && u.status === 'ready' && $('.use input', document.getElementById('u-' + u.id))?.checked)
    .sort((a, b) => a.created - b.created)
    .map((u) => u.id);
}
function updateMake() {
  const ids = selectedClips();
  const pending = [...state.uploads.values()].some((u) => u.status === 'uploading');
  $('#make').disabled = ids.length === 0;
  $('#make-note').textContent = ids.length === 0
    ? (pending ? 'Waiting for uploads to finish…' : 'Add at least one clip.')
    : `${ids.length} clip${ids.length > 1 ? 's' : ''} selected${pending ? ' · more still uploading' : ''}`;
}
$('#make').addEventListener('click', async () => {
  const ids = selectedClips();
  if (!ids.length) return;
  $('#make').disabled = true;
  try {
    const job = await api('jobs', { json: { upload_ids: ids, music_id: state.musicId, brief: brief() } });
    renderJob(job);
    $('#jobs-card').scrollIntoView({ behavior: 'smooth' });
    pollJobs();
  } catch (err) {
    alert(err.message);
  } finally {
    updateMake();
  }
});

// ------------------------------------------------------------------ jobs
function renderJob(j) {
  state.jobs.set(j.id, j);
  $('#jobs .empty')?.remove();
  let li = document.getElementById('j-' + j.id);
  if (!li) {
    li = $('#job-tpl').content.firstElementChild.cloneNode(true);
    li.id = 'j-' + j.id;
    $('.logbtn', li).addEventListener('click', async () => {
      const pre = $('.log', li);
      pre.hidden = !pre.hidden;
      if (!pre.hidden) pre.textContent = (await api(`jobs/${j.id}`)).log || '(no log yet)';
    });
    $('.cancel', li).addEventListener('click', () => api(`jobs/${j.id}/cancel`, { method: 'POST' }).then(pollJobs));
    $('.del', li).addEventListener('click', async () => {
      if (!confirm('Delete this short?')) return;
      await api(`jobs/${j.id}`, { method: 'DELETE' }).catch((e) => alert(e.message));
      li.remove();
      state.jobs.delete(j.id);
    });
    $('.remix', li).addEventListener('click', async () => {
      try {
        const nj = await api(`jobs/${j.id}/remix`, { json: { brief: brief(), music_id: state.musicId } });
        renderJob(nj);
        pollJobs();
      } catch (e) { alert(e.message); }
    });
    $('.share', li).addEventListener('click', () => shareVideo(j.id));
    const list = $('#jobs');
    const newer = [...list.children].find((c) => (state.jobs.get(c.id.slice(2))?.created || 0) < j.created);
    list.insertBefore(li, newer || null);
  }
  const b = j.brief || {};
  $('.title', li).textContent = (j.result && j.result.title) || `${b.length}s ${b.vibe} short`;
  const pill = $('.pill', li);
  pill.className = 'pill ' + j.status;
  pill.textContent = j.status === 'running' ? `${Math.round(j.progress * 100)}%` : j.status;
  $('.stage', li).textContent = j.status === 'error' ? j.error : j.stage || (j.status === 'queued' ? 'Waiting for the worker…' : '');
  $('.bar', li).hidden = !(j.status === 'running' || j.status === 'queued');
  $('.bar i', li).style.width = Math.round(j.progress * 100) + '%';
  const active = j.status === 'running' || j.status === 'queued';
  $('.cancel', li).hidden = !active;
  $('.del', li).hidden = active;
  $('.remix', li).hidden = active;
  if (j.status === 'done' && j.result) {
    const res = $('.result', li);
    if (res.hidden) {
      res.hidden = false;
      const v = $('video', li);
      v.poster = `api/jobs/${j.id}/files/cover.jpg?t=${j.finished}`;
      v.src = `api/jobs/${j.id}/files/short.mp4?t=${j.finished}`;
      $('.save', li).href = `api/jobs/${j.id}/files/short.mp4?download=1`;
      const r = j.result;
      const post = $('.post', li);
      const text = [r.post_caption, (r.hashtags || []).map((t) => '#' + t).join(' ')].filter(Boolean).join('\n');
      post.replaceChildren();
      if (text) {
        post.append(text);
        const copy = document.createElement('button');
        copy.type = 'button';
        copy.textContent = 'Copy caption';
        copy.addEventListener('click', () => navigator.clipboard?.writeText(text).then(() => { copy.textContent = 'Copied'; }));
        post.append(document.createElement('br'), copy);
      }
      const warn = $('.warn', li);
      warn.replaceChildren(...(r.warnings || []).map((w) => Object.assign(document.createElement('li'), { textContent: w })));
      const meta = document.createElement('p');
      meta.className = 'muted small';
      meta.textContent = `${r.seconds}s · ${r.segments} shots · edited by ${r.planner === 'ollama' ? 'AI' : 'rules'}` +
        (r.transitions && (r.transitions.ai || r.transitions.still) ? ` · ${r.transitions.ai} AI + ${r.transitions.still} still-motion transitions` : '');
      post.prepend(meta);
    }
  }
}

async function shareVideo(id) {
  const url = `api/jobs/${id}/files/short.mp4`;
  try {
    const blob = await (await fetch(url, { credentials: 'same-origin' })).blob();
    const file = new File([blob], `short-${id}.mp4`, { type: 'video/mp4' });
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file] });
      return;
    }
  } catch (err) {
    if (err.name === 'AbortError') return;
  }
  window.location.href = url + '?download=1';
}

let pollTimer = null;
async function loadJobs() {
  const list = await api('jobs');
  list.forEach(renderJob);
  if (list.some((j) => j.status === 'running' || j.status === 'queued')) pollJobs();
}
function pollJobs() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    let list = [];
    try { list = await api('jobs'); } catch { /* retry below */ }
    list.forEach(renderJob);
    const active = list.some((j) => j.status === 'running' || j.status === 'queued');
    keepAwake(active || state.uploading);
    // refresh clip analysis state while the worker is busy
    if (active || [...state.uploads.values()].some((u) => u.analysis === 'running' || u.analysis === 'none')) loadUploads().catch(() => {});
    if (active) pollJobs();
  }, 2000);
}

// ------------------------------------------------------------------ start
(async () => {
  try {
    await api('me');
    await showApp();
    setInterval(loadHealth, 30000);
  } catch {
    showLogin();
  }
})();
