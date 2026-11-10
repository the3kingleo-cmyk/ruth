// Ruth's interface. Plain JS, no libraries, nothing loaded from the internet.
"use strict";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  // While she is in the middle of a dream step the app answers 503 "she is
  // dreaming". That is a pause, not a failure: wait and ask again.
  for (let attempt = 0; ; attempt++) {
    const r = await fetch(path, opt);
    const data = await r.json().catch(() => ({}));
    if (r.status === 503 && attempt < 5) {
      await new Promise((ok) => setTimeout(ok, 1000 * (data.retry_in || 3)));
      continue;
    }
    if (!r.ok) throw new Error(data.error || r.statusText);
    return data;
  }
}

// ---------------------------------------------------------------- conversation
function bubble(kind, text, meta) {
  const el = document.createElement("div");
  el.className = "msg " + kind;
  el.textContent = text;
  if (meta) {
    const m = document.createElement("span");
    m.className = "meta";
    m.textContent = meta;
    el.appendChild(m);
  }
  $("log").appendChild(el);
  $("log").scrollTop = $("log").scrollHeight;
  return el;
}

function showThoughts(list) {
  const ol = $("thoughtList");
  ol.replaceChildren();
  for (const t of list || []) {
    const li = document.createElement("li");
    li.textContent = `${JSON.stringify(t.thought)}  · free energy ${t.free_energy}`;
    if (t.chosen) li.className = "chosen";
    ol.appendChild(li);
  }
}

$("say").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("msg").value.trim();
  if (!text) return;
  const priv = $("private").checked;
  $("msg").value = "";
  const send = e.submitter || document.querySelector("#say .primary");
  send.disabled = true;
  try {
    if (priv) {
      bubble("private", "\u{1F512} ••••••", "told in confidence");
      const r = await api("/api/talk", { text, private: true });
      bubble("system", `She will keep ${r.kept} bytes to herself.`);
    } else {
      bubble("you", text);
      const r = await api("/api/talk", { text });
      const shown = r.text || (r.withheld ? "" : "…");
      const b = bubble("her", shown, r.withheld ? "" : undefined);
      if (r.withheld) {
        const w = document.createElement("span");
        w.className = "meta withheld";
        w.textContent = "she stopped herself: something here is private";
        b.appendChild(w);
      }
      showThoughts(r.considered);
      if ($("voiceOn").checked && r.text) speak(r.text);
    }
  } catch (err) {
    bubble("system", "Something went wrong: " + err.message);
  } finally {
    send.disabled = false;
    $("msg").focus();
  }
});
$("msg").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("say").requestSubmit(); }
});
$("good").onclick = () => api("/api/feedback", { good: true }).then(() => bubble("system", "You told her that was good."));
$("bad").onclick = () => api("/api/feedback", { good: false }).then(() => bubble("system", "You told her that wasn't good."));

$("sleepBtn").onclick = async () => {
  const b = $("sleepBtn");
  b.disabled = true; b.textContent = "Sleeping…";
  try {
    const r = await api("/api/sleep", {});
    const c = r.consolidation || {}, n = r.nightmares || {}, p = r.policy || {};
    bubble("system", `She slept: ${c.longterm ?? "?"} memory basins, rehearsed ${n.rehearsed ?? 0} confidences ` +
      `(${n.strengthened ?? 0} strengthened), caution now ${(p.adopted_caution ?? r.temperament.caution).toFixed(2)}.`);
    loadDreams();
  } catch (err) { bubble("system", "She couldn't sleep: " + err.message); }
  b.disabled = false; b.textContent = "Sleep";
};

async function loadDreams() {
  const list = await api("/api/dreams");
  const box = $("dreams");
  if (!list.length) return;
  box.replaceChildren();
  for (const night of list.slice(-3).reverse()) {
    for (const d of night.dreams || []) {
      const el = document.createElement("div");
      el.className = "dream";
      el.textContent = d.dream;
      const m = document.createElement("div");
      m.className = "meta";
      m.textContent = `${new Date(night.t * 1000).toLocaleString()} · seeded by "${d.seed.trim()}"`;
      el.appendChild(m);
      box.appendChild(el);
    }
  }
}

// ---------------------------------------------------------------- mind view
function drawNeurons(x) {
  const c = $("neurons"), g = c.getContext("2d");
  const cols = 32, rows = Math.max(1, Math.ceil(x.length / cols));
  const w = c.width / cols, h = c.height / rows;
  g.clearRect(0, 0, c.width, c.height);
  const pos = css("--accent"), neg = css("--private");
  x.forEach((v, i) => {
    g.globalAlpha = Math.min(1, Math.abs(v)) * 0.9 + 0.05;
    g.fillStyle = v >= 0 ? pos : neg;
    g.fillRect((i % cols) * w + 1, Math.floor(i / cols) * h + 1, w - 2, h - 2);
  });
  g.globalAlpha = 1;
}

function drawSurprise(s) {
  const c = $("surprise"), g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  if (s.length < 2) return;
  const max = Math.max(1, ...s);
  g.strokeStyle = css("--accent"); g.lineWidth = 2; g.beginPath();
  s.forEach((v, i) => {
    const px = (i / (s.length - 1)) * c.width, py = c.height - 4 - (v / max) * (c.height - 8);
    i ? g.lineTo(px, py) : g.moveTo(px, py);
  });
  g.stroke();
}

function renderTraits(t) {
  const box = $("traits");
  box.replaceChildren();
  for (const [k, v] of Object.entries(t)) {
    const row = document.createElement("div");
    row.className = "trait";
    const name = document.createElement("span"); name.textContent = k;
    const bar = document.createElement("div"); bar.className = "bar";
    const fill = document.createElement("i");
    const frac = k === "mood" ? (v + 1) / 2 : v;
    fill.style.width = Math.round(Math.max(0, Math.min(1, frac)) * 100) + "%";
    bar.appendChild(fill);
    const val = document.createElement("span"); val.textContent = v.toFixed(2);
    row.append(name, bar, val);
    box.appendChild(row);
  }
}

function renderMemory(s) {
  const dl = $("memory");
  dl.replaceChildren();
  const rows = [["neurons", `${s.neurons}${s.grown ? ` (+${s.grown} grown)` : ""}`], ["synapses", s.synapses],
    ["moments lived", s.moments], ["conversations", s.conversations], ["nights slept", s.sleeps],
    ["working memories", s.working], ["long-term basins", s.longterm], ["confidences kept", s.cues], ["background replays", s.background_replays]];
  for (const [k, v] of rows) {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v;
    dl.append(dt, dd);
  }
}

async function poll() {
  try {
    const s = await api("/api/state");
    const born = s.identity && s.identity.born ? new Date(s.identity.born.replace(/([+-]\d\d)(\d\d)$/, "$1:$2")) : null;
    $("vitals").textContent = `v${s.version}` + (born && !isNaN(born) ? ` · born ${born.toLocaleDateString()}` : "") +
      ` · ${s.moments.toLocaleString()} moments lived` + (s.dreaming ? " · dreaming" : "");
    drawNeurons(s.activity);
    $("coreLabel").textContent = `cortex · ${s.neurons} ${s.cell} neurons`;
    drawSurprise(s.surprise);
    $("sNow").textContent = s.surprise.length ? s.surprise[s.surprise.length - 1].toFixed(3) : "";
    $("valence").textContent = `sense of privacy ${s.valence.toFixed(2)}`;
    renderTraits(s.temperament);
    renderMemory(s);
  } catch (e) {
    $("vitals").textContent = e instanceof TypeError
      ? "resting (the app server is not running)"
      : "she is dreaming; she'll be back in a moment";
  }
}

// ---------------------------------------------------------------- senses
let audio = null;
$("micBtn").onclick = async () => {
  if (audio) {
    audio.stream.getTracks().forEach((t) => t.stop());
    audio.ctx.close(); audio = null;
    $("micBtn").textContent = "Ears: off"; $("micBtn").classList.remove("on");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const factor = Math.max(1, Math.round(ctx.sampleRate / 16000));
    let buf = [];
    proc.onaudioprocess = (ev) => {
      const d = ev.inputBuffer.getChannelData(0);
      for (let i = 0; i + factor <= d.length; i += factor) {
        let s = 0; for (let k = 0; k < factor; k++) s += d[i + k];
        buf.push(+(s / factor).toFixed(4));
      }
      if (buf.length >= 4000) {
        const samples = buf; buf = [];
        api("/api/hear", { rate: Math.round(ctx.sampleRate / factor), samples }).catch(() => {});
      }
    };
    src.connect(proc); proc.connect(ctx.destination);
    audio = { stream, ctx };
    $("micBtn").textContent = "Ears: listening"; $("micBtn").classList.add("on");
  } catch (e) { bubble("system", "Microphone unavailable: " + e.message); }
};

let vision = null;
$("camBtn").onclick = async () => {
  if (vision) {
    clearInterval(vision.timer); vision.stream.getTracks().forEach((t) => t.stop()); vision = null;
    $("cam").hidden = true;
    $("camBtn").textContent = "Eyes: off"; $("camBtn").classList.remove("on");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 160, height: 120 } });
    const v = $("cam"); v.srcObject = stream; v.hidden = false; await v.play();
    const c = $("retina"), g = c.getContext("2d", { willReadFrequently: true });
    const timer = setInterval(() => {
      g.drawImage(v, 0, 0, c.width, c.height);
      const px = g.getImageData(0, 0, c.width, c.height).data, grey = [];
      for (let i = 0; i < px.length; i += 4) grey.push(+((px[i] * 0.299 + px[i + 1] * 0.587 + px[i + 2] * 0.114) / 255).toFixed(3));
      api("/api/see", { w: c.width, h: c.height, pixels: grey, t: performance.now() / 1000 }).catch(() => {});
    }, 125);
    vision = { stream, timer };
    $("camBtn").textContent = "Eyes: watching"; $("camBtn").classList.add("on");
  } catch (e) { bubble("system", "Camera unavailable: " + e.message); }
};

async function speak(text) {
  try {
    const r = await api("/api/voice", { text });
    if (!r.samples.length) return;
    const ctx = new AudioContext({ sampleRate: r.rate });
    const b = ctx.createBuffer(1, r.samples.length, r.rate);
    b.copyToChannel(Float32Array.from(r.samples), 0);
    const s = ctx.createBufferSource(); s.buffer = b; s.connect(ctx.destination);
    s.onended = () => ctx.close(); s.start();
  } catch (e) { /* voice is optional */ }
}

bubble("system", "Ruth learns from everything you say here, every moment, on this computer only.");
poll(); loadDreams().catch(() => {});
setInterval(poll, 1500);
