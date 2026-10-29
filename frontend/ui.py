"""
Frontend look-and-feel helpers for the Streamlit app (frontend/app.py).

Nothing in here affects predictions; it only draws things:
  inject_css()      global dark "security operations centre" styling
  hero()            animated page header with model status chips
  threat_dial()     animated result panel: radar scan -> speed-dial gauge -> verdict
  event_feed()      list of recent connections for the live replay tab

The dial is plain HTML/SVG/JavaScript rendered in an iframe with st.iframe,
so it needs no extra Python packages.
"""

import json

import streamlit as st

GREEN = "#22c55e"
RED = "#ef4444"
CYAN = "#22d3ee"
MUTED = "#94a3b8"

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

CSS = """
<style>
/* page background: dark navy with a faint cyan glow */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(900px 500px at 85% -10%, rgba(34,211,238,0.10), transparent 60%),
    radial-gradient(700px 400px at -10% 10%, rgba(168,85,247,0.08), transparent 60%),
    #070b14;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1200px; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1222 0%, #070b14 100%);
  border-right: 1px solid rgba(34,211,238,0.15);
}

/* header */
.hero {
  position: relative; display: flex; gap: 22px; align-items: center;
  padding: 26px 28px; margin-bottom: 18px; border-radius: 20px;
  background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(15,23,42,0.75));
  border: 1px solid rgba(34,211,238,0.25);
  box-shadow: 0 0 40px rgba(34,211,238,0.08), inset 0 0 30px rgba(34,211,238,0.04);
  overflow: hidden;
}
.hero::before {               /* light beam sweeping across the header */
  content: ""; position: absolute; top: 0; left: -40%; width: 40%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(34,211,238,0.10), transparent);
  animation: sweep 6s ease-in-out infinite;
}
@keyframes sweep { 0% { left: -40%; } 60%, 100% { left: 110%; } }
.hero-icon {
  position: relative; flex: 0 0 74px; height: 74px; border-radius: 50%;
  display: grid; place-items: center; font-size: 36px;
  background: radial-gradient(circle, rgba(34,211,238,0.25), rgba(34,211,238,0.04) 70%);
  border: 1px solid rgba(34,211,238,0.45);
}
.hero-icon::after {           /* radar ping around the shield */
  content: ""; position: absolute; inset: -1px; border-radius: 50%;
  border: 2px solid rgba(34,211,238,0.6); animation: ping 2.2s ease-out infinite;
}
@keyframes ping { 0% { transform: scale(1); opacity: 0.9; } 100% { transform: scale(1.7); opacity: 0; } }
.hero-kicker { font-size: 0.72rem; letter-spacing: 0.22em; color: #67e8f9; font-weight: 600; }
.hero-title {
  margin: 2px 0 4px 0; font-size: 2.1rem; font-weight: 800; line-height: 1.15;
  background: linear-gradient(90deg, #e2e8f0, #67e8f9 45%, #a78bfa 90%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: #94a3b8; font-size: 0.95rem; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.chip {
  font-size: 0.78rem; padding: 4px 11px; border-radius: 999px; color: #cbd5e1;
  background: rgba(30,41,59,0.8); border: 1px solid rgba(148,163,184,0.25);
}
.chip.live { color: #86efac; border-color: rgba(34,197,94,0.5); }
.chip.live::before {
  content: ""; display: inline-block; width: 7px; height: 7px; margin-right: 6px;
  border-radius: 50%; background: #22c55e; animation: blink 1.4s infinite;
}
@keyframes blink { 50% { opacity: 0.25; } }

/* metric tiles */
[data-testid="stMetric"] {
  background: linear-gradient(160deg, rgba(30,41,59,0.75), rgba(15,23,42,0.75));
  border: 1px solid rgba(148,163,184,0.18); border-radius: 14px; padding: 12px 16px;
  transition: transform .15s ease, border-color .15s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: rgba(34,211,238,0.5); }
[data-testid="stMetricLabel"] { color: #94a3b8; }
[data-testid="stMetric"] { padding: 10px 12px; }
.block-container [data-testid="stMetricValue"] p { font-size: clamp(1.1rem, 2.2vw, 1.9rem) !important; }
.block-container [data-testid="stMetricLabel"] p { font-size: 0.8rem; }
[data-testid="stSidebar"] [data-testid="stMetric"] { padding: 10px 12px; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] p { font-size: 1.2rem !important; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
  background: rgba(15,23,42,0.7); border: 1px solid rgba(148,163,184,0.18);
  border-radius: 10px 10px 0 0; padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(180deg, rgba(34,211,238,0.18), rgba(15,23,42,0.7));
  border-color: rgba(34,211,238,0.55);
}

/* primary buttons */
button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(90deg, #0891b2, #7c3aed); border: none; color: white;
  box-shadow: 0 0 18px rgba(34,211,238,0.25); transition: box-shadow .2s, transform .1s;
}
button[data-testid="stBaseButton-primary"]:hover {
  box-shadow: 0 0 28px rgba(34,211,238,0.5); transform: translateY(-1px); color: white;
}

/* panels made with st.container(border=True) */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(15,23,42,0.55); border-radius: 16px;
}

.section-title {
  font-size: 0.75rem; letter-spacing: 0.18em; color: #67e8f9; font-weight: 700;
  text-transform: uppercase; margin: 4px 0 8px 0;
}
.note { color: #94a3b8; font-size: 0.85rem; }

/* live replay feed */
.feed { display: flex; flex-direction: column; gap: 6px; font-family: ui-monospace, Menlo, monospace; }
.feed-row {
  display: grid; grid-template-columns: 48px 92px 1fr 64px 22px; align-items: center; gap: 10px;
  padding: 7px 12px; border-radius: 10px; font-size: 0.8rem;
  background: rgba(30,41,59,0.6); border-left: 3px solid #475569;
}
.feed-row.attack { border-left-color: #ef4444; background: rgba(127,29,29,0.25); }
.feed-row.normal { border-left-color: #22c55e; }
.feed-row.new { animation: slidein .35s ease-out; }
@keyframes slidein { from { opacity: 0; transform: translateX(-16px); } }
.badge { font-weight: 700; font-size: 0.72rem; letter-spacing: 0.06em; }
.badge.attack { color: #fca5a5; } .badge.normal { color: #86efac; }
.bar { height: 6px; border-radius: 3px; background: #1e293b; overflow: hidden; }
.bar > span { display: block; height: 100%; border-radius: 3px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle, chips):
    """Animated header. `chips` is a list of short strings; the first one is shown as 'live'."""
    chip_html = "".join(
        f'<span class="chip{" live" if i == 0 else ""}">{c}</span>' for i, c in enumerate(chips))
    st.markdown(f"""
    <div class="hero">
      <div class="hero-icon">🛡️</div>
      <div>
        <div class="hero-kicker">MACHINE LEARNING · NETWORK SECURITY</div>
        <div class="hero-title">{title}</div>
        <div class="hero-sub">{subtitle}</div>
        <div class="chips">{chip_html}</div>
      </div>
    </div>""", unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Threat dial: radar scan -> speed-dial gauge -> verdict
# ---------------------------------------------------------------------------

DIAL_TEMPLATE = """
<!doctype html><html><head><style>
  html, body { margin: 0; background: transparent; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .panel {
    position: relative; height: __HEIGHT__px; box-sizing: border-box; border-radius: 18px;
    padding: 14px 16px; overflow: hidden; color: #e2e8f0;
    background: radial-gradient(circle at 50% 45%, rgba(34,211,238,0.10), transparent 60%),
                linear-gradient(160deg, #0f172a, #0b1222);
    border: 1px solid rgba(34,211,238,0.25);
  }
  .panel.alert { animation: alarm 1.6s ease-in-out infinite; }
  @keyframes alarm {
    0%,100% { box-shadow: 0 0 0 rgba(239,68,68,0); border-color: rgba(239,68,68,0.45); }
    50%     { box-shadow: 0 0 34px rgba(239,68,68,0.45); border-color: rgba(239,68,68,0.95); }
  }
  .panel.safe { box-shadow: 0 0 26px rgba(34,197,94,0.18); border-color: rgba(34,197,94,0.5); }
  .top { display: flex; justify-content: space-between; font-size: 11px; letter-spacing: .18em; color: #67e8f9; font-weight: 700; }
  .status { color: #94a3b8; }
  .stage { position: relative; height: 205px; margin-top: 4px; }

  /* radar */
  .radar { position: absolute; left: 50%; top: 50%; width: 180px; height: 180px; margin: -90px 0 0 -90px;
    border-radius: 50%; border: 1px solid rgba(34,211,238,0.5); transition: opacity .4s, transform .4s;
    background: repeating-radial-gradient(circle, rgba(34,211,238,0.16) 0 1px, transparent 1px 30px),
                radial-gradient(circle, rgba(34,211,238,0.12), rgba(8,47,73,0.35)); }
  .radar::before { content: ""; position: absolute; inset: 0; border-radius: 50%;
    background: conic-gradient(from 0deg, rgba(34,211,238,0.55), rgba(34,211,238,0) 70deg);
    animation: spin 1.1s linear infinite; }
  .radar::after { content: ""; position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: rgba(34,211,238,0.25); }
  @keyframes spin { to { transform: rotate(360deg); } }
  .blip { position: absolute; width: 7px; height: 7px; border-radius: 50%; background: #67e8f9;
    box-shadow: 0 0 10px #22d3ee; animation: blip 1.1s ease-in-out infinite; }
  @keyframes blip { 0%,100% { opacity: .15; } 50% { opacity: 1; } }
  .scanline { position: absolute; left: 0; right: 0; bottom: -2px; text-align: center; font-size: 12px; color: #67e8f9;
    font-family: ui-monospace, Menlo, monospace; transition: opacity .3s; }

  /* gauge */
  svg { position: absolute; left: 50%; top: 0; width: 100%; max-width: 320px; transform: translateX(-50%);
        opacity: 0; transition: opacity .45s; }
  .readout { text-align: center; margin-top: -6px; opacity: 0; transition: opacity .45s; }
  .num { font-size: 40px; font-weight: 800; font-variant-numeric: tabular-nums; line-height: 1; }
  .lbl { font-size: 11px; letter-spacing: .2em; color: #94a3b8; margin-top: 4px; }
  .verdict { margin: 12px auto 0; width: max-content; padding: 8px 18px; border-radius: 999px; font-weight: 800;
    letter-spacing: .08em; font-size: 15px; opacity: 0; transform: scale(.6);
    transition: opacity .35s, transform .45s cubic-bezier(.34,1.56,.64,1); }
  .verdict.show { opacity: 1; transform: scale(1); }
  .verdict.attack { color: #fecaca; background: rgba(239,68,68,0.18); border: 1px solid rgba(239,68,68,0.7); }
  .verdict.normal { color: #bbf7d0; background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.6); }
  .detail { text-align: center; font-size: 12px; color: #94a3b8; margin-top: 8px; opacity: 0; transition: opacity .4s; }
</style></head><body>
<div class="panel" id="panel">
  <div class="top"><span>THREAT ANALYSIS</span><span class="status" id="status">STANDBY</span></div>
  <div class="stage">
    <div class="radar" id="radar">
      <div class="blip" style="left:38px;top:52px"></div>
      <div class="blip" style="left:122px;top:40px;animation-delay:.35s"></div>
      <div class="blip" style="left:104px;top:128px;animation-delay:.7s"></div>
    </div>
    <div class="scanline" id="scanline">initialising…</div>
    <svg id="gauge" viewBox="0 0 320 200">
      <defs>
        <linearGradient id="grad" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0" stop-color="#22c55e"/><stop offset="0.42" stop-color="#a3e635"/>
          <stop offset="0.55" stop-color="#f59e0b"/><stop offset="1" stop-color="#ef4444"/>
        </linearGradient>
        <filter id="glow"><feGaussianBlur stdDeviation="3.5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      <path d="M 30 170 A 130 130 0 0 1 290 170" stroke="#1e293b" stroke-width="20" fill="none" stroke-linecap="round"/>
      <path id="arc" d="M 30 170 A 130 130 0 0 1 290 170" stroke="url(#grad)" stroke-width="20" fill="none"
            stroke-linecap="round" pathLength="100" stroke-dasharray="100" stroke-dashoffset="100" filter="url(#glow)"/>
      <g id="ticks"></g>
      <g id="needle" transform="rotate(-90 160 170)">
        <polygon points="156,170 164,170 160,52" fill="#e2e8f0" filter="url(#glow)"/>
        <circle cx="160" cy="170" r="12" fill="#0f172a" stroke="#22d3ee" stroke-width="3"/>
      </g>
    </svg>
  </div>
  <div class="readout" id="readout">
    <div class="num" id="num">0%</div>
    <div class="lbl" id="lbl">ATTACK PROBABILITY</div>
  </div>
  <div class="verdict" id="verdict"></div>
  <div class="detail" id="detail"></div>
</div>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const CX = 160, CY = 170;

// tick marks every 10 %, labels at 0 / 50 / 100, and the decision threshold
let ticks = "";
for (let i = 0; i <= 10; i++) {
  const a = Math.PI - Math.PI * i / 10, r1 = i % 5 ? 104 : 98, r2 = 112;
  ticks += `<line x1="${CX + r1 * Math.cos(a)}" y1="${CY - r1 * Math.sin(a)}" x2="${CX + r2 * Math.cos(a)}"
            y2="${CY - r2 * Math.sin(a)}" stroke="#475569" stroke-width="${i % 5 ? 1.5 : 2.5}"/>`;
  if (i % 5 === 0) ticks += `<text x="${CX + 84 * Math.cos(a)}" y="${CY - 84 * Math.sin(a) + 4}" fill="#64748b"
            font-size="11" text-anchor="middle">${i * 10}</text>`;
}
if (D.threshold !== null) {
  const a = Math.PI - Math.PI * D.threshold;
  ticks += `<line x1="${CX + 120 * Math.cos(a)}" y1="${CY - 120 * Math.sin(a)}" x2="${CX + 146 * Math.cos(a)}"
            y2="${CY - 146 * Math.sin(a)}" stroke="#f8fafc" stroke-width="2" stroke-dasharray="3 2"/>
            <text x="${CX}" y="14" fill="#94a3b8" font-size="10" text-anchor="middle">decision threshold</text>`;
}
$("ticks").innerHTML = ticks;

const clamp = v => Math.max(-0.02, Math.min(1.02, v));
function setDial(v) {
  v = clamp(v);
  $("needle").setAttribute("transform", `rotate(${-90 + 180 * v} ${CX} ${CY})`);
  $("arc").setAttribute("stroke-dashoffset", 100 - 100 * Math.max(0, Math.min(1, v)));
  $("arc").style.opacity = v < 0.01 ? 0 : 1;
  $("num").textContent = Math.round(Math.max(0, Math.min(1, v)) * 100) + "%";
  $("num").style.color = v >= 0.5 ? "#f87171" : (v >= 0.3 ? "#fbbf24" : "#4ade80");
}
const easeInOut = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
const easeOutBack = t => 1 + 2.2 * Math.pow(t - 1, 3) + 1.2 * Math.pow(t - 1, 2);  // small overshoot
function tween(from, to, ms, ease) {
  return new Promise(done => {
    const t0 = performance.now();
    (function frame(now) {
      const t = Math.min(1, (now - t0) / ms);
      setDial(from + (to - from) * ease(t));
      t < 1 ? requestAnimationFrame(frame) : done();
    })(t0);
  });
}
const wait = ms => new Promise(r => setTimeout(r, ms));

function showVerdict() {
  const v = $("verdict");
  v.className = "verdict show " + (D.is_attack ? "attack" : "normal");
  v.textContent = D.is_attack ? "🚨 " + D.verdict : "✅ " + D.verdict;
  $("detail").textContent = D.detail; $("detail").style.opacity = 1;
  $("panel").className = "panel " + (D.is_attack ? "alert" : "safe");
  $("status").textContent = D.is_attack ? "ALERT" : "CLEAR";
  $("status").style.color = D.is_attack ? "#f87171" : "#4ade80";
}

async function run() {
  setDial(D.previous);
  if (D.calibrate) {
    // 1) radar scan, showing the features being checked
    $("status").textContent = "SCANNING";
    let i = 0;
    const ticker = setInterval(() => {
      $("scanline").textContent = "checking " + D.features[i % D.features.length] + " …"; i++;
    }, 90);
    await wait(1300);
    clearInterval(ticker);
    $("radar").style.opacity = 0; $("radar").style.transform = "scale(.6)"; $("scanline").style.opacity = 0;
    // 2) gauge appears and does a full calibration sweep, like a car dashboard
    $("gauge").style.opacity = 1; $("readout").style.opacity = 1;
    $("status").textContent = "CALIBRATING"; $("lbl").textContent = "CALIBRATING";
    await tween(0, 1, 650, easeInOut);
    await wait(120);
    await tween(1, 0, 450, easeInOut);
    $("lbl").textContent = "ATTACK PROBABILITY";
    // 3) settle on the model's value
    await tween(0, D.value, 900, easeOutBack);
  } else {
    $("radar").style.display = "none"; $("scanline").style.display = "none";
    $("gauge").style.opacity = 1; $("readout").style.opacity = 1;
    await tween(D.previous, D.value, 750, easeOutBack);  // needle moves up or down from the last value
  }
  setDial(D.value);
  showVerdict();
}
run();
</script>
</body></html>
"""


def threat_dial(value, is_attack, verdict, detail, features, previous=0.0, calibrate=True,
                threshold=0.5, height=400):
    """
    Animated result panel.
      value      model's attack probability (0-1)
      previous   where the needle starts (last value shown), so it moves up or down
      calibrate  True: radar scan + full calibration sweep first (after pressing Analyse)
                 False: needle just moves from `previous` to `value` (live slider changes)
      threshold  drawn on the dial; None for multiclass, where the decision is not a single threshold
    """
    data = {"value": float(value), "previous": float(previous), "is_attack": bool(is_attack),
            "verdict": verdict, "detail": detail, "features": list(features),
            "calibrate": bool(calibrate), "threshold": threshold}
    html = (DIAL_TEMPLATE.replace("__DATA__", json.dumps(data))
            .replace("__HEIGHT__", str(height - 8)))
    st.iframe(html, height=height)


def standby_panel(message, height=400):
    """Idle panel shown before the first scan: a slowly turning radar and an instruction."""
    st.markdown(f"""
    <div style="height:{height - 8}px;border-radius:18px;border:1px dashed rgba(34,211,238,0.35);
                background:linear-gradient(160deg,#0f172a,#0b1222);display:flex;flex-direction:column;
                align-items:center;justify-content:center;gap:22px;color:#94a3b8;text-align:center;">
      <div style="position:relative;width:150px;height:150px;border-radius:50%;
                  border:1px solid rgba(34,211,238,0.35);overflow:hidden;
                  background:repeating-radial-gradient(circle,rgba(34,211,238,0.12) 0 1px,transparent 1px 25px);">
        <div style="position:absolute;inset:0;border-radius:50%;animation:spin 4s linear infinite;
                    background:conic-gradient(from 0deg,rgba(34,211,238,0.35),rgba(34,211,238,0) 60deg);"></div>
      </div>
      <div style="font-size:0.9rem;max-width:260px;">{message}</div>
    </div>
    <style>@keyframes spin {{ to {{ transform: rotate(360deg); }} }}</style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Live replay feed
# ---------------------------------------------------------------------------

def event_feed(events):
    """events: newest first, dicts with n, prediction, probability, correct."""
    rows = []
    for i, e in enumerate(events):
        kind = "normal" if e["prediction"] == "Normal" else "attack"
        color = RED if kind == "attack" else GREEN
        rows.append(
            f'<div class="feed-row {kind}{" new" if i == 0 else ""}">'
            f'<span>#{e["n"]:03d}</span>'
            f'<span class="badge {kind}">{e["prediction"].upper()}</span>'
            f'<div class="bar"><span style="width:{e["probability"] * 100:.0f}%;background:{color}"></span></div>'
            f'<span>{e["probability"]:.0%}</span>'
            f'<span title="matches the true class">{"✔" if e["correct"] else "✘"}</span></div>')
    st.markdown(f'<div class="feed">{"".join(rows)}</div>', unsafe_allow_html=True)
