import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { issuesFor, issueRanges, reviewCounts, timingExplanation, framingSummary, shotVerdict } from "./review.mjs";

const sheet = document.createElement("link");
sheet.rel = "stylesheet";
sheet.href = new URL("./platelock_review.css", import.meta.url).href;
document.head.append(sheet);

const COLORS = { copy: "#5a5a5a", interp: "#ffa028", generated: "#be6eff", missing: "#e63c3c", timing: "#ffe14a", moved: "#ff5050" };
const MODES = [["three", "Three panels", "Plate / original delivery / repaired output"], ["wipe", "Wipe", "Drag the line across the frame"],
  ["flicker", "Flicker", "Click or press space to switch between plate and result"],
  ["overlay", "Overlay", "Plate and generation at 50% opacity"],
  ["edges", "Edges", "Outlines: plate green, result magenta, white where they line up"]];
const MOVED_PX = 4;
const HELP = "←/→ frame · shift+←/→ 10 frames · N/P next/previous frame to check · space flicker · K play";
const LEGEND = [["copy", "from the video"], ["interp", "filled"], ["generated", "generated"], ["missing", "missing"],
  ["timing", "shows a neighbouring plate frame"], ["moved", `moved more than ${MOVED_PX} px`]];
const pad = (i) => String(i).padStart(6, "0");

function timecode(i, fps) {
  const r = Math.round(fps);
  const s = Math.floor(i / r);
  const two = (n) => String(n).padStart(2, "0");
  return `${two(Math.floor(s / 3600))}:${two(Math.floor(s / 60) % 60)}:${two(s % 60)}:${two(i % r)}`;
}

function el(tag, style = {}, props = {}) {
  const e = document.createElement(tag);
  Object.assign(e.style, style);
  Object.assign(e, props);
  return e;
}

function edges(data, w, h) {
  const g = new Float32Array(w * h);
  for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = data[j] * 0.299 + data[j + 1] * 0.587 + data[j + 2] * 0.114;
  const out = new Float32Array(w * h);
  let max = 1;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = g[i - w + 1] + 2 * g[i + 1] + g[i + w + 1] - g[i - w - 1] - 2 * g[i - 1] - g[i + w - 1];
      const gy = g[i + w - 1] + 2 * g[i + w] + g[i + w + 1] - g[i - w - 1] - 2 * g[i - w] - g[i - w + 1];
      out[i] = Math.hypot(gx, gy);
      if (out[i] > max) max = out[i];
    }
  }
  const sorted = Float32Array.from(out).sort();
  const p98 = sorted[Math.floor(sorted.length * 0.98)] || max;
  for (let i = 0; i < out.length; i++) out[i] = Math.min(1, out[i] / p98);
  return out;
}

class Viewer {
  constructor() {
    this.i = 0;
    this.mode = "three";
    this.filter = "all";
    this.b = "l";
    this.wipe = 0.5;
    this.showB = true;
    this.images = new Map();
    this.root = el("section", {}, { className: "pl-review", tabIndex: -1 });
    const header = el("header", {}, { className: "pl-header" });
    const heading = el("div");
    heading.append(el("div", {}, { className: "pl-brand", textContent: "PlateLock / Review" }));
    this.subtitle = el("div", {}, { className: "pl-subtitle", textContent: "Frame by frame, against your plate" });
    heading.append(this.subtitle);
    const close = el("button", {}, { textContent: "Close review  ×", onclick: () => this.close() });
    header.append(heading, close);
    const workspace = el("div", {}, { className: "pl-workspace" });
    const main = el("main", {}, { className: "pl-main" });
    const sidebar = el("aside", {}, { className: "pl-sidebar" });
    const bar = el("div", {}, { className: "pl-toolbar" });
    const button = (text, onclick, title = "") => {
      const b = el("button", {}, { textContent: text, onclick, title });
      bar.append(b);
      return b;
    };
    this.modeButtons = MODES.map(([id, label, title], n) => {
      const b = button(label, () => this.set({ mode: id }), `${title} (${n + 1})`);
      b.dataset.mode = id;
      return b;
    });
    this.zoom = 1;
    this.pan = {x: .5, y: .5};
    this.regions = false;
    button("Zoom +", () => {this.zoom = Math.min(8, this.zoom * 2); this.draw();}, "Zoom all panels together; drag to pan");
    button("Fit", () => {this.zoom = 1; this.pan = {x:.5,y:.5}; this.draw();});
    button("Show differences", () => {this.regions = !this.regions; this.draw();}, "Highlight appearance differences after alignment; these may be intentional");
    this.bButton = button("Matched original", () => this.mode === "three" ? this.set({threeMatched: !this.threeMatched}) : this.set({ b: this.b === "l" ? "g" : "l" }), "Switch between matched frames and delivery order");
    const picture = el("div", {}, { className: "pl-picture" });
    this.canvas = el("canvas");
    picture.append(this.canvas);
    const transport = el("div", {}, { className: "pl-transport" });
    this.playButton = el("button", {}, { textContent: "Play", onclick: () => this.play() });
    const prev = el("button", {}, { textContent: "←", title: "Previous frame", onclick: () => this.go(this.i - 1) });
    const next = el("button", {}, { textContent: "→", title: "Next frame", onclick: () => this.go(this.i + 1) });
    this.scrub = el("input", {}, { type: "range", min: 0, max: 0, value: 0, ariaLabel: "Plate frame", oninput: e => this.go(Number(e.target.value)) });
    this.frameNumber = el("span", {}, { className: "pl-frame" });
    transport.append(this.playButton, prev, next, this.scrub, this.frameNumber);
    this.info = el("div", {}, { className: "pl-frame" });
    this.timeline = el("canvas", { width: "100%", height: "80px", flexShrink: "0", cursor: "pointer" });
    this.legend = el("div", { display: "flex", gap: "12px", flexWrap: "wrap", fontSize: "11px", color: "#9caab6" });
    for (const [key, text] of [["interp", "Gap / fill"], ["timing", "Timing"], ["moved", "Position > 4 px"]]) {
      const item = el("span", {}, { textContent: text });
      item.prepend(el("span", {display: "inline-block", width: "7px", height: "7px", marginRight: "5px", background: COLORS[key], borderRadius: "50%"}));
      this.legend.append(item);
    }
    main.append(bar, picture, transport, this.info, this.timeline, this.legend);
    this.verdict = el("section", {}, {className:"pl-verdict"});
    this.verdictTitle = el("h2");
    this.verdictEvidence = el("p");
    this.verdictNext = el("p", {}, {className:"pl-next"});
    this.prepareRepair = el("button", {}, {textContent:"Prepare recommended repair", onclick:()=>{
      const decision=shotVerdict(this.m);
      if (decision.task && this.onPrepareRepair) {this.onPrepareRepair(decision.task);this.close();}
    }});
    this.verdict.append(this.verdictTitle,this.verdictEvidence,this.verdictNext,this.prepareRepair);
    this.overview = el("div", {}, { className: "pl-overview" });
    const shotDetails=el("details", {}, {className:"pl-details"});
    shotDetails.append(el("summary", {}, {textContent:"Shot measurements and limits"}),this.overview);
    this.trend = el("canvas", {width:"100%", height:"120px", cursor:"pointer"}, {width:560, height:240});
    this.trend.setAttribute("aria-label", "Residual movement across the shot. Horizontal red, vertical blue. Click to inspect a frame.");
    this.trend.onclick = e => {
      const series = this.m?.framing?.rendered_series || this.m?.framing?.series;
      if (!series?.length) return;
      const rect = this.trend.getBoundingClientRect();
      const target = (e.clientX - rect.left) / rect.width * (this.frames - 1);
      const nearest = series.reduce((a, b) => Math.abs(a.frame-target) < Math.abs(b.frame-target) ? a : b);
      this.set({i:nearest.frame});
    };
    this.diagnosis = el("div", {}, { className: "pl-diagnosis" });
    const advanced = el("details", {}, { className: "pl-details" });
    this.details = el("div");
    advanced.append(el("summary", {}, { textContent: "Measurement details" }), this.details);
    this.filterSelect = el("select", {}, { ariaLabel: "Filter issues" });
    for (const [value, text] of [["all", "Candidate issues"], ["timing", "Timing"], ["position", "Position"], ["gap", "Gaps / fills"], ["uncertain", "Not measurable"]]) {
      this.filterSelect.append(el("option", {}, { value, textContent: text }));
    }
    this.filterSelect.onchange = () => { this.filter = this.filterSelect.value; this.drawRanges(); };
    const navigation = el("div", {display:"flex", gap:"6px"});
    navigation.append(el("button", {}, {textContent:"← Previous", onclick:()=>this.next(-1)}), el("button", {}, {textContent:"Next →", onclick:()=>this.next(1)}));
    this.ranges = el("div", {}, { className: "pl-ranges" });
    sidebar.append(this.verdict, this.trend, shotDetails, el("div", {}, {className:"pl-label", textContent:"Selected frame"}), this.diagnosis, advanced,
      el("div", {}, {className:"pl-label", textContent:"Issue ranges"}), this.filterSelect, navigation, this.ranges);
    workspace.append(main, sidebar);
    this.help = el("footer", {}, { className: "pl-footer", textContent: HELP + " · Esc close" });
    this.root.append(header, workspace, this.help);
    this.root.addEventListener("keydown", (e) => this.key(e));
    const at = (e) => {
      const r = this.timeline.getBoundingClientRect();
      const x = (e.clientX - r.left - this.gutter) / (r.width - this.gutter);
      return Math.max(0, Math.min(this.frames - 1, Math.floor(x * this.frames)));
    };
    this.timeline.addEventListener("pointerdown", (e) => { this.timeline.setPointerCapture(e.pointerId); this.go(at(e)); });
    this.timeline.addEventListener("pointermove", (e) => {
      if (!this.m) return;
      if (e.buttons) this.go(at(e));
      else this.info.textContent = `hover: ${this.describe(at(e))}`;
    });
    this.timeline.addEventListener("pointerleave", () => this.drawInfo());
    const wipe = (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.set({ wipe: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) });
    };
    this.canvas.addEventListener("pointerdown", (e) => {
      this.root.focus();
      if (this.mode === "three") { this.canvas.setPointerCapture(e.pointerId); this.dragAt = {x:e.clientX,y:e.clientY}; }
      if (this.mode === "wipe") { this.canvas.setPointerCapture(e.pointerId); wipe(e); }
      if (this.mode === "flicker") this.set({ showB: !this.showB });
    });
    this.canvas.addEventListener("pointermove", (e) => {
      if (e.buttons && this.mode === "wipe") wipe(e);
      if (e.buttons && this.mode === "three" && this.dragAt) {
        const r = this.canvas.getBoundingClientRect();
        this.pan.x = Math.max(.5/this.zoom, Math.min(1-.5/this.zoom, this.pan.x-(e.clientX-this.dragAt.x)*3/r.width/this.zoom));
        this.pan.y = Math.max(.5/this.zoom, Math.min(1-.5/this.zoom, this.pan.y-(e.clientY-this.dragAt.y)/r.height/this.zoom));
        this.dragAt = {x:e.clientX,y:e.clientY}; this.draw();
      }
    });
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(this.root);
  }

  async load(subfolder) {
    this.sub = subfolder;
    const url = api.apiURL(`/view?type=output&subfolder=${encodeURIComponent(subfolder)}&filename=manifest.json&t=${Date.now()}`);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Review files unavailable (${response.status}). Run PlateLock again.`);
    this.m = await response.json();
    this.images.clear();
    this.cacheKey = null;
    this.frames = this.m.ops.length;
    this.scrub.max = Math.max(0, this.frames - 1);
    this.subtitle.textContent = `${this.m.name} · ${this.m.analysis_only ? "Analysis · originals unchanged" : "Repair review"}`;
    this.fps = this.m.fps.split("/").reduce((a, b) => a / b);
    this.generated = new Set(this.m.generated);
    this.issues = issuesFor(this.m);
    this.moved = this.issues.map(f => f.position);
    this.problems = this.issues.map(f => f.gap || f.timing || f.position);
    const count = key => this.issues.filter(f => f[key]).length;
    const counts = reviewCounts(this.issues);
    const tb = this.m.timebase;
    this.overview.textContent = `${this.m.analysis_only ? "ANALYSIS · no repairs applied" : "REPAIR · " + (this.m.repair_scope || "timing only")}\n\n${counts.candidates} frames with candidate issues\n${counts.notMeasurable} not fully measurable · not errors\n${counts.consistent} consistent in measured checks\n\n${count("timing")} timing · ${count("position")} position · ${count("gap")} proposed / filled gaps${tb ? `\n\nPlate ${tb.plate_fps} fps · AI ${tb.ai_fps} fps${tb.different ? "\nDifferent timebases. Comparing frame correspondence, not playback seconds." : ""}` : ""}`;
    const verdict=shotVerdict(this.m);
    this.verdict.dataset.state=verdict.state;
    this.verdictTitle.textContent=verdict.title;
    this.verdictEvidence.textContent=verdict.evidence;
    this.verdictNext.textContent=verdict.next;
    this.prepareRepair.hidden=!verdict.task || !this.onPrepareRepair;
    const framingText = framingSummary(this.m.framing);
    if (framingText) this.overview.textContent = framingText + `\n\nSecondary frame checks: ${count("timing")} timing candidates · ${count("gap")} gaps · ${this.m.framing.rendered_samples ?? this.m.framing.series?.length ?? 0} shared-feature samples / ${this.frames} frames`;
    if (this.m.timing_preserved)
      this.overview.textContent += `\n\nAll ${this.m.preserved_frames} AI frames and the original frame rate were preserved. Timing was not repaired or verified. Comparison covers ${this.m.compared_frames} frames shared with the plate.`;
    if (this.m.summary?.segments?.some(s => s.confidence == null || s.confidence < 2))
      this.overview.textContent += "\n\nSequence edits have weak evidence. Automatic repair is blocked until explicitly allowed.";
    const trendSeries = this.m.framing?.rendered_series || this.m.framing?.series || [];
    this.trend.style.display = trendSeries.length ? "block" : "none";
    if (trendSeries.length) {
      const c=this.trend.getContext("2d"), w=560, h=240;
      c.clearRect(0,0,w,h); c.fillStyle="#18232c";c.fillRect(0,0,w,h);
      const limit=Math.max(4,...trendSeries.flatMap(s=>[Math.abs(s.dx),Math.abs(s.dy)]));
      c.font="20px sans-serif";c.fillStyle="#c1cdd7";
      c.fillText(`Residual movement · ±${limit.toFixed(0)} px`,12,26);
      c.strokeStyle="#53606c";c.beginPath();c.moveTo(12,120);c.lineTo(548,120);c.stroke();
      for (const [key,color,label,x] of [["dx","#ff947c","X",12],["dy","#72c9fa","Y",60]]) {
        c.strokeStyle=color;c.lineWidth=3;c.beginPath();
        trendSeries.forEach((s,j)=>{const x=12+s.frame/Math.max(1,this.frames-1)*536,y=120-s[key]/limit*70;j?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();
        c.fillStyle=color;c.fillText(label,x,222);
      }
      c.fillStyle="#c1cdd7";c.fillText("Click curve to inspect a sample",108,222);
    }
    this.drawRanges();
    this.i = Math.min(this.i, this.frames - 1);
    this.set({});
    return this.m;
  }

  url(kind, i) {
    return api.apiURL(`/view?type=output&subfolder=${encodeURIComponent(`${this.sub}/${kind}`)}&filename=${pad(i)}.jpg&v=${this.m.version}`);
  }

  image(kind, i) {
    if (i < 0 || i >= (this.m.counts[kind] ?? 0)) return null;
    const key = `${kind}${i}`;
    let img = this.images.get(key);
    if (!img) {
      img = new Image();
      img.onload = () => { if (Math.abs(i - this.i) < 2 || (kind === "g" && this.m?.ops[this.i]?.gen === i)) this.draw(); };
      img.src = this.url(kind, i);
      this.images.set(key, img);
      if (this.images.size > 600) this.images.delete(this.images.keys().next().value);
    }
    return img.complete && img.naturalWidth ? img : null;
  }

  preload() {
    for (let k = 1; k <= 12; k++) for (const kind of ["p", this.b]) this.image(kind, this.i + k);
  }

  set(state) {
    Object.assign(this, state);
    if (!this.m) return;
    this.modeButtons.forEach((b) => (b.style.background = b.dataset.mode === this.mode ? "#3d5a80" : "#2a2a2a"));
    this.bButton.textContent = this.mode === "three" ? `Original: ${this.threeMatched ? "matched frame" : "delivery order"}` : `B: ${this.b === "l" ? (this.m.analysis_only ? "matched original" : "output") : "as delivered"}`;
    this.draw();
  }

  go(i) {
    if (!this.m) return;
    this.i = Math.max(0, Math.min(this.frames - 1, i));
    this.draw();
  }

  drawRanges() {
    if (!this.issues) return;
    this.ranges.replaceChildren();
    const ranges = issueRanges(this.issues, this.filter);
    for (const r of ranges) {
      const item = el("button", {}, { onclick: () => this.go(r.start) });
      item.dataset.start = r.start;
      item.dataset.end = r.end;
      item.append(el("strong", {}, {textContent: `Frame ${r.start}${r.end !== r.start ? `–${r.end}` : ""}`}),
        el("span", {}, {textContent: r.key.replaceAll("gap", this.m.analysis_only ? "Needs a frame" : "Filled / missing").replaceAll("position", "Position shift").replaceAll("timing", "Possible timing mismatch").replaceAll("uncertain", "Not measurable")}));
      this.ranges.append(item);
    }
    if (!ranges.length) this.ranges.textContent = "No flags in this category. Measurements are estimates, not a pixel-lock guarantee.";
  }

  next(dir) {
    if (!this.issues) return;
    const ranges = issueRanges(this.issues, this.filter);
    const r = dir > 0 ? ranges.find(r => r.start > this.i) : ranges.reverse().find(r => r.start < this.i);
    if (r) this.go(r.start);
  }

  dispose() {
    clearInterval(this.timer);
    this.observer.disconnect();
    this.overlay?.remove();
    this.images.clear();
  }

  play() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    } else {
      this.timer = setInterval(() => (this.i + 1 >= this.frames ? this.play() : this.go(this.i + 1)), 1000 / this.fps);
    }
    this.playButton.textContent = this.timer ? "Pause" : "Play";
  }

  key(e) {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    const k = e.key.toLowerCase();
    const step = e.shiftKey ? 10 : 1;
    const actions = {
      arrowleft: () => this.go(this.i - step), arrowright: () => this.go(this.i + step),
      n: () => this.next(1), p: () => this.next(-1), b: () => this.bButton.onclick(), k: () => this.play(),
      " ": () => this.set({ mode: "flicker", showB: !this.showB }), escape: () => this.close(),
    };
    const n = parseInt(k, 10);
    if (n >= 1 && n <= MODES.length) actions[k] = () => this.set({ mode: MODES[n - 1][0] });
    if (actions[k]) {
      actions[k]();
      e.preventDefault();
      e.stopPropagation();
    }
  }

  open() {
    if (this.overlay) return;
    this.previousFocus = document.activeElement;
    this.overlay = el("div", { position:"fixed", inset:"0", zIndex:100000, background:"#11171c" });
    this.overlay.setAttribute("role", "dialog");
    this.overlay.setAttribute("aria-modal", "true");
    this.overlay.setAttribute("aria-label", "PlateLock review");
    this.overlay.append(this.root);
    document.body.append(this.overlay);
    this.root.focus();
    this.draw();
  }

  close() {
    if (this.timer) this.play();
    this.root.remove();
    this.overlay?.remove();
    this.overlay = null;
    this.previousFocus?.focus();
  }

  draw() {
    if (!this.m || !this.root.isConnected) return;
    const c = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(1, Math.round(c.clientWidth * dpr));
    const H = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== W || c.height !== H) Object.assign(c, { width: W, height: H });
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, W, H);
    const a = this.image("p", this.i);
    const op = this.m.ops[this.i];
    const b = this.m.analysis_only && this.b === "l"
      ? (op.op === "copy" ? this.image("g", op.gen) : null)
      : this.image(this.b, this.i);
    const ref = a || b;
    const labelB = this.b === "l" ? (this.m.analysis_only ? "MATCHED ORIGINAL" : "OUTPUT") : "AS DELIVERED";
    if (ref) {
      const aspect = ref.naturalWidth / ref.naturalHeight;
      let w = W, h = w / aspect;
      if (h > H) { h = H; w = h * aspect; }
      const x0 = (W - w) / 2, y0 = (H - h) / 2;
      const put = (img, x) => (img ? ctx.drawImage(img, x, y0, w, h) : null);
      const label = (text, x, y = y0 + 6) => {
        ctx.font = `${12 * dpr}px system-ui`;
        ctx.fillStyle = "rgba(0,0,0,.6)";
        ctx.fillRect(x, y, ctx.measureText(text).width + 10 * dpr, 18 * dpr);
        ctx.fillStyle = "#fff";
        ctx.fillText(text, x + 5 * dpr, y + 13 * dpr);
      };
      if (this.mode === "three") {
        const sourceIndex = this.threeMatched ? (op.op === "copy" ? op.gen : -1) : this.i;
        const images = [a, this.image("g", sourceIndex), this.m.analysis_only ? null : this.image("l", this.i)];
        const titles = ["PLATE · F" + this.i, (this.threeMatched ? "AI MATCHED · F" : "AI ORIGINAL · F") + (sourceIndex < 0 ? "—" : sourceIndex), this.m.analysis_only ? "OUTPUT · not rendered" : "OUTPUT · F" + this.i];
        const cell = W/3, pw = cell-12*dpr, ph = Math.min(H-70*dpr, pw/aspect), py = (H-ph)/2;
        images.forEach((img,k) => {
          const px = k*cell+6*dpr;
          ctx.fillStyle="#131c24"; ctx.fillRect(px,py,pw,ph);
          if (img) {
            const sw=img.naturalWidth/this.zoom, sh=img.naturalHeight/this.zoom;
            ctx.drawImage(img,this.pan.x*img.naturalWidth-sw/2,this.pan.y*img.naturalHeight-sh/2,sw,sh,px,py,pw,ph);
          }
          label(titles[k],px,py-28*dpr);
          if (!img) {
            ctx.font=`${13*dpr}px system-ui`; ctx.fillStyle="#a4b6c5";
            ctx.fillText(k===2 && this.m.analysis_only ? "Run Repair to compare the result" : "No frame available",px+12*dpr,py+ph/2);
          }
          if (this.regions && k===0 && this.zoom===1) {
            const geom=this.m.drift[this.i]?.geometry;
            ctx.strokeStyle="#e5ad62"; ctx.lineWidth=2*dpr;
            for (const r of geom?.regions || []) ctx.strokeRect(px+r.x*pw,py+r.y*ph,r.w*pw,r.h*ph);
          }
        });
        label(`${this.threeMatched ? "AI uses proposed source match" : "Same frame index; sidebar measures the proposed match"} · ${this.zoom}× zoom · drag to pan${this.regions ? " · boxes: appearance differences, not confirmed defects" : ""}`, 8*dpr,H-25*dpr);
      } else if (this.mode === "wipe") {
        put(a, x0);
        ctx.save();
        ctx.beginPath();
        ctx.rect(x0 + w * this.wipe, y0, w * (1 - this.wipe), h);
        ctx.clip();
        if (b) put(b, x0); else { ctx.fillStyle = "#10171d"; ctx.fillRect(x0, y0, w, h); }
        ctx.restore();
        ctx.fillStyle = "#fff";
        ctx.fillRect(x0 + w * this.wipe - dpr, y0, 2 * dpr, h);
        label("PLATE", x0 + 6); label(labelB, x0 + w - 120 * dpr);
      } else if (this.mode === "overlay") {
        put(a, x0); ctx.globalAlpha = 0.5; put(b, x0); ctx.globalAlpha = 1;
        label("50% PLATE / 50% COMPARISON", x0 + 6);
      } else if (this.mode === "flicker") {
        put(this.showB ? b : a, x0);
        label(this.showB ? labelB : "PLATE", x0 + 6);
      } else {
        this.compare(ctx, a, b, x0, y0, w, h);
        label(`PLATE green · ${labelB} magenta · white = lined up`, x0 + 6);
      }
    }
    if (this.m.analysis_only && this.b === "l" && op.op !== "copy") {
      ctx.fillStyle = "#e8edf1"; ctx.font = `${16 * dpr}px system-ui`; ctx.textAlign = "center";
      ctx.fillText("No original frame matched here · repair needed", W / 2, H - 25 * dpr); ctx.textAlign = "left";
    }
    this.drawTimeline();
    this.drawInfo();
    this.preload();
  }

  compare(ctx, a, b, x0, y0, w, h) {
    if (!a || !b) return;
    const cw = Math.round(Math.min(w, 960)), ch = Math.round(cw * h / w);
    const key = `${this.mode}${this.b}${this.i}${cw}`;
    if (this.cacheKey !== key) {
      const off = el("canvas", {}, { width: cw, height: ch });
      const o = off.getContext("2d", { willReadFrequently: true });
      o.drawImage(a, 0, 0, cw, ch);
      const da = o.getImageData(0, 0, cw, ch).data;
      o.drawImage(b, 0, 0, cw, ch);
      const db = o.getImageData(0, 0, cw, ch).data;
      const out = o.createImageData(cw, ch);
      const ea = edges(da, cw, ch), eb = edges(db, cw, ch);
      for (let i = 0, j = 0; i < ea.length; i++, j += 4) {
        out.data[j] = eb[i] * 255; out.data[j + 1] = ea[i] * 255; out.data[j + 2] = eb[i] * 255; out.data[j + 3] = 255;
      }
      o.putImageData(out, 0, 0);
      this.cacheKey = key;
      this.cacheImage = off;
    }
    ctx.drawImage(this.cacheImage, x0, y0, w, h);
  }

  drawTimeline() {
    const c = this.timeline;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(c.clientWidth * dpr), H = Math.round(c.clientHeight * dpr);
    if (c.width !== W || c.height !== H) Object.assign(c, { width: W, height: H });
    const ctx = c.getContext("2d");
    this.gutter = 86;
    const g = this.gutter * dpr, row = H / 4, n = this.frames, fw = (W - g) / n;
    ctx.fillStyle = "#101010";
    ctx.fillRect(0, 0, W, H);
    ctx.font = `${10 * dpr}px system-ui`;
    ["frame plan", "timing", "position", "unmeasured"].forEach((label, j) => {
      ctx.fillStyle = "#aaa";
      ctx.fillText(label, 4 * dpr, row * j + 12 * dpr);
    });
    this.m.ops.forEach((o, i) => {
      const x = g + i * fw, w = Math.max(fw, dpr), f = this.issues[i];
      const colors = [this.generated.has(i) ? COLORS.generated : COLORS[o.op] || COLORS.copy,
        f.timing ? COLORS.timing : "#242424", f.position ? COLORS.moved : "#242424", f.uncertain ? "#7ba4bc" : "#242424"];
      colors.forEach((color, j) => { ctx.fillStyle = color; ctx.fillRect(x, j * row, w, row - 2 * dpr); });
    });
    ctx.fillStyle = "#fff";
    ctx.fillRect(g + this.i * fw, 0, Math.max(fw, 2 * dpr), H);
  }

  describe(i) {
    const o = this.m.ops[i];
    const d = this.m.drift[i];
    const source = this.generated.has(i) ? "generated by the video model"
      : o.op === "copy" ? `from video frame ${o.gen}`
      : o.op === "interp" ? `${this.m.analysis_only ? "needs fill" : "filled"} between video frames ${o.gen} and ${o.gen_b}`
      : this.m.analysis_only ? "no matching original frame" : "missing (black)";
    const check = this.m.framing ? "Shared-feature residuals are sampled; inspect the shot verdict" : !d ? "unmeasured" : !d.sure ? timingExplanation(d) : d.dt ? `⚠ may show plate frame ${i + d.dt}` : this.moved[i] ? `⚠ moved ${d.shift.toFixed(1)} px` : "timing consistent";
    return [`frame ${i}`, timecode(i, this.fps), source, check].filter(Boolean).join("  ·  ");
  }

  drawInfo() {
    this.scrub.value = this.i;
    this.frameNumber.textContent = `F ${this.i} / ${this.frames - 1}`;
    this.ranges.querySelectorAll("button").forEach(b => b.setAttribute("aria-current", String(this.i >= Number(b.dataset.start) && this.i <= Number(b.dataset.end))));
    this.info.textContent = this.describe(this.i);
    const d = this.m.drift[this.i], op = this.m.ops[this.i];
    this.diagnosis.textContent = `${op.op === "copy" ? `Generated F${op.gen} → plate F${this.i}` : "No matching original frame"}\n${d?.dx != null ? `${Math.abs(d.dx).toFixed(1)} px ${d.dx >= 0 ? "right" : "left"} · ${Math.abs(d.dy).toFixed(1)} px ${d.dy >= 0 ? "down" : "up"}` : "Position unmeasured"}\n${d?.sure ? (d.dt ? `Timing may be ${Math.abs(d.dt)} frame${Math.abs(d.dt) === 1 ? "" : "s"} ${d.dt > 0 ? "ahead" : "behind"}` : "Timing looks consistent") : timingExplanation(d)}`;
    const signed = n => `${n >= 0 ? "+" : ""}${n.toFixed(1)}`;
    const position = d?.dx != null ? `X ${signed(d.dx)} px (${d.dx >= 0 ? "right" : "left"}) · Y ${signed(d.dy)} px (${d.dy >= 0 ? "down" : "up"}) · distance ${d.shift.toFixed(1)} px` : "Position: no reliable measurement";
    const timing = d?.sure ? `Residual timing estimate: ${d.dt > 0 ? "+" : ""}${d.dt} frames (search ±2)` : timingExplanation(d);
    this.details.textContent = `${position}\n${timing}${op.op === "copy" ? ` · source F${op.gen} → plate F${this.i} (${op.gen - this.i >= 0 ? "+" : ""}${op.gen - this.i} frames)` : ""}\n${d?.geometry ? `Rotation ${d.geometry.rotation_deg}° · scale ${d.geometry.scale} · ${d.geometry.inliers} agreeing tracks. Local boxes show appearance differences, not proven deformation.` : "Rotation / scale not measurable. Local deformation not verified."} Steady pans can confuse timing with position.`;
    if (this.b === "g") this.details.textContent += "\nViewing as delivered; measurements above refer to the matched / repaired frame.";
    const before = this.m.before_drift?.[this.i];
    if (before?.shift != null && d?.shift != null) this.diagnosis.textContent += `\nBefore → after: ${before.shift.toFixed(1)} → ${d.shift.toFixed(1)} px`;
    this.diagnosis.textContent += `\n\n${op.op !== "copy" ? "Proposed action: reconstruct this gap; inspect both endpoints." : d?.dt ? "Proposed action: inspect neighbouring frames before remapping." : d?.shift > 4 ? "Proposed action: inspect the position offset; constant shifts can be corrected in Repair." : !d?.sure ? "Action: no automatic timing conclusion from this measurement." : "No timing correction suggested by this check."}`;
    if (this.m.framing) {
      const f = this.m.framing;
      const sample = (f.applied ? f.rendered_series : f.series)?.find(s => s.frame === this.i);
      this.diagnosis.textContent = sample
        ? `F${this.i} · ${f.applied ? "after fixed framing correction" : "predicted after fixed correction"}\nShared detail: ${Math.abs(sample.dx).toFixed(1)} px ${sample.dx >= 0 ? "right" : "left"} · ${Math.abs(sample.dy).toFixed(1)} px ${sample.dy >= 0 ? "down" : "up"}\n\nA single frame cannot establish drift. Use the whole-shot verdict above. Inspect Overlay / Edges for local differences.`
        : `F${this.i} · no shared-feature sample at this frame\n\nUse Overlay / Edges to inspect this frame. The whole-shot verdict uses samples spread across the clip.`;
      this.details.textContent = "Secondary phase-correlation estimate (can be affected by scale and changed backgrounds):\n" + this.details.textContent;
    }
    const x = this.m.framing ? null : this.m.exactness;
    if (x?.offset_px && Math.hypot(...x.offset_px) > MOVED_PX) {
      this.info.textContent += `  ·  whole clip sits ${x.offset_px[0]} px x, ${x.offset_px[1]} px y off the plate`;
    }
  }
}

app.registerExtension({
  name: "platelock.viewer",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["PlateLockStudio", "PlateLock", "PlateLockFillMerge"].includes(nodeData.name)) return;
    function updateControls(node) {
      if (nodeData.name !== "PlateLockStudio") return;
      const value=name=>node.widgets?.find(w=>w.name===name)?.value;
      const advanced=!value("repair_task") || value("repair_task")==="Advanced settings";
      const repairing=value("mode")==="Repair";
      const ltx=value("repair_task")==="Reconstruct missing footage (experimental)" || advanced && value("method")==="LTX";
      for (const widget of node.widgets || []) {
        if (!["method","position","allow_ambiguous_timing","ltx_prompt"].includes(widget.name)) continue;
        const hide=!repairing || (widget.name === "ltx_prompt" ? !ltx : !advanced);
        if (hide && !widget.platelockHidden) {
          widget.platelockHidden={type:widget.type,computeSize:widget.computeSize};
          widget.type="converted-widget";widget.computeSize=()=>[0,-4];
          if (widget.inputEl) widget.inputEl.style.display="none";
        } else if (!hide && widget.platelockHidden) {
          Object.assign(widget,widget.platelockHidden);delete widget.platelockHidden;
          if (widget.inputEl) widget.inputEl.style.display="";
        }
      }
      node.setSize([Math.max(360,node.size[0]),Math.max(280,node.computeSize()[1])]);
      app.graph?.setDirtyCanvas(true,true);
    }
    function attach(node) {
      if (node.platelockCard) return;
      const card = el("div", {}, {className:"pl-launcher"});
      const status = el("div", {}, {textContent:"Run analysis to inspect timing and position."});
      const open = el("button", {}, {textContent:"Open Review ↗", disabled:true, onclick:()=>node.platelockViewer?.open()});
      card.append(status, open);
      node.platelockCard = {card, status, open};
      node.addDOMWidget("review", "platelock_review", card, {serialize:false, hideOnZoom:false, getMinHeight:()=>100, getMaxHeight:()=>110});
      for (const widget of node.widgets || []) {
        if (!["mode","method","repair_task"].includes(widget.name)) continue;
        const callback=widget.callback;
        widget.callback=function(...args) {const result=callback?.apply(this,args);updateControls(node);return result;};
      }
      updateControls(node);
    }
    async function load(node, subfolder) {
      attach(node);
      node.platelockViewer ??= new Viewer();
      if (node.type === "PlateLockStudio") node.platelockViewer.onPrepareRepair = task => {
        for (const [name,value] of [["mode","Repair"],["repair_task",task]]) {
          const widget=node.widgets?.find(w=>w.name===name);
          if (widget) {widget.value=value;widget.callback?.(value);}
        }
        app.graph.setDirtyCanvas(true,true);
      };
      const {status, open} = node.platelockCard;
      status.textContent = "Loading review…";
      open.disabled = true;
      try {
        const m = await node.platelockViewer.load(subfolder);
        const counts = reviewCounts(node.platelockViewer.issues);
        status.textContent = `${m.analysis_only ? "Analysis ready" : "Repair ready for review"} · ${m.ops.length} frames · ${counts.candidates} candidates · ${counts.notMeasurable} not measurable`;
        if (m.framing) status.textContent = shotVerdict(m).title;
        open.disabled = false;
      } catch (e) { status.textContent = e.message; }
    }
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      attach(this);
      this.setSize([360, Math.max(this.computeSize()[1], 300)]);
      return result;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      configured?.apply(this, arguments);
      updateControls(this);
      if (this.properties?.platelockReview) load(this, this.properties.platelockReview);
    };
    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () { this.platelockViewer?.dispose(); return removed?.apply(this, arguments); };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      executed?.apply(this, arguments);
      const info = message?.platelock_viewer?.[0];
      if (!info) return;
      this.properties ??= {};
      this.properties.platelockReview = info.subfolder;
      load(this, info.subfolder);
    };
  },
});
