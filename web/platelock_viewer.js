import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const COLORS = { copy: "#5a5a5a", interp: "#ffa028", generated: "#be6eff", missing: "#e63c3c", timing: "#ffe14a", moved: "#ff5050" };
const MODES = [["wipe", "Wipe", "Drag the line across the frame"],
  ["flicker", "Flicker", "Click or press space to switch between plate and result"],
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
    this.mode = "wipe";
    this.b = "l";
    this.wipe = 0.5;
    this.showB = true;
    this.images = new Map();
    this.root = el("div", { display: "flex", flexDirection: "column", gap: "4px", width: "100%", height: "100%",
      background: "#161616", color: "#ddd", font: "12px system-ui, sans-serif", outline: "none", padding: "4px",
      boxSizing: "border-box" }, { tabIndex: 0 });
    const bar = el("div", { display: "flex", gap: "3px", flexWrap: "wrap", alignItems: "center" });
    const button = (text, onclick, title = "") => {
      const b = el("button", { background: "#2a2a2a", color: "#ddd", border: "1px solid #444", borderRadius: "4px",
        padding: "2px 7px", cursor: "pointer", font: "inherit" }, { textContent: text, onclick, title });
      bar.append(b);
      return b;
    };
    this.modeButtons = MODES.map(([id, label, title], n) => {
      const b = button(label, () => this.set({ mode: id }), `${title} (${n + 1})`);
      b.dataset.mode = id;
      return b;
    });
    this.bButton = button("", () => this.set({ b: this.b === "l" ? "g" : "l" }), "Compare the plate with (B)");
    this.playButton = button("▶", () => this.play(), "Play (K)");
    button("⤢", () => this.fullscreen(), "Full screen");
    this.canvas = el("canvas", { width: "100%", flex: "1 1 auto", minHeight: "120px", cursor: "ew-resize",
      background: "#000", borderRadius: "3px" });
    this.timeline = el("canvas", { width: "100%", height: "34px", cursor: "pointer" });
    this.legend = el("div", { display: "flex", gap: "10px", flexWrap: "wrap", fontSize: "11px", color: "#aaa" });
    for (const [key, text] of LEGEND) {
      const item = el("span", {}, { textContent: text });
      item.prepend(el("span", { display: "inline-block", width: "9px", height: "9px", marginRight: "4px",
        background: COLORS[key], borderRadius: "2px" }));
      this.legend.append(item);
    }
    this.info = el("div", { whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", fontVariantNumeric: "tabular-nums" });
    this.help = el("div", { color: "#777", fontSize: "11px" }, { textContent: HELP });
    this.root.append(bar, this.canvas, this.info, this.timeline, this.legend, this.help);
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
      if (this.mode === "wipe") { this.canvas.setPointerCapture(e.pointerId); wipe(e); }
      if (this.mode === "flicker") this.set({ showB: !this.showB });
    });
    this.canvas.addEventListener("pointermove", (e) => { if (e.buttons && this.mode === "wipe") wipe(e); });
    new ResizeObserver(() => this.draw()).observe(this.root);
  }

  async load(subfolder) {
    this.sub = subfolder;
    const url = api.apiURL(`/view?type=output&subfolder=${encodeURIComponent(subfolder)}&filename=manifest.json&t=${Date.now()}`);
    this.m = await (await fetch(url)).json();
    this.images.clear();
    this.frames = this.m.ops.length;
    this.fps = this.m.fps.split("/").reduce((a, b) => a / b);
    this.generated = new Set(this.m.generated);
    const [ox, oy] = this.m.exactness?.offset_px ?? [0, 0];
    this.moved = this.m.drift.map((d) => d?.dx != null && Math.hypot(d.dx - ox, d.dy - oy) > MOVED_PX);
    this.problems = this.m.ops.map((o, i) => o.op !== "copy" || this.generated.has(i) || !!this.m.drift[i]?.dt || this.moved[i]);
    this.i = Math.min(this.i, this.frames - 1);
    this.set({});
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
      img.onload = () => { if (Math.abs(i - this.i) < 2) this.draw(); };
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
    this.bButton.textContent = `B: ${this.b === "l" ? this.m.label.toLowerCase() : "as delivered"}`;
    this.draw();
  }

  go(i) {
    if (!this.m) return;
    this.i = Math.max(0, Math.min(this.frames - 1, i));
    this.draw();
  }

  next(dir) {
    for (let i = this.i + dir; i >= 0 && i < this.frames; i += dir) {
      if (this.problems[i] && !this.problems[i - dir]) return this.go(i);
    }
  }

  play() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    } else {
      this.timer = setInterval(() => (this.i + 1 >= this.frames ? this.play() : this.go(this.i + 1)), 1000 / this.fps);
    }
    this.playButton.textContent = this.timer ? "❚❚" : "▶";
  }

  key(e) {
    const k = e.key.toLowerCase();
    const step = e.shiftKey ? 10 : 1;
    const actions = {
      arrowleft: () => this.go(this.i - step), arrowright: () => this.go(this.i + step),
      n: () => this.next(1), p: () => this.next(-1), b: () => this.bButton.onclick(), k: () => this.play(),
      " ": () => this.set({ mode: "flicker", showB: !this.showB }), escape: () => this.overlay && this.fullscreen(),
    };
    const n = parseInt(k, 10);
    if (n >= 1 && n <= MODES.length) actions[k] = () => this.set({ mode: MODES[n - 1][0] });
    if (actions[k]) {
      actions[k]();
      e.preventDefault();
      e.stopPropagation();
    }
  }

  fullscreen() {
    if (this.overlay) {
      this.home.append(this.root);
      this.overlay.remove();
      this.overlay = null;
    } else {
      this.home = this.root.parentElement;
      this.overlay = el("div", { position: "fixed", inset: "0", zIndex: 10000, background: "#111", padding: "16px" });
      this.overlay.append(this.root);
      document.body.append(this.overlay);
    }
    this.root.focus();
    this.draw();
  }

  draw() {
    if (!this.m) return;
    const c = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(1, Math.round(c.clientWidth * dpr));
    const H = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== W || c.height !== H) Object.assign(c, { width: W, height: H });
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, W, H);
    const a = this.image("p", this.i);
    const b = this.image(this.b, this.i);
    const ref = a || b;
    const labelB = this.b === "l" ? this.m.label : "AS DELIVERED";
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
      if (this.mode === "wipe") {
        put(a, x0);
        ctx.save();
        ctx.beginPath();
        ctx.rect(x0 + w * this.wipe, y0, w * (1 - this.wipe), h);
        ctx.clip();
        put(b, x0);
        ctx.restore();
        ctx.fillStyle = "#fff";
        ctx.fillRect(x0 + w * this.wipe - dpr, y0, 2 * dpr, h);
        label("PLATE", x0 + 6); label(labelB, x0 + w - 120 * dpr);
      } else if (this.mode === "flicker") {
        put(this.showB ? b : a, x0);
        label(this.showB ? labelB : "PLATE", x0 + 6);
      } else {
        this.compare(ctx, a, b, x0, y0, w, h);
        label(`PLATE green · ${labelB} magenta · white = lined up`, x0 + 6);
      }
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
    this.gutter = 52;
    const g = this.gutter * dpr, row = (H - 4 * dpr) / 2, n = this.frames, fw = (W - g) / n;
    ctx.fillStyle = "#101010";
    ctx.fillRect(0, 0, W, H);
    ctx.font = `${10 * dpr}px system-ui`;
    ctx.fillStyle = "#999";
    ctx.fillText("frames", 4 * dpr, row * 0.5 + 4 * dpr);
    ctx.fillText("check", 4 * dpr, row * 1.5 + 8 * dpr);
    ctx.fillStyle = "#1c1c1c";
    ctx.fillRect(g, row + 3 * dpr, W - g, row);
    this.m.ops.forEach((o, i) => {
      const x = g + i * fw, w = Math.max(fw, dpr);
      ctx.fillStyle = this.generated.has(i) ? COLORS.generated : COLORS[o.op] || COLORS.copy;
      ctx.fillRect(x, 0, w, row);
      const flag = this.m.drift[i]?.dt ? COLORS.timing : this.moved[i] ? COLORS.moved : null;
      if (flag) {
        ctx.fillStyle = flag;
        ctx.fillRect(x, row + 3 * dpr, w, row);
      }
    });
    ctx.fillStyle = "#fff";
    ctx.fillRect(g + this.i * fw, 0, Math.max(fw, 2 * dpr), H);
  }

  describe(i) {
    const o = this.m.ops[i];
    const d = this.m.drift[i];
    const source = this.generated.has(i) ? "generated by the video model"
      : o.op === "copy" ? `from video frame ${o.gen}`
      : o.op === "interp" ? `filled between video frames ${o.gen} and ${o.gen_b}`
      : "missing (black)";
    const check = !d ? "" : d.dt ? `⚠ shows plate frame ${i + d.dt}` : this.moved[i] ? `⚠ moved ${d.shift.toFixed(1)} px` : "✓";
    return [`frame ${i}`, timecode(i, this.fps), source, check].filter(Boolean).join("  ·  ");
  }

  drawInfo() {
    this.info.textContent = this.describe(this.i);
    const x = this.m.exactness;
    if (x?.offset_px && Math.hypot(...x.offset_px) > MOVED_PX) {
      this.info.textContent += `  ·  whole clip sits ${x.offset_px[0]} px x, ${x.offset_px[1]} px y off the plate`;
    }
  }
}

app.registerExtension({
  name: "platelock.viewer",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["PlateLock", "PlateLockFillMerge"].includes(nodeData.name)) return;
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const info = message?.platelock_viewer?.[0];
      if (!info) return;
      const fillUsed = nodeData.name === "PlateLock" && this.outputs?.find((o) => o.name === "fill")?.links?.length;
      if (fillUsed) {
        if (this.platelockViewer) this.platelockViewer.root.style.display = "none";
        return;
      }
      if (this.platelockViewer) this.platelockViewer.root.style.display = "";
      if (!this.platelockViewer) {
        this.platelockViewer = new Viewer();
        this.addDOMWidget("viewer", "platelock_viewer", this.platelockViewer.root, {
          serialize: false, hideOnZoom: false, getMinHeight: () => 340,
        });
        const [w, h] = this.size;
        this.setSize([Math.max(w, 640), Math.max(h, 560)]);
      }
      this.platelockViewer.load(info.subfolder);
    };
  },
});
