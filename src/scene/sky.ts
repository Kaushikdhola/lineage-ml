import * as THREE from "three";
import type { SkyLayout, StarLayout } from "../atlas/layout";
import { FAMILY_LABEL, type Edge } from "../atlas/schema";

/**
 * The sky: every formula is a star on the lineage disc. Five draw calls in total:
 * instanced stars, all filaments, rings + sector ticks, dust, and the lettering plane.
 * The scene computes nothing about the math; positions come from src/atlas/layout.
 */

const ACCENT = new THREE.Color(0x5ee3d6);
const PALE = new THREE.Color(0xdde6e8);

/** Values of the per-star aState attribute, read by the star shader. */
const StarState = { Normal: 0, Hover: 1, Selected: 2, Path: 3 } as const;

interface CameraState {
  target: THREE.Vector3;
  dist: number;
  az: number;
  el: number;
  /** Horizontal framing shift as a fraction of the viewport width (positive moves content left). */
  shift: number;
  /** Vertical framing shift as a fraction of the viewport height (positive moves content up). */
  shiftY?: number;
}

export interface SkyCallbacks {
  hover(id: string | null): void;
  select(id: string): void;
  background(): void;
  frame(): void;
}

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const wrapPi = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

export class Sky {
  readonly renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(40, 1, 0.1, 400);
  private stars: THREE.InstancedMesh;
  private starState: THREE.InstancedBufferAttribute;
  private filaments: THREE.LineSegments;
  private edgeHi: THREE.BufferAttribute;
  private edgeVertexRanges: [number, number][] = [];
  private lettering: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
  private letteringCanvas = document.createElement("canvas");
  private letteringFlags = "";
  private letteringSize: number;
  private uniforms = { uTime: { value: 0 }, uDim: { value: 0 } };
  private index = new Map<string, number>();
  private cam: CameraState;
  private flight: {
    from: CameraState;
    to: CameraState;
    t0: number;
    dur: number;
    done: () => void;
  } | null = null;
  private drift = true;
  private lastInput = performance.now();
  private running = false;
  private size = { w: 1, h: 1 };
  private ro: ResizeObserver;
  private listeners: [EventTarget, string, EventListener, AddEventListenerOptions?][] = [];
  private hoverId: string | null = null;
  private selectedId: string | null = null;
  private pathIds: string[] = [];
  private tmp = new THREE.Vector3();
  private prevT = performance.now();
  private cpuSamples: number[] = [];
  readonly stats = { cpuMs: 0, calls: 0 };

  constructor(
    private readonly el: HTMLElement,
    private readonly layout: SkyLayout,
    private readonly edges: Edge[],
    private readonly cb: SkyCallbacks,
    private readonly reducedMotion: boolean,
  ) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x05070a, 1);
    el.append(this.renderer.domElement);
    layout.stars.forEach((s, i) => this.index.set(s.id, i));

    this.stars = this.buildStars();
    this.filaments = this.buildFilaments();
    this.edgeHi = this.filaments.geometry.getAttribute("aHi") as THREE.BufferAttribute;
    this.starState = this.stars.geometry.getAttribute("aState") as THREE.InstancedBufferAttribute;
    this.letteringSize = layout.outerRadius + 7;
    this.lettering = this.buildLettering();
    this.scene.add(this.buildRings(), this.buildDust(), this.lettering, this.filaments, this.stars);

    this.cam = this.overviewState(0);
    this.applyCamera();

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(el);
    this.resize();
    this.bindInput();
    void document.fonts?.ready.then(() => this.drawLettering(true));
  }

  // ---------- build ----------

  private buildStars(): THREE.InstancedMesh {
    const { stars } = this.layout;
    const degree = new Map<string, number>();
    for (const e of this.edges) {
      degree.set(e.from, (degree.get(e.from) ?? 0) + 1);
      degree.set(e.to, (degree.get(e.to) ?? 0) + 1);
    }
    const geo = new THREE.PlaneGeometry(1, 1);
    const n = stars.length;
    const size = new Float32Array(n);
    const bright = new Float32Array(n);
    const phase = new Float32Array(n);
    const hub = new Float32Array(n);
    stars.forEach((s, i) => {
      const d = degree.get(s.id) ?? 0;
      size[i] = s.hub ? 5.2 : 1.9 + 0.22 * Math.min(d, 5);
      bright[i] = s.hub ? 1.15 : 0.7 + 0.08 * Math.min(d, 5);
      phase[i] = (i * 2.399) % (2 * Math.PI);
      hub[i] = s.hub ? 1 : 0;
    });
    geo.setAttribute("aSize", new THREE.InstancedBufferAttribute(size, 1));
    geo.setAttribute("aBright", new THREE.InstancedBufferAttribute(bright, 1));
    geo.setAttribute("aPhase", new THREE.InstancedBufferAttribute(phase, 1));
    geo.setAttribute("aHub", new THREE.InstancedBufferAttribute(hub, 1));
    const state = new THREE.InstancedBufferAttribute(new Float32Array(n), 1);
    state.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("aState", state);

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        ...this.uniforms,
        uPale: { value: PALE },
        uAccent: { value: ACCENT },
        uTwinkle: { value: this.reducedMotion ? 0 : 1 },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: /* glsl */ `
        attribute float aSize; attribute float aBright; attribute float aPhase;
        attribute float aHub; attribute float aState;
        uniform float uTime; uniform float uTwinkle;
        varying vec2 vUv; varying float vBright; varying float vState; varying float vHub;
        void main() {
          vUv = uv; vState = aState; vHub = aHub;
          float tw = 1.0 + uTwinkle * 0.06 * sin(uTime * 1.3 + aPhase * 3.0);
          vBright = aBright * tw;
          float grow = aState == 1.0 ? 1.35 : (aState == 2.0 ? 1.5 : (aState == 3.0 ? 1.25 : 1.0));
          vec4 mv = modelViewMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
          mv.xy += position.xy * aSize * grow;
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uPale; uniform vec3 uAccent; uniform float uDim;
        varying vec2 vUv; varying float vBright; varying float vState; varying float vHub;
        void main() {
          float d = length(vUv * 2.0 - 1.0);
          if (d > 1.0) discard;
          float core = smoothstep(0.13, 0.04, d);
          float halo = exp(-d * d * 14.0) * 0.75 + exp(-d * d * 3.5) * 0.12;
          float lit = vState > 0.5 ? 1.0 : 0.0;
          float ring = vState == 2.0 ? smoothstep(0.025, 0.0, abs(d - 0.36)) * 0.8 : 0.0;
          vec3 col = mix(uPale, uAccent, max(lit, vHub * 0.85));
          float dim = mix(1.0, 0.42, uDim * (1.0 - lit) * (1.0 - vHub * 0.5));
          float a = (core + halo * vBright * (1.0 + 0.6 * lit)) * dim + ring;
          gl_FragColor = vec4(col * a, a);
        }`,
    });
    const mesh = new THREE.InstancedMesh(geo, mat, n);
    const m = new THREE.Matrix4();
    stars.forEach((s, i) => mesh.setMatrixAt(i, m.makeTranslation(s.x, s.y, s.z)));
    mesh.frustumCulled = false;
    mesh.renderOrder = 3;
    return mesh;
  }

  private buildFilaments(): THREE.LineSegments {
    const SEG = 28;
    const pos: number[] = [];
    const t: number[] = [];
    const kind: number[] = [];
    const hub: number[] = [];
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const c = new THREE.Vector3();
    const p = new THREE.Vector3();
    const q = new THREE.Vector3();
    const curve = new THREE.QuadraticBezierCurve3(a, c, b);
    this.edges.forEach((e) => {
      const s0 = this.layout.stars[this.index.get(e.from)!];
      const s1 = this.layout.stars[this.index.get(e.to)!];
      a.set(s0.x, s0.y, s0.z);
      b.set(s1.x, s1.y, s1.z);
      const len = a.distanceTo(b);
      // a gentle arch: lift the midpoint so filaments read as 3D, not drawn on the floor
      c.addVectors(a, b).multiplyScalar(0.5);
      c.y += 0.08 * len;
      const start = pos.length / 3;
      for (let i = 0; i < SEG; i++) {
        curve.getPoint(i / SEG, p);
        curve.getPoint((i + 1) / SEG, q);
        pos.push(p.x, p.y, p.z, q.x, q.y, q.z);
        t.push(i / SEG, (i + 1) / SEG);
        kind.push(e.kind === "historical" ? 1 : 0, e.kind === "historical" ? 1 : 0);
        hub.push(s0.hub ? 1 : 0, s0.hub ? 1 : 0);
      }
      this.edgeVertexRanges.push([start, pos.length / 3]);
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute("aT", new THREE.Float32BufferAttribute(t, 1));
    geo.setAttribute("aKind", new THREE.Float32BufferAttribute(kind, 1));
    geo.setAttribute("aHub", new THREE.Float32BufferAttribute(hub, 1));
    const hi = new THREE.Float32BufferAttribute(new Float32Array(pos.length / 3), 1);
    hi.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("aHi", hi);
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        ...this.uniforms,
        uPale: { value: PALE },
        uAccent: { value: ACCENT },
        uFlow: { value: this.reducedMotion ? 0 : 1 },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: /* glsl */ `
        attribute float aT; attribute float aKind; attribute float aHub; attribute float aHi;
        varying float vT; varying float vKind; varying float vHub; varying float vHi;
        void main() {
          vT = aT; vKind = aKind; vHub = aHub; vHi = aHi;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uPale; uniform vec3 uAccent; uniform float uTime; uniform float uDim; uniform float uFlow;
        varying float vT; varying float vKind; varying float vHub; varying float vHi;
        void main() {
          float ends = smoothstep(0.0, 0.06, vT) * smoothstep(1.0, 0.94, vT);
          float base = mix(0.16, 0.3, vKind) + vHub * 0.06;
          // parent end slightly brighter: filaments read as flowing outward
          base *= mix(1.15, 0.7, vT);
          float pulse = uFlow * smoothstep(0.12, 0.0, abs(fract(vT - uTime * 0.35) - 0.5) - 0.38);
          float hi = vHi * (0.55 + 0.45 * pulse);
          float a = ends * (base * mix(1.0, 0.45, uDim) * (1.0 - vHi) + hi);
          vec3 col = mix(uPale, uAccent, max(vHi, vHub * 0.6));
          gl_FragColor = vec4(col * a, a);
        }`,
    });
    const lines = new THREE.LineSegments(geo, mat);
    lines.frustumCulled = false;
    lines.renderOrder = 2;
    return lines;
  }

  private buildRings(): THREE.LineSegments {
    const pos: number[] = [];
    const N = 256;
    for (const ring of this.layout.rings) {
      for (let i = 0; i < N; i++) {
        const a0 = (i / N) * Math.PI * 2;
        const a1 = ((i + 1) / N) * Math.PI * 2;
        pos.push(ring.radius * Math.cos(a0), 0, ring.radius * Math.sin(a0));
        pos.push(ring.radius * Math.cos(a1), 0, ring.radius * Math.sin(a1));
      }
    }
    // sector boundaries: short ticks just outside the outermost ring
    const r0 = this.layout.outerRadius + 0.6;
    const r1 = this.layout.outerRadius + 1.6;
    for (const s of this.layout.sectors)
      for (const a of [s.start, s.end])
        pos.push(r0 * Math.cos(a), 0, r0 * Math.sin(a), r1 * Math.cos(a), 0, r1 * Math.sin(a));
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    const mat = new THREE.LineBasicMaterial({
      color: 0x8fa3ad,
      transparent: true,
      opacity: 0.09,
      depthWrite: false,
    });
    const rings = new THREE.LineSegments(geo, mat);
    rings.renderOrder = 0;
    rings.matrixAutoUpdate = false;
    return rings;
  }

  private buildDust(): THREE.Points {
    // deterministic dust (golden-angle spiral + hash), for depth cues only
    const N = 2200;
    const R = this.layout.outerRadius * 1.5 + 8;
    const pos = new Float32Array(N * 3);
    const size = new Float32Array(N);
    const hash = (i: number) => {
      const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
      return x - Math.floor(x);
    };
    for (let i = 0; i < N; i++) {
      const r = R * Math.sqrt(hash(i));
      const a = i * 2.39996 + hash(i + 9) * 0.6;
      const spread = 0.6 + 2.2 * (r / R);
      pos[i * 3] = r * Math.cos(a);
      pos[i * 3 + 1] = (hash(i + 3) - 0.5) * 2 * spread + (hash(i + 7) < 0.08 ? (hash(i + 5) - 0.5) * 30 : 0);
      pos[i * 3 + 2] = r * Math.sin(a);
      size[i] = 0.5 + 1.4 * hash(i + 11) ** 3;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("aSize", new THREE.BufferAttribute(size, 1));
    const mat = new THREE.ShaderMaterial({
      uniforms: { uScale: { value: 1 }, uDim: this.uniforms.uDim },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: /* glsl */ `
        attribute float aSize; uniform float uScale; varying float vA;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = aSize * uScale / -mv.z;
          vA = clamp(aSize * 0.22, 0.05, 0.35);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */ `
        uniform float uDim; varying float vA;
        void main() {
          float d = length(gl_PointCoord * 2.0 - 1.0);
          float a = smoothstep(1.0, 0.0, d) * vA * mix(1.0, 0.6, uDim);
          gl_FragColor = vec4(vec3(0.62, 0.72, 0.78) * a, a);
        }`,
    });
    const pts = new THREE.Points(geo, mat);
    pts.frustumCulled = false;
    pts.renderOrder = 1;
    pts.matrixAutoUpdate = false;
    return pts;
  }

  private buildLettering(): THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial> {
    const L = this.letteringSize;
    this.letteringCanvas.width = this.letteringCanvas.height = 2048;
    const tex = new THREE.CanvasTexture(this.letteringCanvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    const geo = new THREE.PlaneGeometry(2 * L, 2 * L);
    geo.rotateX(-Math.PI / 2);
    const mat = new THREE.MeshBasicMaterial({
      map: tex,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = -0.01;
    mesh.renderOrder = 0;
    return mesh;
  }

  /**
   * Letters family names along arcs at each sector's outer edge, plus generation numerals. Glyph
   * orientation depends on which side of the disc the camera is on, so labels on the near side
   * are not upside down; it is redrawn only when that changes.
   */
  private drawLettering(force = false): void {
    const camDir = Math.atan2(this.camera.position.z, this.camera.position.x);
    const near = (a: number) => Math.cos(a - camDir) > 0;
    const flags =
      this.layout.sectors.map((s) => (near((s.start + s.end) / 2) ? 1 : 0)).join("") +
      (near(-Math.PI / 2) ? 1 : 0);
    if (!force && flags === this.letteringFlags) return;
    this.letteringFlags = flags;

    const cv = this.letteringCanvas;
    const ctx = cv.getContext("2d")!;
    const S = cv.width;
    const L = this.letteringSize;
    const px = S / (2 * L);
    ctx.clearRect(0, 0, S, S);
    ctx.save();
    ctx.translate(S / 2, S / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    const arcText = (
      text: string,
      radius: number,
      mid: number,
      font: string,
      color: string,
      tracking: number,
    ) => {
      ctx.font = font;
      ctx.fillStyle = color;
      const r = radius * px;
      const chars = [...text];
      const widths = chars.map((ch) => ctx.measureText(ch).width + tracking);
      const total = widths.reduce((x, y) => x + y, 0) - tracking;
      const isNear = near(mid);
      // near side: tops point inward and the text runs counter-clockwise, so it reads upright
      const dir = isNear ? -1 : 1;
      let a = mid - (dir * total) / 2 / r;
      chars.forEach((ch, i) => {
        const w = widths[i];
        const ca = a + (dir * (w - tracking)) / 2 / r;
        ctx.save();
        ctx.translate(r * Math.cos(ca), r * Math.sin(ca));
        ctx.rotate(ca + (isNear ? -Math.PI / 2 : Math.PI / 2));
        ctx.fillText(ch, 0, 0);
        ctx.restore();
        a += (dir * w) / r;
      });
    };

    for (const s of this.layout.sectors) {
      const mid = (s.start + s.end) / 2;
      arcText(
        FAMILY_LABEL[s.family].toUpperCase(),
        s.edgeRadius,
        mid,
        `600 ${Math.round(1.25 * px)}px "Barlow Condensed", "Arial Narrow", sans-serif`,
        "rgba(176, 194, 202, 0.62)",
        0.55 * px,
      );
    }
    // generation numerals along the gap ray at -π/2 (between the last and first sector)
    for (const ring of this.layout.rings) {
      arcText(
        String(ring.generation),
        ring.radius,
        -Math.PI / 2,
        `500 ${Math.round(0.62 * px)}px "JetBrains Mono", ui-monospace, monospace`,
        "rgba(143, 163, 173, 0.55)",
        0,
      );
    }
    ctx.restore();
    this.lettering.material.map!.needsUpdate = true;
  }

  // ---------- camera ----------

  private fitCam = new THREE.PerspectiveCamera(40, 1, 0.1, 800);
  private rimPts = Array.from({ length: 48 }, (_, i) => (i / 48) * Math.PI * 2);

  /**
   * The overview frames the whole disc, lettering included, between the header and the legend.
   * The rim is a circle, so the fit holds at every drift angle.
   */
  private overviewState(az = this.cam?.az ?? Math.PI / 2): CameraState {
    // portrait screens have height to spare: look down more steeply to use it
    const aspect = this.size.w / this.size.h || 1.6;
    const el = aspect < 1 ? 0.95 : aspect < 1.3 ? 0.78 : 0.62;
    const R = this.layout.outerRadius + 2.8;
    const cam = this.fitCam;
    cam.fov = this.camera.fov;
    cam.aspect = this.size.w / this.size.h || 1.6;
    cam.updateProjectionMatrix();
    const v = this.tmp;
    const measure = (d: number) => {
      cam.position.set(d * Math.cos(el) * Math.cos(az), d * Math.sin(el), d * Math.cos(el) * Math.sin(az));
      cam.lookAt(0, 0, 0);
      cam.updateMatrixWorld();
      let xMax = 0;
      let yMin = Infinity;
      let yMax = -Infinity;
      for (const a of this.rimPts) {
        v.set(R * Math.cos(a), 0, R * Math.sin(a)).project(cam);
        xMax = Math.max(xMax, Math.abs(v.x));
        yMin = Math.min(yMin, v.y);
        yMax = Math.max(yMax, v.y);
      }
      return { xMax, yMin, yMax };
    };
    // vertical band left free by the header (top) and the legend (bottom), in NDC
    const narrow = this.size.w < 900;
    const band = narrow ? 1.25 : 1.46;
    let lo = 5;
    let hi = 600;
    for (let i = 0; i < 28; i++) {
      const mid = (lo + hi) / 2;
      const m = measure(mid);
      if (m.xMax <= 0.94 && m.yMax - m.yMin <= band) hi = mid;
      else lo = mid;
    }
    const m = measure(hi);
    const centre = (m.yMin + m.yMax) / 2;
    const want = narrow ? -0.04 : 0.0;
    return { target: new THREE.Vector3(0, 0, 0), dist: hi, az, el, shift: 0, shiftY: (want - centre) / 2 };
  }

  private applyCamera(): void {
    const { target, dist, az, el, shift } = this.cam;
    const shiftY = this.cam.shiftY ?? 0;
    this.camera.position.set(
      target.x + dist * Math.cos(el) * Math.cos(az),
      target.y + dist * Math.sin(el),
      target.z + dist * Math.cos(el) * Math.sin(az),
    );
    this.camera.lookAt(target);
    const { w, h } = this.size;
    if (shift || shiftY) this.camera.setViewOffset(w, h, shift * w, shiftY * h, w, h);
    else if (this.camera.view?.enabled) this.camera.clearViewOffset();
  }

  private fly(to: CameraState, dur: number): Promise<void> {
    this.flight?.done();
    if (this.reducedMotion || dur <= 0) {
      this.cam = to;
      this.applyCamera();
      return Promise.resolve();
    }
    return new Promise((done) => {
      const from = { ...this.cam, target: this.cam.target.clone() };
      // take the short way round
      to.az = from.az + wrapPi(to.az - from.az);
      this.flight = { from, to, t0: performance.now(), dur, done };
    });
  }

  /** The page-load moment: settle from high and far out into the overview. */
  intro(): Promise<void> {
    const o = this.overviewState();
    this.cam = { ...o, target: o.target.clone(), dist: o.dist * 1.65, el: 1.1, az: o.az - 0.55 };
    this.applyCamera();
    return this.fly(o, 3200);
  }

  overview(dur = 1400): Promise<void> {
    this.drift = true;
    return this.fly(this.overviewState(), dur);
  }

  /** Flies to a star. `shift` frames it left of centre to leave room for the card. */
  focus(id: string, shift: number, dur = 1500): Promise<void> {
    const s = this.layout.stars[this.index.get(id)!];
    this.drift = false;
    // look at the star from its outer side, so its parents sit behind it toward the hub
    const outward = s.hub ? this.cam.az : Math.atan2(s.z, s.x);
    const az = this.cam.az + wrapPi(outward - this.cam.az) * 0.55;
    return this.fly(
      { target: new THREE.Vector3(s.x, s.y, s.z), dist: s.hub ? 16 : 11, az, el: 0.42, shift },
      dur,
    );
  }

  /** Places the camera at a star immediately (for arriving from the lab). */
  jumpTo(id: string, shift: number): void {
    void this.focus(id, shift, 0);
  }

  // ---------- state ----------

  setHover(id: string | null): void {
    this.hoverId = id;
    this.refreshStates();
  }

  setSelected(id: string | null): void {
    this.selectedId = id;
    this.uniforms.uDim.value = id ? 1 : 0;
    this.refreshStates();
  }

  /** Lights a chain of stars and the filaments between them (tour). */
  setPath(ids: string[]): void {
    this.pathIds = ids;
    this.refreshStates();
  }

  private refreshStates(): void {
    const arr = this.starState.array as Float32Array;
    arr.fill(StarState.Normal);
    for (const id of this.pathIds) arr[this.index.get(id)!] = StarState.Path;
    if (this.selectedId) arr[this.index.get(this.selectedId)!] = StarState.Selected;
    if (this.hoverId && this.hoverId !== this.selectedId)
      arr[this.index.get(this.hoverId)!] = StarState.Hover;
    this.starState.needsUpdate = true;

    const hi = this.edgeHi.array as Float32Array;
    hi.fill(0);
    const pathSet = new Set<string>();
    for (let i = 1; i < this.pathIds.length; i++) pathSet.add(`${this.pathIds[i - 1]}>${this.pathIds[i]}`);
    const focusId = this.selectedId ?? this.hoverId;
    this.edges.forEach((e, i) => {
      const lit =
        pathSet.has(`${e.from}>${e.to}`) ||
        (this.pathIds.length === 0 && focusId !== null && (e.from === focusId || e.to === focusId));
      if (!lit) return;
      const [a, b] = this.edgeVertexRanges[i];
      hi.fill(1, a, b);
    });
    this.edgeHi.needsUpdate = true;
  }

  // ---------- projection & picking ----------

  /** Screen position (CSS px, relative to the canvas) of a star. */
  project(id: string): { x: number; y: number; visible: boolean } {
    const s = this.layout.stars[this.index.get(id)!];
    this.tmp.set(s.x, s.y, s.z).project(this.camera);
    return {
      x: ((this.tmp.x + 1) / 2) * this.size.w,
      y: ((1 - this.tmp.y) / 2) * this.size.h,
      visible: this.tmp.z < 1 && Math.abs(this.tmp.x) < 1.1 && Math.abs(this.tmp.y) < 1.1,
    };
  }

  /** Nearest star to a canvas-relative point, within a radius that scales with apparent size. */
  pick(x: number, y: number): StarLayout | null {
    let best: StarLayout | null = null;
    let bestD = Infinity;
    for (const s of this.layout.stars) {
      const p = this.project(s.id);
      if (!p.visible) continue;
      const d = Math.hypot(p.x - x, p.y - y);
      const reach = s.hub ? 34 : 22;
      if (d < reach && d < bestD) {
        best = s;
        bestD = d;
      }
    }
    return best;
  }

  // ---------- input ----------

  private on(t: EventTarget, type: string, fn: EventListener, opts?: AddEventListenerOptions) {
    t.addEventListener(type, fn, opts);
    this.listeners.push([t, type, fn, opts]);
  }

  private bindInput(): void {
    const cv = this.renderer.domElement;
    cv.style.touchAction = "none";
    let down: { x: number; y: number; az: number; el: number; moved: boolean; id: number } | null = null;
    const local = (e: PointerEvent) => {
      const r = cv.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    this.on(cv, "pointerdown", ((e: PointerEvent) => {
      const p = local(e);
      down = { ...p, az: this.cam.az, el: this.cam.el, moved: false, id: e.pointerId };
      this.lastInput = performance.now();
    }) as EventListener);
    this.on(window, "pointermove", ((e: PointerEvent) => {
      const p = local(e);
      if (down && down.id === e.pointerId) {
        const dx = p.x - down.x;
        const dy = p.y - down.y;
        if (!down.moved && Math.hypot(dx, dy) > 5) {
          down.moved = true;
          cv.setPointerCapture?.(e.pointerId);
          this.flight?.done();
          this.flight = null;
        }
        if (down.moved) {
          this.cam.az = down.az + dx * 0.005;
          this.cam.el = THREE.MathUtils.clamp(down.el + dy * 0.004, 0.12, 1.35);
          this.applyCamera();
          this.lastInput = performance.now();
        }
        return;
      }
      if (e.target !== cv) return;
      const hit = this.pick(p.x, p.y);
      const id = hit?.id ?? null;
      cv.style.cursor = id ? "pointer" : "grab";
      if (id !== this.hoverId) this.cb.hover(id);
    }) as EventListener);
    this.on(window, "pointerup", ((e: PointerEvent) => {
      if (!down || down.id !== e.pointerId) return;
      const wasDrag = down.moved;
      down = null;
      if (wasDrag) return;
      const p = local(e);
      const hit = this.pick(p.x, p.y);
      if (hit) this.cb.select(hit.id);
      else this.cb.background();
    }) as EventListener);
    this.on(cv, "pointerleave", (() => {
      if (!down) this.cb.hover(null);
    }) as EventListener);
    this.on(
      cv,
      "wheel",
      ((e: WheelEvent) => {
        e.preventDefault();
        this.flight?.done();
        this.flight = null;
        const ov = this.overviewState().dist;
        this.cam.dist = THREE.MathUtils.clamp(this.cam.dist * Math.exp(e.deltaY * 0.0012), 6, ov * 1.6);
        this.applyCamera();
        this.lastInput = performance.now();
      }) as EventListener,
      { passive: false },
    );
  }

  // ---------- loop ----------

  private resize(): void {
    const w = this.el.clientWidth || 1;
    const h = this.el.clientHeight || 1;
    this.size = { w, h };
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const dust = (this.scene.children.find((c) => c instanceof THREE.Points) as THREE.Points | undefined)
      ?.material as THREE.ShaderMaterial | undefined;
    if (dust) dust.uniforms.uScale.value = (h * this.renderer.getPixelRatio()) / 18;
    if (this.drift && !this.flight) this.cam = this.overviewState();
    this.applyCamera();
    if (!this.running) this.render();
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.prevT = performance.now();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  stop(): void {
    this.running = false;
    this.renderer.setAnimationLoop(null);
  }

  private tick(): void {
    const now = performance.now();
    const dt = Math.min((now - this.prevT) / 1000, 0.1);
    this.prevT = now;
    const t0 = performance.now();

    if (this.flight) {
      const { from, to, t0: ft, dur, done } = this.flight;
      const k = Math.min((now - ft) / dur, 1);
      const e = ease(k);
      this.cam.target.lerpVectors(from.target, to.target, e);
      // arc the distance outward mid-flight: a pull-back reads as travel, not a zoom
      const lift = Math.sin(Math.PI * e) * Math.min(from.target.distanceTo(to.target) * 0.35, 14);
      this.cam.dist = THREE.MathUtils.lerp(from.dist, to.dist, e) + lift;
      this.cam.az = THREE.MathUtils.lerp(from.az, to.az, e);
      this.cam.el = THREE.MathUtils.lerp(from.el, to.el, e);
      this.cam.shift = THREE.MathUtils.lerp(from.shift, to.shift, e);
      this.cam.shiftY = THREE.MathUtils.lerp(from.shiftY ?? 0, to.shiftY ?? 0, e);
      this.applyCamera();
      if (k >= 1) {
        this.flight = null;
        done();
      }
    } else if (this.drift && !this.reducedMotion && now - this.lastInput > 2500 && !this.hoverId) {
      this.cam.az += dt * 0.022;
      this.applyCamera();
    }

    this.uniforms.uTime.value = now / 1000;
    this.drawLettering();
    this.lettering.material.opacity = THREE.MathUtils.clamp((this.cam.dist - 16) / 30, 0.06, 1);
    this.cb.frame();
    this.render();

    const cpu = performance.now() - t0;
    this.cpuSamples.push(cpu);
    if (this.cpuSamples.length >= 60) {
      this.stats.cpuMs = this.cpuSamples.reduce((a, b) => a + b, 0) / this.cpuSamples.length;
      this.cpuSamples.length = 0;
    }
  }

  private render(): void {
    this.renderer.render(this.scene, this.camera);
    this.stats.calls = this.renderer.info.render.calls;
  }

  dispose(): void {
    this.stop();
    this.ro.disconnect();
    for (const [t, type, fn, opts] of this.listeners) t.removeEventListener(type, fn, opts);
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose();
      const mat = m.material as THREE.Material | undefined;
      if (mat) {
        for (const v of Object.values(mat))
          if ((v as THREE.Texture)?.isTexture) (v as THREE.Texture).dispose();
        mat.dispose();
      }
    });
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
