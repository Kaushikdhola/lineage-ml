import * as THREE from "three";
import type { SymbolBus } from "../math/symbols";

/**
 * Lab scene for the softmax family. Each logit z_i is an orb: its altitude is the logit,
 * its volume is the probability (radius ∝ ∛p), and the ring beneath is the normalizer that
 * makes the volumes sum to one. Bound to the symbol bus in both directions.
 * Placeholder quality: /scene replaces it with the production design.
 */
export class SoftmaxField {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  private orbs: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>[] = [];
  private halos: THREE.Sprite[] = [];
  private ring: THREE.LineLoop;
  private target: { p: number[]; z: number[] } = { p: [], z: [] };
  private focusKey: string | null = null;
  private raf = 0;
  private ro: ResizeObserver;
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2(9, 9);
  private disposers: (() => void)[] = [];

  constructor(
    private readonly el: HTMLElement,
    private readonly bus: SymbolBus,
    n: number,
  ) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    el.append(this.renderer.domElement);

    this.camera.position.set(0, 3.2, 6.4);
    this.camera.lookAt(0, 0.3, 0);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.35));
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(3, 6, 4);
    this.scene.add(key);

    const ringGeo = new THREE.BufferGeometry().setFromPoints(
      Array.from({ length: 128 }, (_, i) => {
        const a = (i / 128) * Math.PI * 2;
        return new THREE.Vector3(Math.cos(a) * 2.4, -0.9, Math.sin(a) * 2.4);
      }),
    );
    this.ring = new THREE.LineLoop(
      ringGeo,
      new THREE.LineBasicMaterial({ color: 0x5ee3d6, transparent: true, opacity: 0.25 }),
    );
    this.scene.add(this.ring);

    const haloTex = this.haloTexture();
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 - Math.PI / 2;
      const orb = new THREE.Mesh(
        new THREE.SphereGeometry(1, 48, 32),
        new THREE.MeshStandardMaterial({
          color: 0x0d1416,
          emissive: 0x5ee3d6,
          emissiveIntensity: 0.4,
          roughness: 0.35,
          metalness: 0.1,
        }),
      );
      orb.position.set(Math.cos(a) * 2.4, 0, Math.sin(a) * 2.4);
      orb.userData.index = i;
      const halo = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: haloTex,
          color: 0x5ee3d6,
          blending: THREE.AdditiveBlending,
          transparent: true,
          depthWrite: false,
        }),
      );
      this.scene.add(orb, halo);
      this.orbs.push(orb);
      this.halos.push(halo);
    }

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(el);
    this.resize();

    const onMove = (e: PointerEvent) => {
      const r = this.renderer.domElement.getBoundingClientRect();
      this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const hit = this.raycaster.intersectObjects(this.orbs)[0];
      this.bus.focus(hit ? "zi" : null, "scene");
    };
    const onLeave = () => this.bus.focus(null, "scene");
    this.renderer.domElement.addEventListener("pointermove", onMove);
    this.renderer.domElement.addEventListener("pointerleave", onLeave);
    this.disposers.push(
      () => this.renderer.domElement.removeEventListener("pointermove", onMove),
      () => this.renderer.domElement.removeEventListener("pointerleave", onLeave),
      bus.onFocus((k) => (this.focusKey = k)),
    );

    const clock = new THREE.Clock();
    const tick = () => {
      this.raf = requestAnimationFrame(tick);
      this.frame(clock.getElapsedTime());
    };
    tick();
  }

  update(p: number[], z: number[]): void {
    this.target = { p, z };
  }

  private frame(t: number): void {
    const f = this.focusKey;
    const lit = (i: number) =>
      f === "zi" || f === "z" || f === "softmax" || (f === "tau" && this.target.p[i] > 0);
    this.orbs.forEach((orb, i) => {
      const p = this.target.p[i] ?? 0;
      const z = this.target.z[i] ?? 0;
      const radius = 0.08 + 0.75 * Math.cbrt(p);
      const pulse = f === "tau" ? 1 + 0.06 * Math.sin(t * 6 + i) : 1;
      orb.scale.lerp(new THREE.Vector3(radius * pulse, radius * pulse, radius * pulse), 0.12);
      orb.position.y += (z * 0.28 - orb.position.y) * 0.12;
      const glow = 0.25 + 2.2 * p + (lit(i) ? 0.9 : 0);
      orb.material.emissiveIntensity += (glow - orb.material.emissiveIntensity) * 0.15;
      const halo = this.halos[i];
      halo.position.copy(orb.position);
      const hs = orb.scale.x * (3.2 + (lit(i) ? 1 : 0));
      halo.scale.set(hs, hs, 1);
      halo.material.opacity = 0.15 + 0.6 * p + (lit(i) ? 0.25 : 0);
    });
    const ringTarget = f === "norm" || f === "softmax" ? 0.9 : 0.22;
    const m = this.ring.material as THREE.LineBasicMaterial;
    m.opacity += (ringTarget - m.opacity) * 0.15;
    this.scene.rotation.y = t * 0.08;
    this.renderer.render(this.scene, this.camera);
  }

  private resize(): void {
    const w = this.el.clientWidth || 1;
    const h = this.el.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private haloTexture(): THREE.Texture {
    const c = document.createElement("canvas");
    c.width = c.height = 128;
    const g = c.getContext("2d")!;
    const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0, "rgba(255,255,255,0.9)");
    grad.addColorStop(0.25, "rgba(255,255,255,0.25)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grad;
    g.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(c);
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
    this.disposers.forEach((d) => d());
    this.scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      mesh.geometry?.dispose();
      const mat = mesh.material as THREE.Material | undefined;
      mat?.dispose();
    });
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
