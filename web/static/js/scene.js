/* ===========================================================================
   Hero 3D — a double helix drawn as line work.

   Deliberately not a glowing neon render. The rest of the interface is paper,
   hairlines and one accent colour, and a luminous centrepiece would fight it.
   So the helix is drawn the way it would be in a textbook: two continuous
   strands, thin rungs between them, small nodes at the base pairs, all in a
   single ink colour at low opacity. It reads as an illustration rather than a
   screensaver, and it costs almost nothing to render.

   Everything is procedural — no model file, no textures, no lights, because
   unlit materials need none. Colour comes from the CSS tokens so the theme
   toggle carries it.
   =========================================================================== */
import * as THREE from "three";

const RADIUS = 1.75;
const TURNS = 2.15;
const HEIGHT = 11;
const SEGMENTS = 220;      // samples along each strand — high enough to read smooth
const RUNGS = 26;

function token(name, fallback) {
  const raw = getComputedStyle(document.documentElement)
    .getPropertyValue(name).trim();
  try { return new THREE.Color(raw || fallback); }
  catch { return new THREE.Color(fallback); }
}

const strandPoint = (t, phase) => new THREE.Vector3(
  Math.cos(t * Math.PI * 2 * TURNS + phase) * RADIUS,
  (t - 0.5) * HEIGHT,
  Math.sin(t * Math.PI * 2 * TURNS + phase) * RADIUS,
);

export function initHelix(canvas) {
  if (!canvas) return null;

  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, alpha: true, powerPreference: "low-power",
  });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
  camera.position.set(0, 0, 15);

  const group = new THREE.Group();
  group.rotation.z = 0.12;
  scene.add(group);

  const inkStrong = token("--accent", "#0F6C63");
  const inkSoft = token("--ink-3", "#7C838B");

  // ---- the two strands, as continuous polylines -------------------------
  const strandMat = new THREE.LineBasicMaterial({
    color: inkStrong, transparent: true, opacity: 0.85,
  });
  const strands = [0, Math.PI].map((phase) => {
    const pts = [];
    for (let i = 0; i < SEGMENTS; i++) pts.push(strandPoint(i / (SEGMENTS - 1), phase));
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts), strandMat);
    group.add(line);
    return line;
  });

  // ---- rungs between the strands ----------------------------------------
  const rungMat = new THREE.LineBasicMaterial({
    color: inkSoft, transparent: true, opacity: 0.4,
  });
  const rungPts = [];
  const nodePts = [];
  for (let i = 0; i < RUNGS; i++) {
    // inset from the ends so the ladder fades out rather than stopping abruptly
    const t = 0.055 + (i / (RUNGS - 1)) * 0.89;
    const a = strandPoint(t, 0), b = strandPoint(t, Math.PI);
    rungPts.push(a, b);
    nodePts.push(a, b);
  }
  const rungs = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(rungPts), rungMat);
  group.add(rungs);

  // ---- small nodes at each base pair ------------------------------------
  const nodeMat = new THREE.PointsMaterial({
    color: inkStrong, size: 0.085, transparent: true, opacity: 0.9,
    sizeAttenuation: true,
  });
  const nodes = new THREE.Points(
    new THREE.BufferGeometry().setFromPoints(nodePts), nodeMat);
  group.add(nodes);

  // ------------------------------------------------------------- resizing
  function resize() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.position.z = w < 420 ? 19 : 15;
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  resize();

  // ----------------------------------------------------------------- loop
  let running = true, raf = 0;
  document.addEventListener("visibilitychange", () => {
    running = !document.hidden;
    if (running) loop(); else cancelAnimationFrame(raf);
  });

  const clock = new THREE.Clock();
  function loop() {
    if (!running) return;
    raf = requestAnimationFrame(loop);
    if (!still) {
      // one slow axis only. Anything faster starts to demand attention, which
      // is the opposite of what a background element should do.
      group.rotation.y = clock.getElapsedTime() * 0.16;
    }
    renderer.render(scene, camera);
  }
  loop();

  return {
    refreshTheme() {
      strandMat.color = nodeMat.color = token("--accent", "#0F6C63");
      rungMat.color = token("--ink-3", "#7C838B");
    },
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      strands.forEach((l) => l.geometry.dispose());
      rungs.geometry.dispose();
      nodes.geometry.dispose();
      [strandMat, rungMat, nodeMat].forEach((m) => m.dispose());
      renderer.dispose();
    },
  };
}
