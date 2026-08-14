/* ===========================================================================
   Hero 3D — a procedural DNA double helix.

   Built entirely from primitives at runtime, so there is no model file to load
   and nothing to go stale. Geometry is instanced: two strands of ~110 spheres
   plus their connecting rungs would be ~330 draw calls as individual meshes,
   which is wasteful for something purely decorative. As three InstancedMesh
   objects it is three draw calls and the tab stays cool.

   The scene reads its colours from the CSS custom properties, so the theme
   toggle recolours the helix without duplicating the palette in JavaScript.
   =========================================================================== */
import * as THREE from "three";

const RADIUS = 2.05;        // helix radius
const TURNS = 2.6;          // full rotations across the visible height
const HEIGHT = 15;
const PER_STRAND = 112;
const RUNG_EVERY = 4;

function cssColour(name, fallback) {
  const raw = getComputedStyle(document.documentElement)
    .getPropertyValue(name).trim();
  try { return new THREE.Color(raw || fallback); }
  catch { return new THREE.Color(fallback); }
}

export function initHelix(canvas) {
  if (!canvas) return null;

  const prefersStill = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, alpha: true, powerPreference: "low-power",
  });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 0, 15.5);

  const group = new THREE.Group();
  group.rotation.z = 0.22;
  scene.add(group);

  // ---------------------------------------------------------------- lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.PointLight(cssColour("--primary", "#2DD4BF"), 90, 60);
  key.position.set(6, 7, 9);
  scene.add(key);
  const rim = new THREE.PointLight(cssColour("--accent", "#38BDF8"), 70, 60);
  rim.position.set(-8, -5, 6);
  scene.add(rim);

  // ------------------------------------------------------------- geometry
  const nodeGeo = new THREE.SphereGeometry(0.20, 18, 18);
  const rungGeo = new THREE.CylinderGeometry(0.045, 0.045, 1, 8);

  const matA = new THREE.MeshStandardMaterial({
    color: cssColour("--primary", "#2DD4BF"), roughness: 0.25, metalness: 0.55,
    emissive: cssColour("--primary", "#2DD4BF"), emissiveIntensity: 0.28,
  });
  const matB = new THREE.MeshStandardMaterial({
    color: cssColour("--accent", "#38BDF8"), roughness: 0.25, metalness: 0.55,
    emissive: cssColour("--accent", "#38BDF8"), emissiveIntensity: 0.28,
  });
  const matRung = new THREE.MeshStandardMaterial({
    color: cssColour("--text-dim", "#8FA3BF"), roughness: 0.6, metalness: 0.2,
    transparent: true, opacity: 0.42,
  });

  const strandA = new THREE.InstancedMesh(nodeGeo, matA, PER_STRAND);
  const strandB = new THREE.InstancedMesh(nodeGeo, matB, PER_STRAND);
  const rungCount = Math.floor(PER_STRAND / RUNG_EVERY);
  const rungs = new THREE.InstancedMesh(rungGeo, matRung, rungCount);
  group.add(strandA, strandB, rungs);

  const dummy = new THREE.Object3D();
  const a = new THREE.Vector3(), b = new THREE.Vector3(), mid = new THREE.Vector3();

  const point = (i, phase) => {
    const t = i / (PER_STRAND - 1);
    const angle = t * Math.PI * 2 * TURNS + phase;
    return new THREE.Vector3(
      Math.cos(angle) * RADIUS,
      (t - 0.5) * HEIGHT,
      Math.sin(angle) * RADIUS,
    );
  };

  for (let i = 0; i < PER_STRAND; i++) {
    // taper the ends so the helix fades out instead of being cut off
    const t = i / (PER_STRAND - 1);
    const taper = Math.sin(t * Math.PI) ** 0.45;

    a.copy(point(i, 0));
    dummy.position.copy(a);
    dummy.scale.setScalar(taper);
    dummy.updateMatrix();
    strandA.setMatrixAt(i, dummy.matrix);

    b.copy(point(i, Math.PI));
    dummy.position.copy(b);
    dummy.scale.setScalar(taper);
    dummy.updateMatrix();
    strandB.setMatrixAt(i, dummy.matrix);

    if (i % RUNG_EVERY === 0 && i / RUNG_EVERY < rungCount) {
      mid.addVectors(a, b).multiplyScalar(0.5);
      dummy.position.copy(mid);
      dummy.scale.set(taper, a.distanceTo(b), taper);
      // cylinders are Y-up; aim this one along the a->b axis
      dummy.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        b.clone().sub(a).normalize(),
      );
      dummy.updateMatrix();
      rungs.setMatrixAt(i / RUNG_EVERY, dummy.matrix);
    }
  }
  strandA.instanceMatrix.needsUpdate = true;
  strandB.instanceMatrix.needsUpdate = true;
  rungs.instanceMatrix.needsUpdate = true;

  // ------------------------------------------------------------ particles
  const dustCount = 190;
  const positions = new Float32Array(dustCount * 3);
  for (let i = 0; i < dustCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 26;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 18;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 12 - 3;
  }
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({
    color: cssColour("--primary", "#2DD4BF"), size: 0.055,
    transparent: true, opacity: 0.5, sizeAttenuation: true,
  }));
  scene.add(dust);

  // --------------------------------------------------------------- resize
  function resize() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // keep the helix in view when the hero gets narrow on mobile
    camera.position.z = w < 720 ? 21 : 15.5;
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  resize();

  // ---------------------------------------------------------------- loop
  let pointerX = 0, pointerY = 0, running = true, raf = 0;
  addEventListener("pointermove", (e) => {
    pointerX = (e.clientX / innerWidth - 0.5) * 2;
    pointerY = (e.clientY / innerHeight - 0.5) * 2;
  }, { passive: true });

  // a hidden tab does not composite, so there is no reason to keep rendering
  document.addEventListener("visibilitychange", () => {
    running = !document.hidden;
    if (running) loop();
    else cancelAnimationFrame(raf);
  });

  const clock = new THREE.Clock();
  function loop() {
    if (!running) return;
    raf = requestAnimationFrame(loop);
    const t = clock.getElapsedTime();

    if (!prefersStill) {
      group.rotation.y = t * 0.26;
      group.position.y = Math.sin(t * 0.55) * 0.28;
      dust.rotation.y = -t * 0.045;
      // a slight lean toward the cursor, damped so it never feels twitchy
      group.rotation.x += (pointerY * 0.14 - group.rotation.x) * 0.045;
      camera.position.x += (pointerX * 1.1 - camera.position.x) * 0.045;
      camera.lookAt(0, 0, 0);
    }
    renderer.render(scene, camera);
  }
  loop();

  return {
    /** Re-read the palette after a theme change. */
    refreshTheme() {
      matA.color = matA.emissive = cssColour("--primary", "#2DD4BF");
      matB.color = matB.emissive = cssColour("--accent", "#38BDF8");
      matRung.color = cssColour("--text-dim", "#8FA3BF");
      key.color = cssColour("--primary", "#2DD4BF");
      rim.color = cssColour("--accent", "#38BDF8");
      dust.material.color = cssColour("--primary", "#2DD4BF");
    },
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      [nodeGeo, rungGeo, dustGeo].forEach((g) => g.dispose());
      [matA, matB, matRung, dust.material].forEach((m) => m.dispose());
      renderer.dispose();
    },
  };
}
