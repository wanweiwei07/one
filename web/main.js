import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { PygletStyleCameraControls } from './PygletStyleCameraControls.js';

const container = document.getElementById('app');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf2f2f0);

const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.01,
  1000
);
camera.position.set(2, 2, 2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const controls = new PygletStyleCameraControls(camera, renderer.domElement);
controls.setLookAt(new THREE.Vector3(0, 0, 0));

const ambient = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambient);
const dir = new THREE.DirectionalLight(0xffffff, 0.6);
dir.position.set(3, 3, 5);
scene.add(dir);

const meshMap = new Map();

function buildMesh({ id, vertices, faces, rgba }) {
  const geometry = new THREE.BufferGeometry();
  const verts = new Float32Array(vertices);
  geometry.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geometry.setIndex(faces);
  geometry.computeVertexNormals();

  const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
  const material = new THREE.MeshStandardMaterial({
    color,
    transparent: rgba[3] < 0.999,
    opacity: rgba[3]
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.matrixAutoUpdate = false;
  meshMap.set(id, mesh);
  scene.add(mesh);
}

function applyTransforms(transformDict) {
  Object.entries(transformDict).forEach(([id, mat]) => {
    const mesh = meshMap.get(id);
    if (!mesh) return;
    const m = new THREE.Matrix4();
    m.fromArray(mat);
    mesh.matrix.copy(m);
  });
}

function connect() {
  const ws = new WebSocket('ws://127.0.0.1:8000/ws');
  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === 'scene_init') {
      payload.meshes.forEach(buildMesh);
    } else if (payload.type === 'scene_update') {
      applyTransforms(payload.transforms);
    }
  };
  ws.onclose = () => {
    setTimeout(connect, 1000);
  };
}

function animate() {
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

connect();
animate();
