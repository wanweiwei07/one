import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';

export class PygletStyleCameraControls {
  constructor(camera, domElement) {
    this.camera = camera;
    this.domElement = domElement;
    this.lookAt = new THREE.Vector3(0, 0, 0);
    this.up = new THREE.Vector3(0, 0, 1);
    this.sensitivityOrbit = 0.002;
    this.sensitivityPan = 0.0003;
    this.sensitivityZoom = 0.05;

    this._isLeft = false;
    this._isMiddle = false;
    this._lastX = 0;
    this._lastY = 0;

    this._bind();
  }

  setLookAt(vec3) {
    this.lookAt.copy(vec3);
    this.camera.lookAt(this.lookAt);
  }

  _bind() {
    this.domElement.addEventListener('mousedown', (e) => this._onDown(e));
    this.domElement.addEventListener('mousemove', (e) => this._onMove(e));
    this.domElement.addEventListener('mouseup', () => this._onUp());
    this.domElement.addEventListener('mouseleave', () => this._onUp());
    this.domElement.addEventListener('wheel', (e) => this._onWheel(e), { passive: false });
    this.domElement.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  _onDown(e) {
    this._lastX = e.clientX;
    this._lastY = e.clientY;
    if (e.button === 0) this._isLeft = true;
    if (e.button === 1) this._isMiddle = true;
  }

  _onMove(e) {
    const dx = e.clientX - this._lastX;
    const dy = e.clientY - this._lastY;
    this._lastX = e.clientX;
    this._lastY = e.clientY;

    if (this._isLeft) {
      this.orbit(dx, dy);
    } else if (this._isMiddle) {
      this.pan(dx, dy);
    }
  }

  _onUp() {
    this._isLeft = false;
    this._isMiddle = false;
  }

  _onWheel(e) {
    e.preventDefault();
    const scroll = e.deltaY > 0 ? 1 : -1;
    this.zoom(scroll);
  }

  orbit(dx, dy) {
    const right = new THREE.Vector3();
    const up = new THREE.Vector3();
    this.camera.updateMatrixWorld(true);
    this.camera.matrixWorld.extractBasis(right, up, new THREE.Vector3());
    this._orbitAroundAxis(up, -dx * this.sensitivityOrbit);
    this._orbitAroundAxis(right, -dy * this.sensitivityOrbit);
  }

  pan(dx, dy) {
    const right = new THREE.Vector3();
    const up = new THREE.Vector3();
    this.camera.updateMatrixWorld(true);
    this.camera.matrixWorld.extractBasis(right, up, new THREE.Vector3());
    const delta = new THREE.Vector3()
      .addScaledVector(right, -dx * this.sensitivityPan)
      .addScaledVector(up, dy * this.sensitivityPan);
    this.camera.position.add(delta);
    this.lookAt.add(delta);
    this.camera.lookAt(this.lookAt);
  }

  zoom(delta) {
    const dir = new THREE.Vector3().subVectors(this.camera.position, this.lookAt);
    const zoomAmount = -delta * this.sensitivityZoom;
    this.camera.position.addScaledVector(dir, zoomAmount);
    this.camera.lookAt(this.lookAt);
  }

  _orbitAroundAxis(axis, angleRad) {
    const dir = new THREE.Vector3().subVectors(this.camera.position, this.lookAt);
    dir.applyAxisAngle(axis.clone().normalize(), angleRad);
    this.camera.position.copy(this.lookAt).add(dir);
    this.camera.up.copy(this.up);
    this.camera.lookAt(this.lookAt);
  }
}
