import asyncio
import json
import builtins
from typing import Dict, Any, List
import threading
import numpy as np
import websockets

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------

def _mat4_to_list(mat: np.ndarray) -> List[float]:
    return mat.astype(np.float32, copy=False).ravel().tolist()

def _serialize_mesh(model, node, mesh_id: str) -> Dict[str, Any]:
    geom = model.geom
    verts = geom.vs.astype(np.float32, copy=False).ravel().tolist()
    faces = geom.fs.astype(np.int32, copy=False).ravel().tolist()
    rgba = [float(model.rgb[0]), float(model.rgb[1]), float(model.rgb[2]), float(model.alpha)]
    return {
        "id": mesh_id,
        "vertices": verts,
        "faces": faces,
        "rgba": rgba
    }

def collect_scene_meshes(scene) -> List[Dict[str, Any]]:
    meshes = []
    for sobj in scene:
        # visuals
        for idx, model in enumerate(sobj.visuals):
            sobj_id=id(sobj)
            mesh_id = f"{sobj_id}:visual:{idx}"
            meshes.append(_serialize_mesh(model, sobj.node, mesh_id))
        # collision visualization (if enabled)
        if sobj.toggle_render_collision:
            for idx, col in enumerate(sobj.collisions):
                model = col.to_render_model()
                sobj_id=id(sobj)
                mesh_id = f"{sobj_id}:collision:{idx}"
                meshes.append(_serialize_mesh(model, sobj.node, mesh_id))
    return meshes

def collect_scene_transforms(scene) -> Dict[str, List[float]]:
    transforms = {}
    for sobj in scene:
        for idx, model in enumerate(sobj.visuals):
            sobj_id=id(sobj)
            mesh_id = f"{sobj_id}:visual:{idx}"
            tf = (sobj.node.wd_tf @ model.tf).T  # match pyglet render
            transforms[mesh_id] = _mat4_to_list(tf)
        if sobj.toggle_render_collision:
            for idx, col in enumerate(sobj.collisions):
                model = col.to_render_model()
                sobj_id=id(sobj)
                mesh_id = f"{sobj_id}:collision:{idx}"
                tf = (sobj.node.wd_tf @ model.tf).T
                transforms[mesh_id] = _mat4_to_list(tf)
    return transforms

# ------------------------------------------------------------
# WebSocket server
# ------------------------------------------------------------

async def handler(websocket, scene, hz=30):
    init_payload = {
        "type": "scene_init",
        "meshes": collect_scene_meshes(scene)
    }
    await websocket.send(json.dumps(init_payload))

    interval = 1.0 / float(hz)
    try:
        while True:
            update_payload = {
                "type": "scene_update",
                "transforms": collect_scene_transforms(scene)
            }
            await websocket.send(json.dumps(update_payload))
            await asyncio.sleep(interval)
    except websockets.ConnectionClosed:
        return

async def start_stream(scene, host="127.0.0.1", port=8000, hz=30):
    async with websockets.serve(lambda ws: handler(ws, scene, hz=hz), host, port, max_size=2**24):
        await asyncio.Future()  # run forever

def run_stream(scene, host="127.0.0.1", port=8000, hz=30):
    def _thread_entry():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.create_task(start_stream(scene, host=host, port=port, hz=hz))
        loop.run_forever()

    thread = threading.Thread(target=_thread_entry, daemon=True)
    thread.start()
# ------------------------------------------------------------
# Entry point (optional)
# ------------------------------------------------------------

def get_scene_from_builtins():
    base = getattr(builtins, "base", None)
    if base is None or getattr(base, "scene", None) is None:
        raise RuntimeError("No scene found. Ensure `builtins.base.scene` exists.")
    return base.scene

async def main():
    scene = get_scene_from_builtins()
    await start_stream(scene)

if __name__ == "__main__":
    asyncio.run(main())