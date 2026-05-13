import numpy as np
import pyrealsense2 as rs
from one import ovw, ossop

STRIDE = 2
DEPTH_MIN, DEPTH_MAX = 0.2, 3.0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)
pc = rs.pointcloud()

base = ovw.World(cam_pos=(0.0, -0.6, -0.8), cam_lookat_pos=(0.0, 0.0, 1.2))
state = {"pcd": None}


def update(_dt):
    frames = pipeline.poll_for_frames()
    if not frames:
        return
    frames = align.process(frames)
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    if not depth or not color:
        return

    pc.map_to(color)
    points = pc.calculate(depth)
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    uvs = np.asanyarray(
        points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

    H, W = depth.get_height(), depth.get_width()
    verts = verts.reshape(H, W, 3)[::STRIDE, ::STRIDE].reshape(-1, 3)
    uvs = uvs.reshape(H, W, 2)[::STRIDE, ::STRIDE].reshape(-1, 2)

    z = verts[:, 2]
    keep = (z > DEPTH_MIN) & (z < DEPTH_MAX)
    verts = verts[keep]
    uvs = uvs[keep]
    if verts.shape[0] == 0:
        return

    color_img = np.asanyarray(color.get_data())
    ch, cw = color_img.shape[:2]
    px = np.clip((uvs[:, 0] * cw).astype(np.int32), 0, cw - 1)
    py = np.clip((uvs[:, 1] * ch).astype(np.int32), 0, ch - 1)
    rgb = color_img[py, px][:, ::-1].astype(np.float32) / 255.0

    old = state["pcd"]
    if old is not None:
        old.detach_from(base.scene)
    pcd = ossop.point_cloud(verts.astype(np.float32), rgb)
    pcd.attach_to(base.scene)
    state["pcd"] = pcd


base.schedule_interval(update, interval=1.0 / 30.0)


@base.event
def on_close():
    pipeline.stop()


base.run()
