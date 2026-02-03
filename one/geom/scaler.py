import glob
import numpy as np
import one.geom.loader as ogl


def scale_stl_files(scale=(1,1,1)):
    stl_files = sorted(glob.glob("*.stl"))
    if not stl_files:
        print("No .stl files found in current directory.")
        return
    scale = np.asarray(scale, dtype=np.float32)
    if scale.shape != (3,):
        raise ValueError("scale must be a 3-tuple (sx, sy, sz)")
    print(f"Scaling {len(stl_files)} STL files by {scale} (overwrite).")
    for path in stl_files:
        geom = ogl.load_geometry(path)
        vs = np.asarray(geom.vs, dtype=np.float32) * scale[None, :]
        fs = np.asarray(geom.fs, dtype=np.int32)
        ogl._save_stl(vs, fs, path)
        print(f"  scaled: {path}")


if __name__ == "__main__":
    scale_stl_files(scale=(0.001,0.001,0.001))
