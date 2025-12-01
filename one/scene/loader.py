import numpy as np
import struct
import os


def load_stl(path):
    with open(path, "rb") as f:
        f.read(80)  # ignore header
        tri_count_bytes = f.read(4)
        if len(tri_count_bytes) < 4:
            raise ValueError("Invalid STL file")
        tri_count = struct.unpack("<I", tri_count_bytes)[0]
        # Expected binary size = 84 + M * 50
        file_size = os.path.getsize(path)
        expected = 84 + tri_count * 50
        if file_size == expected:
            return load_stl_binary(path, tri_count)
        else:
            return load_stl_ascii(path)


def load_stl_binary(path, tri_count):
    verts = np.zeros((tri_count * 3, 3), dtype=np.float32)
    faces = np.zeros((tri_count, 3), dtype=np.int32)
    with open(path, "rb") as f:
        f.read(80)  # header
        f.read(4)  # tri count
        for i in range(tri_count):
            f.read(12)  # skip normal
            # triangle vertices
            v0 = struct.unpack("<fff", f.read(12))
            v1 = struct.unpack("<fff", f.read(12))
            v2 = struct.unpack("<fff", f.read(12))
            base = i * 3
            verts[base + 0] = v0
            verts[base + 1] = v1
            verts[base + 2] = v2
            faces[i] = (base + 0, base + 1, base + 2)
            f.read(2)  # skip attribute bytes
    return verts, faces


def load_stl_ascii(path):
    verts = []
    faces = []
    current_face = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                _, x, y, z = line.split()
                current_face.append([float(x), float(y), float(z)])
            elif line.startswith("endfacet"):
                i0 = len(verts) + 0
                i1 = len(verts) + 1
                i2 = len(verts) + 2
                verts.extend(current_face)
                faces.append([i0, i1, i2])
                current_face = []
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)
