import struct
import one.utils.math as rm
import one.scene.geometry_loader as geometry_loader

def save_stl(verts, faces, filename):
    verts = rm.asarray(verts, dtype=rm.float32)
    faces = rm.asarray(faces, dtype=rm.int32)
    with open(filename, "wb") as f:
        # 80-byte header
        header = b"ONE geometry binary STL"
        f.write(header.ljust(80, b"\0"))
        # number of triangles
        f.write(struct.pack("<I", len(faces)))
        for face in faces:
            v0, v1, v2 = verts[face]
            # compute normal
            normal = rm.cross(v1 - v0, v2 - v0)
            n = rm.linalg.norm(normal)
            if n > 1e-12:
                normal /= n
            else:
                normal[:] = 0.0
            # write normal + vertices
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            # attribute byte count (always 0)
            f.write(struct.pack("<H", 0))