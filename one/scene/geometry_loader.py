import numpy as np
import struct
import os
import xml
import one.scene.geometry as geom

_geometry_cache = {}


def load_geometry(path):
    if path in _geometry_cache:
        return _geometry_cache[path]
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        geometry = load_stl(path)
    elif ext == ".dae":
        geometry = load_dae(path)
    else:
        raise ValueError(f"Unsupported geometry format: {ext}")
    _geometry_cache[path] = geometry
    return geometry

# ==============================
# STL Loader
# ==============================
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
            return _load_stl_binary(path, tri_count)
        else:
            return _load_stl_ascii(path)


def _load_stl_binary(path, tri_count):
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
    return geom.Geometry(verts, faces)


def _load_stl_ascii(path):
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
    return geom.Geometry(np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32))


# ==============================
# DAE Loader
# ==============================
def load_dae(filename):
    tree = xml.etree.ElementTree.parse(filename)
    root = tree.getroot()
    # 1) auto extract namespace from root tag
    # example tag: "{http://www.collada.org/2005/11/COLLADASchema}COLLADA"
    tag = root.tag
    namespace = tag[tag.find("{") + 1: tag.find("}")]
    ns = {"ns": namespace}

    # 2) find float array containing vertex positions
    def find_positions():
        # search all <source> elements
        for src in root.findall(".//ns:source", ns):
            sid = src.attrib.get("id", "")
            # only take positions
            if "positions" in sid.lower():
                # float_array inside
                fa = src.find(".//ns:float_array", ns)
                if fa is None:
                    raise ValueError("positions source has no float_array")
                return np.fromstring(fa.text, sep=" ")
        raise ValueError("positions array not found")

    # 3) find face indices: triangles or polylist
    def find_indices():
        # triangles first
        p = root.find(".//ns:mesh//ns:triangles//ns:p", ns)
        if p is not None:
            return np.fromstring(p.text, sep=" ", dtype=np.int32)
        # fallback: polylist (common in COLLADA)
        p = root.find(".//ns:mesh//ns:polylist//ns:p", ns)
        if p is not None:
            return np.fromstring(p.text, sep=" ", dtype=np.int32)
        raise ValueError("no triangle/polylist indices found")

    floats = find_positions()
    # reshape to Nx3 vertices
    verts = floats.reshape((-1, 3)).astype(np.float32)
    idx = find_indices()
    faces = idx.reshape((-1, 3))
    return geom.Geometry(verts, faces)
