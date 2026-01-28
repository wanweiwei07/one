import os
import xml
import struct
import numpy as np
import one.geom.geometry as osg


def load_geometry(path):
    if path in osg._geom_cache:
        return osg._geom_cache[path]
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        geometry = _load_stl(path)
    elif ext == ".dae":
        geometry = _load_dae(path)
    else:
        raise ValueError(f"Unsupported geom format: {ext}")
    osg._geom_cache[path] = geometry
    return geometry


# ==============================
# STL Loader and Saver
# ==============================
def _load_stl(path):
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


def _save_stl(vs, fs, filename):
    """Save geom to binary STL file."""
    vs = np.asarray(vs, dtype=np.float32)
    fs = np.asarray(fs, dtype=np.int32)
    with open(filename, "wb") as f:
        # 80-byte header
        header = b"ONE geom binary STL"
        f.write(header.ljust(80, b"\0"))
        # number of triangles
        f.write(struct.pack("<I", len(fs)))
        for face in fs:
            v0, v1, v2 = vs[face]
            # compute normal
            normal = np.cross(v1 - v0, v2 - v0)
            n = np.linalg.norm(normal)
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


def _load_stl_binary(path, tri_count):
    vs = np.zeros((tri_count * 3, 3), dtype=np.float32)
    fs = np.zeros((tri_count, 3), dtype=np.int32)
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
            vs[base + 0] = v0
            vs[base + 1] = v1
            vs[base + 2] = v2
            fs[i] = (base + 0, base + 1, base + 2)
            f.read(2)  # skip attribute bytes
    return osg.gen_geom_from_raw(vs, fs)


def _load_stl_ascii(path):
    vs = []
    fs = []
    current_face = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                _, x, y, z = line.split()
                current_face.append([float(x), float(y), float(z)])
            elif line.startswith("endfacet"):
                i0 = len(vs) + 0
                i1 = len(vs) + 1
                i2 = len(vs) + 2
                vs.extend(current_face)
                fs.append([i0, i1, i2])
                current_face = []
    return osg.gen_geom_from_raw(np.array(vs, dtype=np.float32),
                                 np.array(fs, dtype=np.int32))


# ==============================
# DAE Loader
# ==============================
def _load_dae(filename):
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
    vs = floats.reshape((-1, 3)).astype(np.float32)
    idx = find_indices()
    fs = idx.reshape((-1, 3))
    return osg.gen_geom_from_raw(vs, fs)
