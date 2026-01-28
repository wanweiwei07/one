import one.utils.constant as ouc
import one.geom.geometry as ogg
import one.scene.render_model as osrm


def gen_mesh_rmodel(vs, fs, rgb, alpha=1.0):
    geom = ogg.gen_geom_from_raw(vs, fs)
    return osrm.RenderModel(geom=geom, rgb=rgb, alpha=alpha)


def gen_pcd_rmodel(vs, vrgbs, alpha=1.0):
    geom = ogg.gen_geom_from_raw(vs)
    return osrm.RenderModel(geom=geom, vrgbs=vrgbs, alpha=alpha)


def gen_cylinder_rmodel(length=0.1, radius=0.05, n_segs=8,
                        rotmat=None, pos=None,
                        rgb=ouc.BasicColor.DEFAULT, alpha=1.0):
    """Gen cylinder render model from (0,0,0) to (0,0,length)."""
    geom = ogg.gen_cylinder_geom(length, radius, n_segs)
    return osrm.RenderModel(
        geom=geom, rotmat=rotmat, pos=pos, rgb=rgb, alpha=alpha)


def gen_cone_rmodel(length=0.1, radius=0.05, n_segs=8,
                    rgb=ouc.BasicColor.DEFAULT, alpha=1.0):
    """Gen cone render model from (0,0,0) to (0,0,length)."""
    geometry = ogg.gen_cone_geom(length, radius, n_segs)
    return osrm.RenderModel(
        geom=geometry, rgb=rgb, alpha=alpha)


def gen_sphere_rmodel(radius=0.05, n_segs=8,
                      rgb=ouc.BasicColor.DEFAULT, alpha=1.0):
    """Gen sphere render model at (0,0,0)."""
    geometry = ogg.gen_sphere_geom(radius, n_segs)
    return osrm.RenderModel(
        geom=geometry, rgb=rgb, alpha=alpha)


def gen_icosphere_rmodel(radius=0.05, n_subs=2,
                         rgb=ouc.BasicColor.DEFAULT, alpha=1.0):
    """Gen icosphere render model at (0,0,0)."""
    geometry = ogg.gen_icosphere_geom(radius, n_subs)
    return osrm.RenderModel(
        geom=geometry, rgb=rgb, alpha=alpha)


def gen_box_rmodel(half_extents=(0.05, 0.05, 0.05),
                   rgb=ouc.BasicColor.DEFAULT, alpha=1.0):
    """Gen box render model centered at (0,0,0)."""
    geometry = ogg.gen_box_geom(half_extents)
    return osrm.RenderModel(
        geom=geometry, rgb=rgb, alpha=alpha)


def gen_arrow_rmodel(length, shaft_radius, head_length, head_radius,
                     n_segs, rotmat=None, pos=None,
                     rgb=ouc.BasicColor.DEFAULT, alpha=1.0):
    """Gen arrow render model from (0,0,0) to (0,0,length)."""
    geometry = ogg.gen_arrow_geom(
        length, shaft_radius, head_length,
        head_radius, n_segs)
    return osrm.RenderModel(
        geom=geometry, rotmat=rotmat,
        pos=pos, rgb=rgb, alpha=alpha)
