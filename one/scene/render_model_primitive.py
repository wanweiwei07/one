import one.utils.constant as ouc
import one.scene.render_model as osrm
import one.scene.geometry_primitive as osgp

def gen_cylinder(length=0.1, radius=0.05, segments=8,
                 rgb=ouc.BasicColor.DEFAULT, alpha=1.0):
    """Gen cylinder render model from (0,0,0) to (0,0,length)."""
    geometry = osgp.gen_cylinder_geom(length, radius, segments)
    return osrm.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha)

def gen_cone(length=0.1, radius=0.05, segments=8,
             rgb=ouc.BasicColor.DEFAULT, alpha=1.0):
    """Gen cone render model from (0,0,0) to (0,0,length)."""
    geometry = osgp.gen_cone_geom(length, radius, segments)
    return osrm.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha)

def gen_sphere(radius=0.05, segments=8,
               rgb=ouc.BasicColor.DEFAULT, alpha=1.0):
    """Gen sphere render model at (0,0,0)."""
    geometry = osgp.gen_sphere_geom(radius, segments)
    return osrm.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha)

def gen_icosphere(radius=0.05, subdivisions=2,
                  rgb=ouc.BasicColor.DEFAULT, alpha=1.0):
    """Gen icosphere render model at (0,0,0)."""
    geometry = osgp.gen_icosphere_geom(radius, subdivisions)
    return osrm.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha)

def gen_box(half_extents=(0.05, 0.05, 0.05),
            rgb=ouc.BasicColor.DEFAULT, alpha=1.0):
    """Gen box render model centered at (0,0,0)."""
    geometry = osgp.gen_box_geom(half_extents)
    return osrm.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha)