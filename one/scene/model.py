import numpy as np
import one.utils.decorator as deco
import one.utils.constant as const
import one.scene.geometry as geom
import one.viewer.device_buffer as dvb


class Model:

    def __init__(self, geometry=None, rotmat=None, pos=None,
                 rgb=None, alpha=1.0, shader=None):
        if isinstance(geometry, tuple):
            verts = geometry[0]
            faces = geometry[1] if len(geometry) > 1 else None
            per_vert_rgbs = geometry[2] if len(geometry) > 2 else None
            self.geometry = geom.Geometry(verts=verts, faces=faces, per_vert_rgbs=per_vert_rgbs)
        else:
            self.geometry = geometry
        self.rgb = const.BasicColor.DEFAULT if rgb is None else rgb
        self.alpha = alpha
        self.shader = shader
        self._rotmat = np.eye(3) if rotmat is None else rotmat
        self._pos = np.zeros(3) if pos is None else pos

    def get_device_buffer(self):
        if self.geometry.device_buffer is None:
            self.geometry.device_buffer = dvb.DeviceBuffer(self.geometry)
        return self.geometry.device_buffer

    @property
    @deco.readonly_view
    def local_tfmat(self):
        tfmat = np.eye(4)
        tfmat[:3, :3] = self._rotmat
        tfmat[:3, 3] = self._pos
        return tfmat

# class Model(nd.Node):
#
#     def __init__(self, geometry=None, rotmat=None, pos=None,
#                  rgb=None, alpha=1.0, parent=None, toggle_collision=False):
#         super().__init__(rotmat=rotmat, pos=pos, parent=parent)
#         if isinstance(geometry, tuple):
#             verts = geometry[0]
#             faces = geometry[1] if len(geometry) > 1 else None
#             per_vert_rgbs = geometry[2] if len(geometry) > 2 else None
#             self.geometry = geom.Geometry(verts=verts, faces=faces, per_vert_rgbs=per_vert_rgbs)
#         else:
#             self.geometry = geometry  # geometry.GeometryBase
#         self._fcl_object = self._build_fcl_object() if toggle_collision else None  # fcl.CollisionObject
#         self.rgb = const.BasicColor.DEFAULT if rgb is None else rgb
#         self.alpha = alpha
#
#     def _build_fcl_object(self):
#         if self.geometry is not None and self.geometry.faces is not None:
#             bvh = fcl.BVHModel()
#             bvh.beginModel(self.geometry.verts.shape[0], self.geometry.faces.shape[0])
#             bvh.addSubModel(self.geometry.verts, self.geometry.faces)
#             bvh.endModel()
#             return fcl.CollisionObject(bvh, fcl.Transform(self.wd_rotmat, self.wd_pos))
#
#     @property
#     def fcl_object(self):
#         if self._fcl_object is None:
#             return None
#         else:
#             self._fcl_object.setTransform(self.wd_tfmat)
#             self._fcl_object.computeAABB()
#             return self._fcl_object
#
#     @property
#     def is_collision(self):
#         return self._fcl_object is not None
