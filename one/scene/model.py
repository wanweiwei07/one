import one.utils.math as rm
import one.scene.node as nd

class Model(nd.Node):

    def __init__(self, geometry=None, rotmat=None, pos=None, parent=None):
        super().__init__(rotmat=rotmat, pos=pos, parent=parent)
        self.geometry = geometry  # geometry.GeometryBase