import one.robot_sim.base.robot_state as rstate

class RobotInstance:

    def __init__(self, structure):
        self.structure = structure
        self.state = rstate.RobotState(structure)
        self.visual_links = []
        for link in structure.link_order:
            clone = link.clone()
            self.visual_links.append(clone)

    def fk(self, qs=None, root_tfmat=None):
        if qs is not None:
            self.state.set_qs(qs)
        self.state.fk(root_tfmat)
        return self.state.link_wd_tfmats

    def update_scene(self):
        tfmats = self.state.link_wd_tfmats
        for i, link in enumerate(self.visual_links):
            link.set_tfmat(tfmats[i])

    def attach_to_scene(self, scene):
        for link in self.visual_links:
            scene.add(link)

    def get_ee_pose(self):
        idx = len(self.structure.link_order) - 1
        return self.state.link_wd_tfmats[idx]

    def snapshot(self):
        """Make static copy of current pose."""
        snap = []
        T = self.state.link_wd_tfmats
        for i, link in enumerate(self.structure.link_order):
            clone = link.clone()
            clone.set_tfmat(T[i].copy())
            snap.append(clone)
        return snap