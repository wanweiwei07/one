class RobotSnapshot:

    def __init__(self, instance):
        self.visual_links = []
        for i, link in enumerate(instance.structure.link_order):
            clone = link.clone()
            clone.set_homomat(instance.state.link_wd_homomats[i])
            self.visual_links.append(clone)

    def attach_to(self, scene):
        for link in self.visual_links:
            scene.add(link)