import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

class MJCFCompiler:

    @staticmethod
    def set_vec3(elem, name, v):
        elem.set(name, f"{v[0]} {v[1]} {v[2]}")

    @staticmethod
    def set_quat(elem, name, q):
        elem.set(name, f"{q[3]} {q[0]} {q[1]} {q[2]}")

    def compile_mjcf(self, world, return_str=True): # TODO : fix return type
        mj = ET.Element("mujoco")
        if world.option:
            opt = ET.SubElement(mj, "option")
            self.set_vec3(opt, "gravity", world.option.gravity)
            opt.set("timestep", str(world.option.timestep))
        if world.default:
            self.compile_default(world.default, mj)
        if world.assets:  # only mesh assets for now
            asset_el = ET.SubElement(mj, "asset")
            for a in world.assets:
                self.compile_mesh_asset(a, asset_el)
        wb = ET.SubElement(mj, "worldbody")
        for child in world.root_body.children:
            self.compile_body(child, wb)
        if world.actuators:
            act_el = ET.SubElement(mj, "actuator")
            for a in world.actuators:
                self.compile_actuator(a, act_el)
        if world.contact_excludes:
            contact_el = ET.SubElement(mj, "contact")
            for b1, b2 in world.contact_excludes:
                ex = ET.SubElement(contact_el, "exclude")
                ex.set("body1", b1.name)
                ex.set("body2", b2.name)
        # compiler settings
        compiler = ET.SubElement(mj, "compiler")
        compiler.set("angle", "radian")
        if return_str:
            binary = ET.tostring(mj, 'utf-8')
            text = binary.decode('utf-8')
            pretty = minidom.parseString(text).toprettyxml(indent="  ")
            return pretty
        return mj

    def compile_default(self, default_node, parent_el):
        de = ET.SubElement(parent_el, "default")
        if default_node.joint:
            je = ET.SubElement(de, "joint")
            for k, v in default_node.joint.items():
                if isinstance(v, (tuple, list)):
                    v = " ".join(map(str, v))
                je.set(k, str(v))
        if default_node.geom:
            ge = ET.SubElement(de, "geom")
            for k, v in default_node.geom.items():
                if isinstance(v, (tuple, list)):
                    v = " ".join(map(str, v))
                ge.set(k, str(v))

    def compile_mesh_asset(self, asset, parent_el):
        m = ET.SubElement(parent_el, "mesh")
        m.set("name", asset.name)
        m.set("file", asset.path)

    def compile_body(self, node, parent_el):
        body_el = ET.SubElement(parent_el, "body")
        body_el.set("name", node.name)
        self.set_vec3(body_el, "pos", node.pos)
        self.set_quat(body_el, "quat", node.quat)
        # joints first
        for j in node.hosting_jnts:
            self.compile_joint(j, body_el)
        # inertial
        if node.inertial:
            i = ET.SubElement(body_el, "inertial")
            i.set("mass", str(node.inertial.mass))
            self.set_vec3(i, "pos", node.inertial.com)
            I = node.inertial.inertia
            Ixx, Ixy, Ixz = I[0]
            Iyx, Iyy, Iyz = I[1]
            Izx, Izy, Izz = I[2]
            is_diag = (abs(Ixy) < 1e-10 and abs(Ixz) < 1e-10 and
                       abs(Iyx) < 1e-10 and abs(Iyz) < 1e-10 and
                       abs(Izx) < 1e-10 and abs(Izy) < 1e-10)
            if is_diag:
                i.set("diaginertia", f"{Ixx} {Iyy} {Izz}")
            else:
                i.set("fullinertia",
                      f"{Ixx} {Iyy} {Izz} {Ixy} {Ixz} {Iyz}")
        # geoms
        for g in node.geoms:
            self.compile_geom(g, body_el)
        # children
        for c in node.children:
            self.compile_body(c, body_el)

    def compile_joint(self, j, parent_el):
        if j.jtype_str == "free":
            ET.SubElement(parent_el, "freejoint")
            return
        je = ET.SubElement(parent_el, "joint")
        je.set("name", j.name)
        je.set("type", j.jtype_str)
        # self.set_vec3(je, "pos", j.pos)
        # self.set_quat(je, "quat", j.quat)
        self.set_vec3(je, "axis", j.ax)
        if j.range is not None:
            lo, hi = j.range
            je.set("range", f"{lo} {hi}")
        je.set("damping", str(j.damping))
        je.set("frictionloss", str(j.frictionloss))
        je.set("armature", str(j.armature))

    def compile_geom(self, g, parent_el):
        ge = ET.SubElement(parent_el, "geom")
        # ge.set("name", g.name) # name is optional for geom
        ge.set("type", g.gtype)
        # size depends on type
        if g.gtype == "sphere":
            ge.set("size", f"{g.size[0]}")
        elif g.gtype == "capsule":
            ge.set("size", f"{g.size[0]} {g.size[1]}")
        else:
            ge.set("size", f"{g.size[0]} {g.size[1]} {g.size[2]}")
        self.set_vec3(ge, "pos", g.pos)
        self.set_quat(ge, "quat", g.quat)
        if g.rgba:
            r = g.rgba
            ge.set("rgba", f"{r[0]} {r[1]} {r[2]} {r[3]}")
        if g.mesh_ref:
            ge.set("mesh", g.mesh_ref.name)
        if g.friction is not None:
            mu, torsion, rolling = g.friction
            ge.set("friction", f"{mu} {torsion} {rolling}")
        if g.contype is not None:
            ge.set("contype", str(int(g.contype)))
        if g.conaffinity is not None:
            ge.set("conaffinity", str(int(g.conaffinity)))

    def compile_actuator(self, a, parent_el):
        ae = ET.SubElement(parent_el, a.atype)
        ae.set("name", a.name)
        ae.set("joint", a.joint.name)
        if a.kp is not None:
            ae.set("kp", str(a.kp))
        if a.kv is not None:
            ae.set("kv", str(a.kv))
