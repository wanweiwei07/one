import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.scene.collision_shape as sco
import one.robots.base.mech_base as orbmb
import one.physics.inertial as opi
import one.physics.mj_nodes as opmno
import one.physics.mj_naming as opmna


class MJOneConverter:

    def __init__(self, margin=0.0):
        self._opt = opmno.OptionNode()
        self._opt.gravity = (0, 0, -9.81)
        self._opt.timestep = 0.002
        self._default = opmno.DefaultNode()
        # self._default.geom["friction"] = (1.0, 0.1, 0.1)
        self._default.geom["margin"] = margin
        self._default.geom["solref"] = (0.02, 1.0)
        self._default.geom["solimp"] = (0.9, 0.95, 0.002)
        self._mesh_assets = {}  # key = file_path, value = MeshAsset
        self._actuators = []  # list of ActuatorNode
        # mappings
        self._sobj2bdy = {}
        self._rutl2bdy = {}  # use runtime lnk as key
        self._mecj2jnt = {}  # use (mecba, jidx) as key
        self._bdy_alias = {}
        # mounted children
        self._mounted_children = set()

    def convert(self, scene):
        self._mesh_assets.clear()
        self._actuators.clear()
        self._sobj2bdy.clear()
        self._rutl2bdy.clear()
        self._mecj2jnt.clear()
        self._bdy_alias.clear()
        self._mounted_children.clear()
        for mecba in scene.mecbas:
            self._collect_mounted_children(mecba)
        world = opmno.WorldNode()
        world.option = self._opt
        world.default = self._default
        root = opmno.BodyNode("world_root")
        world.root_body = root
        for mecba in scene.mecbas:
            if mecba in self._mounted_children:
                continue
            robot = self._cvt_robot(mecba)
            self._merge_empty_geoms(robot, is_root=True)
            robot.parent = root
            root.children.append(robot)
        for sobj in scene.sobjs:
            if sobj in self._mounted_children:
                continue
            if not getattr(sobj, "collisions", None):
                continue
            body = self._cvt_sobj(sobj, ref_tf=sobj.tf)
            self._sobj2bdy[sobj] = body
            body.parent = root
            root.children.append(body)
        world.assets = list(self._mesh_assets.values())
        world.actuators = self._actuators
        self._finalize_alias_maps()
        # process collision ignores
        for mecba in scene.mecbas:
            struct = mecba.structure
            for alidx, blidx in struct.compiled.collision_ignores_idx:
                rta = mecba.runtime_lnks[alidx]
                rtb = mecba.runtime_lnks[blidx]
                body_a = self._rutl2bdy[rta]
                body_b = self._rutl2bdy[rtb]
                world.contact_excludes.append((body_a, body_b))
        return world, self._sobj2bdy, self._rutl2bdy, self._mecj2jnt

    @property
    def gravity(self):
        return self._opt.gravity

    @gravity.setter
    def gravity(self, value):
        self._opt.gravity[:] = value

    @property
    def timestep(self):
        return self._opt.timestep

    @timestep.setter
    def timestep(self, value):
        self._opt.timestep = value

    def _cvt_robot(self, mecba):
        compiled = mecba._compiled

        def _scan_child(mecba, jidx, lidx):
            # hosting joint frame
            jnode = opmno.JointNode(opmna.alloc_name("jnt"))
            self._mecj2jnt[(mecba, jidx)] = jnode
            jtype = compiled.jtypes_by_idx[jidx]
            if jtype == ouc.JntType.REVOLUTE:
                jnode.jtype_str = "hinge"
                act = opmno.ActuatorNode(opmna.alloc_name("ra"))
                act.joint = jnode
                self._actuators.append(act)
            elif jtype == ouc.JntType.PRISMATIC:
                jnode.jtype_str = "slide"
                act = opmno.ActuatorNode(opmna.alloc_name("sa"))
                act.joint = jnode
                self._actuators.append(act)
            else:
                jnode.jtype_str = "fixed"
            jnode.axis = tuple(compiled.jax_by_idx[jidx])
            jnode.range = (float(compiled.jlmt_low_by_idx[jidx]),
                           float(compiled.jlmt_high_by_idx[jidx]))
            # lnk
            lnk = mecba.runtime_lnks[lidx]
            if lnk.is_free:
                raise ValueError(
                    "Free link cannot be child link of a joint")
            jotfmat = compiled.jotfmat_by_idx[jidx]
            body = self._cvt_sobj(lnk, ref_tf=jotfmat)
            self._rutl2bdy[lnk] = body
            # attach joint to parent body
            body.hosting_jnts.append(jnode)
            # recurse into grandchildren
            for clidx in compiled.clnk_ids_of_lidx[lidx]:
                pjidx = compiled.pjidx_of_lidx[clidx]
                if pjidx >= 0:
                    child = _scan_child(mecba, pjidx, clidx)
                    child.parent = body
                    body.children.append(child)
            return body

        ridx = compiled.root_lnk_idx
        root_lnk = mecba.runtime_lnks[ridx]
        root = self._cvt_sobj(root_lnk, ref_tf=mecba.tf)
        self._rutl2bdy[root_lnk] = root
        for clidx in compiled.clnk_ids_of_lidx[ridx]:
            pjidx = compiled.pjidx_of_lidx[clidx]
            if pjidx >= 0:
                child = _scan_child(mecba, pjidx, clidx)
                child.parent = root
                root.children.append(child)
        self._attach_mountings(mecba)
        return root

    def _cvt_sobj(self, sobj, ref_tf=None):
        if ref_tf is None:
            ref_tf = np.eye(4, dtype=np.float32)
        b = opmno.BodyNode(opmna.alloc_name("sobj"))
        if sobj.is_free:
            jnode = opmno.JointNode("free_root")
            jnode.jtype_str = "free"
            b.hosting_jnts.append(jnode)
        b.pos, b.quat = oum.pos_quat_from_tf(ref_tf)
        if sobj.collisions:
            if (sobj.mass is not None and
                    sobj.com is not None and
                    sobj.inrtmat is not None):
                b.inertial = opmno.InertialNode(
                    mass=sobj.mass, com=sobj.com,
                    inertia=sobj.inrtmat)
            elif sobj.mass is not None:
                com, inrtmat = opi.inertia_from_collisions(
                    sobj.collisions, sobj.mass)
                b.inertial = opmno.InertialNode(
                    mass=sobj.mass, com=com,
                    inertia=inrtmat)
            for c in sobj.collisions:
                g = self._cvt_geom(
                    c, opmna.alloc_name("geom"))
                self._apply_collision_filter(sobj, g)
                b.geoms.append(g)
        return b

    def _cvt_geom(self, c, name=None):
        g = opmno.GeomNode(name)
        g.pos = c.pos
        g.quat = c.quat
        if isinstance(c, sco.SphereCollisionShape):
            g.gtype = "sphere"
            g.size = (c.radius,)
        elif isinstance(c, sco.CapsuleCollisionShape):
            g.gtype = "capsule"
            g.size = (c.radius, c.half_length)
        elif isinstance(c, (sco.AABBCollisionShape, sco.OBBCollisionShape)):
            g.gtype = "box"
            g.size = tuple(c.half_extents)
        elif isinstance(c, sco.PlaneCollisionShape):
            g.gtype = "plane"
            g.size = (1.0, 1.0, 0.1)
        elif isinstance(c, sco.MeshCollisionShape):
            g.gtype = "mesh"
            if c.file_path not in self._mesh_assets:
                name = f"mesh_{len(self._mesh_assets)}"
                self._mesh_assets[c.file_path] = opmno.MeshAsset(
                    name=name, path=c.file_path)
            g.mesh_ref = self._mesh_assets[c.file_path]
        else:
            raise NotImplementedError(f"Unsupported collision: {type(c)}")
        return g

    def _collect_mounted_children(self, mecba):
        for m in mecba._mountings.values():
            self._mounted_children.add(m.child)
            if isinstance(m.child, orbmb.MechBase):  # TODO type is not the standdard
                self._collect_mounted_children(m.child)

    def _attach_mountings(self, mecba):
        for m in mecba._mountings.values():
            child = m.child
            engage_tf = m.engage_tf
            # child subtree
            if isinstance(child, orbmb.MechBase):  # MechBase
                child_root = self._cvt_robot(child)
            else:  # SceneObject
                child_root = self._cvt_sobj(child)
                self._sobj2bdy[child] = child_root
            child_root.pos, child_root.quat = oum.pos_quat_from_tf(engage_tf)
            # find parent link body
            plnk_bdy = self._rutl2bdy[m.plnk]
            child_root.parent = plnk_bdy
            plnk_bdy.children.append(child_root)
            if isinstance(child, type(mecba)):
                self._attach_mountings(child)

    def _apply_collision_filter(self, sobj, geom):
        geom.contype = int(sobj.collision_group)
        geom.conaffinity = int(sobj.collision_affinity)

    def _merge_empty_geoms(self, body, is_root=False):
        for child in list(body.children):
            self._merge_empty_geoms(child, is_root=False)
        if is_root:
            return
        parent = body.parent
        if parent is None:
            return
        merge = 0
        if len(parent.geoms) == 0 and len(body.geoms) > 0:
            merge = 1
        if len(parent.geoms) == 0 and len(body.geoms) == 0:
            merge = 2
        if merge == 0:
            return
        ptfmat = oum.tf_from_quat_pos(parent.quat, parent.pos)
        ctfmat = oum.tf_from_quat_pos(body.quat, body.pos)
        newtfmat = ptfmat @ ctfmat
        parent.pos, parent.quat = oum.pos_quat_from_tf(newtfmat)
        if merge == 1:
            parent.geoms.extend(body.geoms)
            parent.inertial = body.inertial
        parent.hosting_jnts.extend(body.hosting_jnts)
        for gc in body.children:
            gc.parent = parent
            parent.children.append(gc)
        parent.children.remove(body)
        self._bdy_alias[body] = parent

    def _finalize_alias_maps(self):
        for k, b in list(self._sobj2bdy.items()):
            self._sobj2bdy[k] = self._resolve_body(b)
        for k, b in list(self._rutl2bdy.items()):
            self._rutl2bdy[k] = self._resolve_body(b)

    def _resolve_body(self, b):
        while b in self._bdy_alias:
            b = self._bdy_alias[b]
        return b
