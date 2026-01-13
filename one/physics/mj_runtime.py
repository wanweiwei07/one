import mujoco


class MJRuntime:
    def __init__(self, xml_string):
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        self._cd_mode = False
        self._dyn_backup = None

    def step(self, substeps=1):
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def enter_cd(self):
        if self._cd_mode:
            return
        self._backup()
        self._cd_mode = True

    def exit_cd(self):
        if not self._cd_mode:
            return
        self._restore()
        self._cd_mode = False

    def is_collided(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        # import one.physics.mj_contact as mjc
        # mjc.debug_contacts(self)
        return self.data.ncon > 0

    def _backup(self):
        self._dyn_backup = dict(qpos=self.data.qpos.copy(),
                                qvel=self.data.qvel.copy(),
                                act=self.data.act.copy(),
                                ctrl=self.data.ctrl.copy(),
                                time=self.data.time, )

    def _restore(self):
        b = self._dyn_backup
        self.data.qpos[:] = b["qpos"]
        self.data.qvel[:] = b["qvel"]
        self.data.act[:] = b["act"]
        self.data.ctrl[:] = b["ctrl"]
        self.data.time = b["time"]
        self._dyn_backup = None
        mujoco.mj_forward(self.model, self.data)