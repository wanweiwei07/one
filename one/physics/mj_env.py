from one.physics.mj_one_cvter import MJOneConverter
from one.physics.mj_runtime import MJRuntime
from one.physics.mj_one_sync import MJSynchronizer


class MJEnv:
    def __init__(self, scene):
        self._cvter = MJOneConverter()
        self._world = self._cvter.convert(scene)
        self.xml_string = self._world.compile_mjcf()
        print(self.xml_string)
        self.runtime = MJRuntime(self.xml_string)
        self.sync = MJSynchronizer(self.runtime, scene)
        self.reset()

    def step(self, dt):
        self.runtime.exit_cd()
        h = self.runtime.model.opt.timestep
        n = int(round(dt / h))
        # self.sync.push_qpos()
        self.runtime.step(n)
        self.sync.pull_qpos()
        self.sync.pull_body_pose()

    def is_collided(self):
        self.runtime.enter_cd()
        collided = self.runtime.is_collided()
        self.runtime.exit_cd()
        return collided

    def reset(self):
        self.sync.push_qpos()
        self.runtime.forward()
        self.sync.pull_body_pose()

    def get_timestep(self):
        return float(self.runtime.model.opt.timestep)

    def save(self, filepath, encoding="utf-8"):
        if self.xml_string is None:
            raise RuntimeError("XML not built yet")
        with open(filepath, "w", encoding=encoding) as f:
            f.write(self.xml_string)

    @property
    def data(self):
        return self.runtime.data

    @property
    def model(self):
        return self.runtime.model

    @property
    def ctrl(self):
        return self.runtime.data.ctrl