import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("collided.xml")
data = mujoco.MjData(model)
model.opt.gravity=[0,0,0]
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()