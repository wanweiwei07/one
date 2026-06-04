# 重构计划：chain / tcp / ik 的统一

> 目标：消除 `ManipulatorBase` 与 humanoid 各搞一套的局面，把运动学能力统一成
> **「MechBase 拥有状态 + chain 选关节 + tcp 选定位点 + 一个 ik 动词」**。

---

## 1. 背景与动机

当前 `ManipulatorBase` 把「绑定在某条 chain 上的 IK + TCP 能力」硬编码成了一个**单 tip 子类**，带来两个问题：

1. **能力被困在子类里**：`L1`（humanoid，裸 `MechBase`）有 `left_arm_chain` 等多条链，却用不了 `ik_tcp`——因为它长在 `ManipulatorBase` 上且假设「只有一个 tip」。
2. **和 `AttachedFrame` 重复造轮子**：manipulator 的 TCP（`runtime_lnks[-1]` + `_loc_tcp_tf` + `_main_chain` + solver）和 `AttachedFrame`（`parent_lnk, loc_tf, chain, solver`）是同一个结构的两份实现，`ik_tcp` / `ik_tcp_nearest` / `ik_attached_frame` 三份重复的偏移换算。

根因：**「实例 / 状态」「结构 / 路径」「定位点 / 偏移」三件事被搅在了一起。**

这套结论与 ROS 主流一致：MoveIt `setFromIK(group, pose, tip_frame)` 把「动哪些关节(group/chain)」和「定位哪个点(tip_frame)」分开传；KDL `tree.getChain(base, tip)` + `wdls.setWeightJS` 是同一个 `ik(chain, …, free=mask)`。我们是在收敛到行业标准模型，而不是发明。

---

## 2. 目标架构（概念模型）

| 概念 | 是什么 | 归属 / 层级 | 数量 |
|---|---|---|---|
| `MechStruct` / `FlatMechStructure` | 蓝图：links + joints + 拓扑 | 类级、所有实例共享 | 一种机器人一份 |
| `KinematicChain` | **动哪些关节**：base→tip 路径 | 结构级、按 `(base,tip)` 缓存、无状态 | 一台机器人可多条 |
| `TCP`（frame） | **定位哪个点**：`(parent_lnk, loc_tf, name)` | **per-instance、运行时定义** | **一个 link 上可多个** |
| `MechBase.ik(chain, tcp, target)` | 求解（动词） | 在状态拥有者 `MechBase` 上 | —— |

三条铁律：

1. **状态只在 `MechBase`**（qs / 位姿 / runtime_lnks / FK / solvers）。chain 无状态、共享；tcp 只持 `(link, offset)`，不持 qs / chain / solver。
2. **chain 和 tcp 解耦（N:M）**：tcp 不绑死 chain，同一个 `thumb_tip` 可被 `finger_chain` 或 `arm_finger_chain` 驱动。
3. **tcp 不是 link**：它是 link 上的命名偏移，所以一个手掌 link 上能挂 `grasp_center` / `pre_grasp` / `approach` 等任意多个。

> **没有 `ManipulatorBase`、没有 humanoid 基类。** 所有机器人——机械臂 / 人形 / 机械手——都**直接继承 `MechBase`**，区别只是 `__init__` 里**定义了哪些 chain、哪些 tcp**。「机械臂」不是一个类型，只是「定义了一条 root→tip chain + 一个 flange tcp」的 MechBase。建 chain 本身就是 setup serial arm，不需要中间基类或辅助函数。

---

## 3. 核心决定

1. **消除 `ManipulatorBase`**（连同 humanoid 基类的念头）——机器人扁平继承 `MechBase`，类型差异下沉为 `__init__` 里的 chain/tcp 定义。
2. IK 动词上移到 `MechBase`，按 chain 参数化，合并现有三个 IK 方法。
3. IK 的 base 锚点 = `runtime_lnks[chain.base_lidx].tf`，**不再是 `self.rotmat/pos`**。这样 manipulator（base=根）和 humanoid 臂（base=内部 link）走同一套。
4. tcp 的偏移**绑定一次**，不当 ik 的 per-call 参数（消除「每次传偏移很奇怪」）。
5. ik **不假设 tcp 在 chain.tip 上**：chain 端点→tcp 的偏移用 **FK 现算**（对本次求解是刚体常量）。要求 tcp 在 chain 可动关节的**下游**，否则报错。
6. `AttachedFrame` 砍成 `TCP`：只剩 `(parent_lnk, loc_tf, name)` + `.tf`，把 chain / solver 踢出去。
7. 每条 chain 的 solver 仍按 `_init_solver(chain)` / `_solvers[chain]` 分派（主链解析解 `CVR038PencilIK`，其余数值解 `SELIKSolver`）。

---

## 4. 具体改动（按文件）

### 4.1 `kine/kinematic_chain.py`
- 现状已够用：`base_lidx` / `tip_lidx` / `active_jnt_ids` / `embed_active_qs` / `extract_active_qs` 都在。
- 无需改动（除非顺手暴露一个 `controls(lnk)` 判定：某 link 是否在本链下游，供 ik 做合法性检查）。

### 4.2 新增 `TCP`（替代 `attached_frame.py` 的 `AttachedFrame`）
```python
class TCP:
    def __init__(self, parent_lnk, loc_tf=None, name=None):
        self.parent_lnk = parent_lnk
        self.loc_tf = oum.ensure_tf(loc_tf)
        self.name = name
    @property
    def tf(self):           # 前向位姿 / 可视化顺带有了
        return self.parent_lnk.tf @ self.loc_tf
    # pos / rotmat / set_loc_* / copy 照搬旧 AttachedFrame
```
- 去掉 `chain` / `solver` 字段。
- 旧 `AttachedFrame` 调用点全部迁移到 `TCP`。

### 4.3 `base/mech_base.py`：新增 tcp 注册 + ik 动词
```python
# --- tcp 注册（manipulator 和 humanoid 共用）---
def add_tcp(self, name, parent_lnk, loc_tf=None):
    tcp = TCP(parent_lnk, loc_tf, name)
    self._tcps[name] = tcp
    return tcp

def tcp(self, name):
    return self._tcps[name]

# --- 唯一的 IK 动词 ---
def ik(self, chain, tcp, tgt_rotmat, tgt_pos,
       max_solutions=8, ref_qs=None, **kw):
    if isinstance(tcp, str):           # 名字 -> 本机构注册表
        tcp = self.tcp(tcp)
    base_tf = self.runtime_lnks[chain.base_lidx].tf
    tip_tf  = self.runtime_lnks[chain.tip_lidx].tf
    # chain 端点 -> tcp 的刚体偏移，用 FK 现算（本次求解内不变）
    T_tip2tcp = np.linalg.inv(tip_tf) @ tcp.tf
    tgt_tcp = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
    tgt_tip = tgt_tcp @ np.linalg.inv(T_tip2tcp)
    solver = self.get_solver(chain)
    results = solver.ik(
        root_rotmat=base_tf[:3, :3], root_pos=base_tf[:3, 3],
        tgt_rotmat=tgt_tip[:3, :3], tgt_pos=tgt_tip[:3, 3],
        max_solutions=max_solutions, ref_qs=ref_qs, **kw)
    return [chain.embed_active_qs(q, self.qs) for q in results]
```
- `__init__` 加 `self._tcps = {}`；`clone()` 复制 `_tcps`（parent_lnk 重映射到新 runtime link）。
- solver 惰性构建：`ik` 内用 `self._init_solver(chain)`（幂等；子类如 `CVR038` 重写它返回解析解）。
- **合法性检查做成 mounting-aware**（见下 §4.3.1）：tcp 在本机构树内 → 在自己树里向上走判是否下游于 `chain.tip`；tcp 在**挂上来的 child**（如夹爪）的 link 上 → 找到那个 mounting，判挂点 `m.plnk` 是否下游于 `chain.tip`；都不是 → 抛错「this chain does not control this tcp」。

#### 4.3.1 跨对象 tcp（hand 挂在 arm 上）——无需特殊桥
`ik` 里 `T_tip2tcp = inv(tip.tf) @ tcp.tf` 这条「FK 现算」规则**本身就覆盖跨对象**：hand 被 mount 后 `_update_mounting` 已调 `hand.fk()`，所以 `hand.tcp(...).tf` 是有效世界位姿；`inv(arm法兰.tf) @ hand_tcp.tf` 就是法兰→tcp 的刚体偏移，fixed mount 下对本次求解是常量。于是：

```python
arm.ik(arm.main_chain, hand.tcp('grasp_center'), rotmat, pos)   # 直接成立
```

- **每次调用重算偏移**，比「engage 时记死」更对（换夹爪 / 改 mount / 改 jaw 宽度自动跟上）。
- 不需要在 engage 里把 ee 的 tcp「抬升」成 arm 法兰 tcp（旧 `_loc_tcp_tf` 的活）——删掉这层。
- 唯一前提：tcp **刚性下游**于 `chain.tip`（fixed mount + tcp 在刚性 link 或其余关节冻结）。mounting-aware 检查负责拦住「chain 根本控制不了该 tcp」的静默错误。

### 4.4 消除 `manipulators/manipulator_base.py`
机械臂直接继承 `MechBase`，在 `__init__` 里定义 chain + tcp（+ 必要时解析解 solver）。建 chain 就是 setup serial arm，没有中间基类、没有辅助函数。样板（CVR038）：

```python
class CVR038(orbmb.MechBase):
    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos, is_free=False)
        c = self.structure.compiled
        self.main_chain = self.structure.get_chain(c.root_lnk, c.tip_lnks[0])  # chain 定义
        self.add_tcp('flange', self.runtime_lnks[-1])                          # tcp 定义
        self._init_solver(self.main_chain)

    def _init_solver(self, chain):
        if chain is getattr(self, 'main_chain', None):
            self._solvers[chain] = ormdci.CVR038PencilIK(chain, (chain.lmt_lo, chain.lmt_up))
            return self._solvers[chain]
        return super()._init_solver(chain)
```

调用统一：`arm.ik(arm.main_chain, 'flange', R, p)`。

`ManipulatorBase` 上旧成员的去处：

| 旧成员 | 去处 |
|---|---|
| `ik_tcp` / `ik_tcp_nearest` | 删；用 `ik(main_chain, 'flange', …)` |
| `_loc_flange_tf` / `_loc_tcp_tf` | 删；偏移进 tcp 的 `loc_tf` |
| `gl_tcp_tf` / `gl_flange_tf` | 删；用 `self.tcp('flange').tf` |
| `engage(ee, loc_tf)` | 用 `MechBase.mount(ee, flange_lnk, loc_tf)`；摆 ee 上的点靠 §4.3.1 跨对象 FK 现算 |
| `set_loc_tcp_rotmat_pos` / `reset_tcp` | 删；改 tcp 的 `loc_tf`（`set_loc_*`）|
| `toggle_tcp` / `toggle_attached_frames` | 可视化助手搬到 `MechBase`，按 tcp 名通用 |
| `define / get / ik_attached_frame`、`_attached_frames` | 删；用 tcp 注册表 + `ik(chain, tcp, …, axis_constraints=…)` |
| `mount` 被禁用的那套 | 删；`mount` 恢复正常使用 |

> 迁移期 `ManipulatorBase` 暂留作兼容（§7.2 的转发版本），其余 arm 全部转完后再删整个文件。

### 4.5 `humanoids/linx/l1/l1.py`：本来就是 MechBase，无需改继承
- 在 `__init__` 里定义好 `left_arm_chain` 等 + `add_tcp('left_eye', head_lnk, …)` / `add_tcp('left_palm', …)`，即可 `robot.ik(left_arm_chain, 'left_palm', pose)`。和机械臂、机械手**同一套调用**。
- **手剥离（已做）**：原 h0602.urdf 把两只 LinkerHand O6 灵巧手（各 11 指关节）硬编码进 body。已切成独立 EE `end_effectors/linkerhand/o6`（`o6_left.urdf`/`o6_right.urdf` + `O6Left`/`O6Right` 两个 MechBase），body URDF 砍到 `*_arm_link_6` 收尾。L1 `__init__` 用 `mount(O6Left(), lnk('left_arm_link_6'), z=0.034)` 把手挂回去——位姿与原一体 URDF **逐点一致（diff=0）**。每只手带两个 center tcp：`power_grasp_center`（全手握爪中心）/ `pinch_center`（拇指-食指捏取中心），抓取走跨对象 ik `robot.ik('left_arm', robot.left_hand.tcp('power_grasp_center'), R, p)`。指链 ik 暂未建模（手当刚体 EE 用）。一次性迁移脚本在 `tools/extract_o6_hands.py` / `tools/strip_o6_hands.py`。

### 4.6 消除 `end_effectors/ee_base.py` 的 `EndEffectorBase`（同 ManipulatorBase）
- `EndEffectorBase` 也是「薄 MechBase 子类 + 一个 TCP」→ TCP 进 tcp 注册表，类删掉，gripper 直接继承 `MechBase`。
- **`GripperMixin` / `PointMixin` 保留**：它们是**行为**（open/close/grasp/release/grip_at/set_jaw_width；touch/attach），可复用、属组合。臂的特殊行为(IK)泛化进了 MechBase 故无需类；夹爪行为不泛化，故留 mixin。
- **平行夹爪不定义 chain、不用 ik**：运动是 jaw 开合，`set_jaw_width` 直接设关节 → fk。chain/ik 只给灵巧手(要解指尖位姿)用。
- tcp 命名约定 **`'grasp_center'`**（GripperMixin 按此查）；`grip_at` 用 `self.tcp('grasp_center').loc_tf`，带兼容回退 `loc_tcp_tf`。
- `contact_pattern`（grasp 规划要用）→ 移到 gripper `__init__` 的属性 + `clone` 复制。

样板 CVR038Gripper（已做）：
```python
class CVR038Gripper(orbmb.MechBase, oreb.GripperMixin):
    def __init__(self):
        super().__init__()                                 # is_free=True
        self.add_tcp('grasp_center', self.runtime_root_lnk, loc_offset)
        self.contact_pattern = np.zeros((1, 3), np.float32)
        self.jaw_range = ...; self.set_jaw_width(...)
    def set_jaw_width(self, w): self.fk(qs=[w * 0.5])      # 无 ik
```

---

## 5. 调用示例（改造后）

```python
# __init__ 里：定义 chain（动哪些关节）和 tcp（定位哪个点），都按名注册
robot.add_chain('main', root_lnk, wrist_lnk)        # chain：结构级、共享
robot.add_tcp('flange', wrist_lnk)                  # tcp：偏移为 0
hand.add_chain('thumb', palm_lnk, thumb_last_lnk)
hand.add_tcp('grasp_center', palm_lnk, offset_c)    # 同一 palm link
hand.add_tcp('pre_grasp',    palm_lnk, offset_p)    #   上挂多个 tcp
hand.add_tcp('thumb_tip',    thumb_last_lnk, off_t)

# ik 把「动哪些关节」和「定位哪个点」拼起来——同机构两者都用名字
robot.ik('main',  'flange',    rotmat, pos)              # 动 arm
hand.ik('thumb', 'thumb_tip', rotmat, pos)              # 动手指（手腕保持）

# 跨对象：hand 挂在 arm 上，grasp_center 在 hand 的注册表 → 传 TCP 对象
arm.mount(hand, arm.runtime_lnks[-1], loc_tf)
arm.ik('main', hand.tcp('grasp_center'), rotmat, pos)   # FK 现算偏移，无需桥
```

> **单目标永远只挑一条 chain**＝你想动的那组关节。"用 arm 摆 hand" 动 arm 链；"动手指" 动手指链。不存在"一个 ik 同时动 arm 和手指"——那是冗余、也跨不了两个独立结构。

**按名字索引**：定义用 `add_chain(name,…)` / `add_tcp(name,…)`。`ik(chain, tcp, …)` 两个参数都接受**名字字符串**（对本机构注册表解析）或**对象**（chain 用 `get_chain` 结果；tcp 用跨对象/一次性目标的对象）。
- chain 与 tcp 注册表的**本质区别**：chain 是 structure 级、共享，clone 直接复制引用、无需重映射、天然 clone-safe；tcp 是 per-instance，clone 要把 parent_lnk 重映射到克隆的 runtime link。字符串索引对 tcp 是必需，对 chain 只是为了调用对称。
- 拼错 = 运行期 KeyError，好查。

---

## 6. 暂不做 / 后续（明确划出范围，别误当本次任务）

| 项 | 说明 | 触发条件 |
|---|---|---|
| **free 关节掩码** | `ik(chain, tcp, …, free=subset)`：路径上一部分关节冻结（「动腿定位眼睛」「腰固定 vs 腰参与」）。对应 KDL `wdls.setWeightJS` / bio_ik active-set。仅数值解支持。 | 出现「只动路径子集」的需求时再加 |
| **ik_multi（多同时目标）** | `ik_multi([(chain,tcp,pose), …])`：**多个同时目标且共享可动自由度**、无法拆成独立单链 ik 时才需要——典型是**手腕也放开**的协调抓取（动手腕同时影响所有指尖）。注意：手腕固定、手指互相独立时，N 个指尖就是 N 个**独立** ik，不需要它。与"arm+手指"无关。 | 出现"多目标+共享自由度"需求时再加 |
| **偏 IK（位置/轴约束）** | 把旧 `ik_partial` 的 `axis_constraints` / 位置-only 作为 `ik` 的参数保下来。 | 迁移 `ik_attached_frame` 时一并处理 |
| **闭环 / 并联** | 双脚同时踩地等。URDF 本身都表达不了（严格树），超出开链表示范围，需约束式求解（pinocchio/Drake/QP）。 | 不在本架构范围 |

---

## 7. 迁移与兼容

- [x] **1.** 加 `TCP` + `MechBase.add_tcp/tcp/ik`（mounting-aware 合法性、clone 重映射），不删旧路径。验证：同机构 round-trip、下游报错、跨对象、L1 构造、clone。
- [x] **2.** `ManipulatorBase.ik_tcp*` 转发到 `MechBase.ik`（兼容桥，验证 `MechBase.ik` 可用）。
- [x] **3.** **CVR038 样板**：直接继承 `MechBase`，`main_chain`（property）+ `add_tcp('flange')` + `_init_solver` 解析解；`cvr038_with_gripper` 用 `mount`，example 用 `tcp('flange').tf`。验证：ik / clone / gripper mount / 跨对象 全过。
- [x] **3b.** `KineVisualizer`：`mode` 字符串 → `chain` 参数（给则画该链、不给画整机），去掉 `tip_lnks[0]` 猜测；3 个调用方更新。
- [ ] **4.** 把其余 arm（ur3/ur3e/fr3/lite6/rs007l/crx5ia/openarm）按样板逐个转继承 `MechBase`，更新各自调用方/demo。
- [x] **5a.** 部分 IK：**分开写**——`ik`（全位姿，不变）+ 新 `MechBase.ik_partial(chain, tcp, tgt_pos=, axis_constraints=, …)`（数值，`solver.ik_partial`）。`ik`/`ik_partial` 统一返回 **qs 列表（空=无解）**，去掉 None。`KinematicChain.name`（`add_chain` 写入）→ SELIK 数据目录用链名（`data/chain_to_j5`）。
- [x] **5b.** 可视化助手 → `MechBase.toggle_tcp('name', …)`（按 tcp 名通用，替代旧 `toggle_tcp`+`toggle_attached_frames`）。
- [x] **5c.** 迁 coripps：`cvr038_4tb_gripper`(转 `MechBase+GripperMixin`)、`tube_picker`(`engage`→`mount`、`chain('main')`、`gripper.toggle_tcp('grasp_center')`)、`leaf_sampler`(`mount` d405 + `add_chain('to_j5')` + `add_tcp('d405')` + `ik_partial`)、`main.py`(`ik`/`ik_partial`/`toggle_tcp`)、`mobisys.py`。验证全过:构造 + tube_picker 跨对象 ik + leaf_sampler **部分 IK 数值解**(从 `data/chain_to_j5` 加载,pos_err≈1e-4、z_err≈0.03°，相机对准目标)。
- [ ] **6.** 全部 arm 迁完后，**删除 `manipulator_base.py` 整个文件** 与 `attached_frame.py`。
- [x] **7.** `L1`：`__init__` 注册 5 条命名 chain（left_arm/right_arm/left_arm_waist/right_arm_waist/neck）+ 2 个 tcp（left_tcp/right_tcp）；删掉遮蔽 `MechBase.chain(name)` 的 2 参 `chain` 方法和链属性。`__main__` 用 `toggle_tcp('left_tcp')`。验证：注册表 / `chain('left_arm')` 解析 / `_chain_controls_tcp` / clone 全过 → arm/人形/手 `ik('chain','tcp',pose)` 同一套。

### EE / 夹爪侧（与 arm 侧并行，见 §4.6）
- [x] **E1.** `GripperMixin.grip_at` 改用 `_grasp_loc_tf()`：有 `'grasp_center'` tcp 用注册表，否则回退 `loc_tcp_tf`（兼容旧夹爪）。
- [x] **E2.** **CVR038Gripper 样板**：`MechBase + GripperMixin`，`add_tcp('grasp_center')` + `contact_pattern` + `set_jaw_width`（无 chain/ik）。验证：jaw / grip_at / clone / 跨对象 ik 全过。
- [ ] **E3.** 其余夹爪（2fg7/robotiq/fr3_gripper/oag/or_sd[PointMixin]）逐个转 `MechBase + Mixin`，注册 `'grasp_center'`，更新各自 examples 的 `gl_tcp_tf`→`tcp('grasp_center').tf`。
- [ ] **E4.** 通用 grasp 代码 `grasp/antipodal.py`、`polypodal.py` 里的 `gl_tcp_tf` 调用随夹爪迁移一并改。
- [ ] **E5.** 全部夹爪转完后删 `EndEffectorBase`（`ee_base.py` 保留 `GripperMixin`/`PointMixin`），去掉 `_grasp_loc_tf` 的兼容回退。

> 迁移期 `ManipulatorBase` 暂留（步骤 2 的转发版），步骤 6 才删，保证中途可运行。

---

## 8. 一句话总结

**没有 `ManipulatorBase`。所有机器人扁平继承 `MechBase`，在 `__init__` 里定义 chain（动哪些关节）和 tcp（定位哪个点）。`MechBase.ik(chain, tcp, target)` 把两者拼起来、用 FK 现算两者间的刚体偏移。机械臂 / 人形 / 机械手是同一套，区别只在装配了哪些 chain 和 tcp。**
