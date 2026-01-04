# 假设你有一个 KinematicChain 实例 chain
chain = my_chain  # 这里是你的对象
low = chain.lmt_low
high = chain.lmt_up

# 1) 构造状态空间
space = RealVectorStateSpace(low=low, high=high)

# 2) 定义状态有效性检查（调用你的 FK + 碰撞检测）
def is_state_valid(q: np.ndarray) -> bool:
    # TODO: 换成你自己的接口
    # 比如：
    #   - chain.set_q(q)
    #   - 更新 robot model 的姿态
    #   - 调用 FCL/MuJoCo / 你自己的 CD 系统
    #   - 返回是否无碰撞
    #
    # 这里只给个占位实现：
    return True  # 临时：先当全自由无碰撞，方便调试路径拓扑

si = SpaceInformation(space, is_state_valid, max_edge_step=0.05)

# 3) 构建问题定义
q_start = np.zeros_like(low)          # 例：全 0 开始
q_goal = (low + high) * 0.5           # 例：中间位置做个测试
pdef = ProblemDefinition(si, q_start, q_goal, goal_tolerance=0.05)

# 4) 建立 planner 并求解
planner = RRTConnectPlanner(
    pdef,
    step_size=0.2,
    goal_bias=0.1,
)

path = planner.solve(max_iters=5000, time_limit=2.0, verbose=True)

if path is None:
    print("No path found")
else:
    print("Path has", len(path), "waypoints")
    # 这里 path 是 List[np.ndarray]，你可以逐点插值执行，或者再做时间参数化
