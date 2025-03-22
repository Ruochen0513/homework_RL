import numpy as np
import time
# 参数设置
OBSTACLES = {(1,1), (1,2), (2,2), (3,1), (3,3), (4,1)}
TARGET = (3,2)
ACTIONS = {'↑': (-1,0), '↓': (1,0), '←': (0,-1), '→': (0,1), '·': (0,0)}
GAMMA = 0.9
THETA = 1e-3
K_EVAL_MAX = 1  # 策略评估阶段的最大迭代次数（可调整截断次数）

# 初始化价值函数和策略
V = np.zeros((5,5))
policy = np.full((5,5), '·', dtype='U2')  # 初始策略为不动

def get_reward(next_state):
    if next_state == TARGET:
        return 1
    if next_state in OBSTACLES:
        return -10
    if not (0 <= next_state[0] <5 and 0 <= next_state[1] <5):
        return -1
    return 0

def valid_transition(s, a):
    ni, nj = s[0]+a[0], s[1]+a[1]
    if (ni, nj) in OBSTACLES or not (0 <= ni <5 and 0 <= nj <5):
        return s
    return (ni, nj)

# 截断策略迭代主循环
start = time.time()
max_policy_iters = 100000
iter_num = 0
for _ in range(max_policy_iters):
    iter_num += 1
    # 策略评估（截断迭代）
    eval_iter = 0
    while True:
        delta = 0
        new_V = V.copy()
        for i in range(5):
            for j in range(5):
                if (i,j) in OBSTACLES or (i,j) == TARGET:
                    continue
                a = ACTIONS[policy[i,j]]
                ns = valid_transition((i,j), a)
                reward = get_reward(ns)
                new_V[i,j] = reward + GAMMA * V[ns]
                delta = max(delta, abs(new_V[i,j] - V[i,j]))
        V[:] = new_V
        eval_iter += 1
        if delta < THETA or eval_iter >= K_EVAL_MAX:
            break
    
    # 策略改进
    policy_stable = True
    new_policy = np.empty_like(policy)
    for i in range(5):
        for j in range(5):
            if (i,j) in OBSTACLES or (i,j) == TARGET:
                new_policy[i,j] = policy[i,j]
                continue
            best_q = -np.inf
            best_a = '·'
            for a_name, a in ACTIONS.items():
                ns = valid_transition((i,j), a)
                reward = get_reward(ns)
                q = reward + GAMMA * V[ns]
                if q > best_q or (q == best_q and a_name == '·'):
                    best_q = q
                    best_a = a_name
            new_policy[i,j] = best_a
            if policy[i,j] != best_a:
                policy_stable = False
    if policy_stable:
        break
    policy = new_policy
end = time.time()
used_time = end - start
# 打印结果
print("优化后的状态值：")
print(np.round(V, 2))
print("\n迭代次数：", iter_num)
print("\n策略评估迭代次数：", K_EVAL_MAX)
print("\n所用时间：", used_time, "s")
print("\n最优策略矩阵：")
for i, row in enumerate(policy):
    print(f"{5-i} ", '  '.join([f"{x:^2}" for x in row]))
print("   1   2   3   4   5")