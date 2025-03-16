import numpy as np

# 坐标转换说明：图片中的坐标(行,列)转换为0-based索引
OBSTACLES = {(1,1), (1,2), (2,2), (3,1), (3,3), (4,1)}  # 转换后的障碍物坐标
TARGET = (3,2)  # 转换后的目标点坐标(原图4,3)
ACTIONS = {'↑':(-1,0), '↓':(1,0), '←':(0,-1), '→':(0,1), '·':(0,0)}
GAMMA = 0.9
THETA = 1e-4

# 初始化价值函数（障碍物保持0值）
V = np.zeros((5,5))

def get_reward(next_state):
    """计算状态转移奖励"""
    if next_state == TARGET:
        return 1
    if next_state in OBSTACLES or \
       not (0 <= next_state[0] <5 and 0 <= next_state[1] <5):
        return -1
    return 0

def valid_transition(s, a):
    """处理状态转移逻辑""" 
    ni, nj = s[0]+a[0], s[1]+a[1]
    
    # 碰撞检测（包含边界和障碍物）
    if (ni, nj) in OBSTACLES or not (0<=ni<5 and 0<=nj<5):
        return s  # 保持原状态
    return (ni, nj)

# 值迭代核心
for _ in range(1000):
    delta = 0
    new_V = V.copy()
    
    for i in range(5):
        for j in range(5):
            
            max_q = -np.inf
            for a in ACTIONS.values():
                ns = valid_transition((i,j), a)
                reward = get_reward(ns)
                q = reward + GAMMA * V[ns]
                max_q = max(max_q, q)
            
            new_V[i,j] = max_q
            delta = max(delta, abs(new_V[i,j]-V[i,j]))
    
    V = new_V
    if delta < THETA:
        break

# 策略可视化
policy = np.empty((5,5), dtype='U2')
for i in range(5):
    for j in range(5):
        best_q = -np.inf
        best_a = '·'
        for a_name, a in ACTIONS.items():
            ns = valid_transition((i,j), a)
            q = get_reward(ns) + GAMMA * V[ns]
            if q > best_q or (q == best_q and a_name == '·'):
                best_q = q
                best_a = a_name
        policy[i,j] = best_a

# 打印结果（与图片数值对齐）
print("优化后的状态值：")
print(np.round(V, 2))

print("\n最优策略矩阵：")
for i,row in enumerate(policy):
    print(f"{5-i} ", '  '.join([f"{x:^2}" for x in row]))  # 保持图片的行号方向
print("   1   2   3   4   5")