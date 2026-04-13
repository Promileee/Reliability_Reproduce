import numpy as np
from scipy.optimize import differential_evolution
import multiprocessing as mp
import time

# 注意：必须在工作进程内部导入环境，避免跨进程共享实例导致内存损坏或随机种子冲突
from environment import UAVEnvironment

# ==========================================
# 1. 启发式策略 (提取为顶层函数)
# ==========================================
def policy_1(state, config, ra0):
    X = state[:config.K]
    ratios = X / config.L_THRESHOLDS
    if np.any(ratios > ra0):
        return 0 # DA (Abort)
    return 1     # DC (Continue)

def policy_2(state, config, custom_thresholds):
    X = state[:config.K]
    if np.any(X > custom_thresholds):
        return 0 # DA
    return 1     # DC

def policy_3(state, config):
    return 1

# ==========================================
# 2. 蒙特卡洛评估核心 (供各进程调用)
# ==========================================
def evaluate_policy_core(env, policy_fn, num_episodes):
    """底层的仿真循环 (引入 CRN 技术消除运气偏差)"""
    total_rewards = []
    
    for ep in range(num_episodes):
        # 【核心魔法】：公共随机数 (CRN)
        # 强制这 2000 次或者 5000 次的每一局，都使用固定的初始种子
        # 保证参数 A 和参数 B 面临的退化轨迹 100% 一模一样
        np.random.seed(10000 + ep)
        
        state = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action = policy_fn(state, env.config)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
        total_rewards.append(episode_reward)
        
    # 评估结束后，解除种子锁定，防止污染外部其他的随机过程
    np.random.seed(None)
    
    return np.mean(total_rewards)

# ==========================================
# 3. 供差分进化 (DE) 调用的顶层目标函数
# ==========================================
# 这些函数会被派发到不同的 CPU 核心上执行
def objective_p1(x, num_episodes):
    local_env = UAVEnvironment()
    ra0 = x[0]
    def p1_wrapper(s, c): return policy_1(s, c, ra0)
    return -evaluate_policy_core(local_env, p1_wrapper, num_episodes)

def objective_p2(x, num_episodes):
    local_env = UAVEnvironment()
    def p2_wrapper(s, c): return policy_2(s, c, custom_thresholds=x)
    return -evaluate_policy_core(local_env, p2_wrapper, num_episodes)

# ==========================================
# 4. 优化器与主程序
# ==========================================
if __name__ == "__main__":
    # Windows 环境下，多进程代码必须放在 __main__ 保护块内！
    cores_to_use = max(1, mp.cpu_count() - 10) 
    print(f"==================================================")
    print(f"🚀 启动并行蒙特卡洛寻优计算引擎 (CRN增强版)")
    print(f"🖥️  检测到 CPU 核心数: {mp.cpu_count()} | 已分配工作进程: {cores_to_use}")
    print(f"==================================================\n")

    env_main = UAVEnvironment()
    
    # 进化阶段参数 (由于引入了 CRN 消除了方差，这里其实降到 500 次也能得到极准的结果，
    # 但保留 2000 次会让基准线更加稳如泰山)
    evolve_episodes = 200 
    final_eval_episodes = 300 

    # ------------------------------------------------
    # Policy 1: 并行寻优
    # ------------------------------------------------
    print("⏳ 正在跨核心并行优化 Policy 1 (固定比例阈值)...")
    start_time = time.time()
    
    bounds_p1 = [(0.5, 1.5)]
    res_p1 = differential_evolution(
        objective_p1, 
        bounds_p1, 
        args=(evolve_episodes,),
        # strategy='best1bin',
        strategy='randtobest1bin',
        maxiter=15, 
        popsize=30, 
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=True,
        workers=cores_to_use,    
        updating='deferred',      
        polish=False             
    )
    
    best_ra0 = res_p1.x[0]
    print(f"✅ Policy 1 寻优完成 (耗时: {time.time()-start_time:.1f}s) -> 最优比例: {best_ra0:.4f}\n")

    # ------------------------------------------------
    # Policy 2: 并行寻优 
    # ------------------------------------------------
    print("⏳ 正在跨核心并行优化 Policy 2 (多维独立阈值)...")
    start_time = time.time()
    
    L = env_main.config.L_THRESHOLDS
    bounds_p2 = [
        (L[0] * 0.5, L[0] * 1.0),
        (L[1] * 0.5, L[1] * 1.5),
        (L[2] * 0.5, L[2] * 1.5)
    ]
    
    res_p2 = differential_evolution(
        objective_p2, 
        bounds_p2, 
        args=(evolve_episodes,),
        maxiter=20, 
        popsize=15, 
        disp=True,
        workers=cores_to_use,    
        updating='deferred',     
        polish=False             
    )
    
    best_th = res_p2.x
    print(f"✅ Policy 2 寻优完成 (耗时: {time.time()-start_time:.1f}s) -> 最优多维阈值: {np.round(best_th, 3)}\n")

    # ------------------------------------------------
    # 大样本量最终评估 (跑满 5000 次)
    # ------------------------------------------------
    print("==================================================")
    print("📊 进入大样本量终极评估阶段 (5000 次)...")
    # 这里的 5000 次评估也会使用相同的种子池 (10000 到 14999)，
    # 这意味着 Policy 1, 2, 3 将在绝对公平的 5000 套考卷上进行终极对比！
    
    def final_p1_wrapper(s, c): return policy_1(s, c, best_ra0)
    def final_p2_wrapper(s, c): return policy_2(s, c, best_th)

    mean_1 = evaluate_policy_core(env_main, final_p1_wrapper, final_eval_episodes)
    mean_2 = evaluate_policy_core(env_main, final_p2_wrapper, final_eval_episodes)
    mean_3 = evaluate_policy_core(env_main, policy_3, final_eval_episodes)

    print("==================================================")
    print("🏆 终极并行评估结果对比 (基于公共随机数):")
    print(f"Policy 2 (多维独立阈值最优解): {mean_2:.2f}")
    print(f"Policy 1 (固定比例阈值最优解): {mean_1:.2f}")
    print(f"Policy 3 (不实施中止策略)    : {mean_3:.2f}")
    print("==================================================")