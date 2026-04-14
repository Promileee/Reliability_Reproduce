import numpy as np
from config import Config
import scipy.stats as stats

class UAVEnvironment:
    """
    无人机动态任务中止决策环境 (基于文献 Section 2 & 3 构建)
    """
    def __init__(self):
        self.config = Config()
        
        # 预计算带有依赖关系的等效漂移(mu)和扩散系数(sigma)
        # 根据公式 (4)
        self.mu_tilde = np.zeros(self.config.K)
        self.sigma_tilde = np.zeros(self.config.K)
        
        for i in range(self.config.K):
            # 等效漂移 mu_tilde_i = mu_i + sum(H_ij * mu_j)
            self.mu_tilde[i] = self.config.MU[i] + np.sum(self.config.H_MATRIX[i] * self.config.MU)
            
            # 等效方差 sigma_tilde_i^2 = sigma_i^2 + sum(H_ij^2 * sigma_j^2)
            var_tilde = self.config.SIGMA[i]**2 + np.sum((self.config.H_MATRIX[i]**2) * (self.config.SIGMA**2))
            self.sigma_tilde[i] = np.sqrt(var_tilde)
            
        self.max_steps = int(np.ceil(self.config.T1 / self.config.DELTA)) - 1 # 任务执行阶段的最大检测步数 (N)
        
        # 状态变量初始化
        self.current_X = np.zeros(self.config.K) # [X_E11, X_E21, X_E22]
        self.current_n = 0                       # 当前决策步数
        
        self.is_mission_payload_failed = False
        self.is_system_failed = False

    def reset(self):
        """重置环境到初始状态"""
        self.current_X = np.zeros(self.config.K)
        self.current_n = 0
        self.is_mission_payload_failed = False
        self.is_system_failed = False
        return self._get_state()

    def _get_state(self):
        """获取当前状态 S_n = (X_n^1, X_n^2, n)"""
        return np.concatenate([self.current_X, [self.current_n]])
    
    def _phi(self, t):
        """
        返航/救援时间函数 phi(t)
        根据文献 Section 5.1: phi(t) = t 如果 t <= 3, 否则为 3
        """
        return min(t, self.config.T2)

    
    def _calc_ig_cdf(self, delta_t, mu_tilde, sigma_tilde, L, x):
        """
        数值稳定版的逆高斯分布 CDF 计算 (公式 5)
        防止 np.exp(term2) 在退化量较小时发生溢出 (Overflow)
        """
        if x >= L:
            return 1.0
        if delta_t <= 0:
            return 0.0

        # 计算标准项
        term1 = (mu_tilde * delta_t - L + x) / (sigma_tilde * np.sqrt(delta_t))
        term2 = (2 * (L - x) * mu_tilde) / (sigma_tilde**2)
        term3 = -(mu_tilde * delta_t + L - x) / (sigma_tilde * np.sqrt(delta_t))
    
        # 获取正态分布 CDF 值
        cdf1 = stats.norm.cdf(term1)
        cdf3 = stats.norm.cdf(term3)
    
        # 核心修正逻辑：
        # 如果 term2 很大，说明失效概率极低，此时 exp(term2) * cdf3 往往趋近于 0
        # 我们通过对数空间计算来规避溢出：exp(term2) * cdf3 = exp(term2 + log(cdf3))
        if term2 > 700: # 接近 float64 极限
            log_cdf3 = stats.norm.logcdf(term3)
            combined_term = np.exp(term2 + log_cdf3)
        else:
            combined_term = np.exp(term2) * cdf3
        
        prob = cdf1 + combined_term
    
        # 最终防御：确保不会因为数值波动产生 NaN 或溢出 1.0
        if np.isnan(prob):
            return 0.0 if x < L else 1.0
        return np.clip(prob, 0.0, 1.0)
    
    def _calc_failure_probs(self, delta_t):
        """
        退回到论文公式(10)-(13)的原始逻辑：
        按照子系统内部的部件之间结构关系处理
        """
        p_fail_E11 = self._calc_ig_cdf(delta_t, self.mu_tilde[0], self.sigma_tilde[0], self.config.L_THRESHOLDS[0], self.current_X[0])
        p_fail_E21 = self._calc_ig_cdf(delta_t, self.mu_tilde[1], self.sigma_tilde[1], self.config.L_THRESHOLDS[1], self.current_X[1])
        p_fail_E22 = self._calc_ig_cdf(delta_t, self.mu_tilde[2], self.sigma_tilde[2], self.config.L_THRESHOLDS[2], self.current_X[2])
        
        p_payload_fail = p_fail_E11

        # 完全按照原逻辑处理
        p_transport_fail = p_fail_E21 * p_fail_E22 
        
        return p_payload_fail, p_transport_fail


    def step(self, action):
        """
        执行动作，推进环境状态。
        action: 0 (DA, Abort), 1 (DC, Continue)
        """
        done = False
        reward = 0.0
        
        current_time = self.current_n * self.config.DELTA
        rescue_time = self._phi(current_time)
        
        if action == 0: # DA: 放弃任务 (Abort)
            # 放弃任务，执行返航/救援。
            # 首先计算救援过程中运输子系统是否会失效
            _, p_sys_fail_rescue = self._calc_failure_probs(rescue_time)
            
            # 使用期望奖励以降低方差，根据公式(16): Q(S_n, DA) = -c_m - c_f * P_rs
            reward = -self.config.C_M - self.config.C_F * p_sys_fail_rescue
            done = True
            
        elif action == 1: # DC: 继续任务 (Continue)
            # 1. 扣除单次检测成本
            reward -= self.config.C_I
            
            # 计算下一次检测间隔(delta)内的失效概率
            p_payload_fail, p_sys_fail_delta = self._calc_failure_probs(self.config.DELTA)
            
            # ====== 基于论文公式 (17) 的概率期望奖励 ======
            # a. 计算若载荷失效引发强制救援，在救援过程中的系统失效概率 (公式 17 中的 Prs)
            next_rescue_time = self._phi((self.current_n + 1) * self.config.DELTA)
            _, p_sys_fail_rescue_after_payload_fail = self._calc_failure_probs(next_rescue_time)
            
            # b. 系统在下个 delta_t 内直接失效的期望惩罚
            reward -= p_sys_fail_delta * (self.config.C_F + self.config.C_M)
            
            # c. 载荷在下个 delta_t 内失效但系统幸存，触发强制救援的期望惩罚
            reward -= (1 - p_sys_fail_delta) * p_payload_fail * (
                self.config.C_M + self.config.C_F * p_sys_fail_rescue_after_payload_fail
            )
            # ==========================================================
            
            # 2. 随机模拟组件退化状态 (仅用于推进 MDP 的物理状态 S_n，不触发离散奖励)
            for i in range(self.config.K):
                # 增量服从正态分布
                delta_x = np.random.normal(self.mu_tilde[i] * self.config.DELTA, 
                                           self.sigma_tilde[i] * np.sqrt(self.config.DELTA))
                self.current_X[i] += delta_x 
            
            # 3. 判定物理状态是否进入吸收态以终结回合 (仅改变 done)
            if self.current_X[0] >= self.config.L_THRESHOLDS[0]:
                self.is_mission_payload_failed = True
                done = True
                
            if self.current_X[1] >= self.config.L_THRESHOLDS[1] and self.current_X[2] >= self.config.L_THRESHOLDS[2]:
                 self.is_system_failed = True
                 done = True

            # 4. 如果没有由于越界结束，推进时间并判断是否到达最后阶段
            if not done:
                self.current_n += 1
                
                # 检查是否完成了任务执行阶段 (到达 N步)
                if self.current_n >= self.max_steps:
                    # 对应论文公式 (20)：到达最后一步，基于概率结算任务成功的期望收益与返航阶段的风险
                    prob_success = (1 - p_sys_fail_delta) * (1 - p_payload_fail)
                    _, p_sys_fail_return = self._calc_failure_probs(self.config.T2)
                    
                    # 加上任务成功的期望奖励，扣除返航时的期望坠毁惩罚
                    reward += prob_success * (self.config.R_M - self.config.C_F * p_sys_fail_return)
                    
                    done = True

        return self._get_state(), reward, done, {"payload_failed": self.is_mission_payload_failed}