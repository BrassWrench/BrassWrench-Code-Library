from bernuli_bandit import *

class ThompsonSampling(Solver):
    """ 汤普森采样算法,继承Solver类 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k

if __name__ == '__main__':

    np.random.seed(1)  # 设定随机种子,使实验具有可重复性
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print("随机生成了一个%d臂伯努利老虎机" % K)
    print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
          (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

    np.random.seed(1)
    thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompson_sampling_solver.run(5000)
    print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])