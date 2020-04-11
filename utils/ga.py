import numpy as np

class Genome():
    def __init__(self, pop_size, chrom_len):
        """ 初始化种群 """
        self.pop_size = pop_size
        self.chrom_len = chrom_len
        self.data = None
        self.best = None

    def select(self):
        """ 选择 """
        raise NotImplementedError()

    def cross(self):
        """ 交叉 """
        raise NotImplementedError()

    def mutate(self):
        """ 突变 """
        raise NotImplementedError()

    def __getitem__(self, index):
        return self.data[index].copy()

    def __setitem__(self, index, value):
        self.data[index] = value.copy()

    def _to_view(self, chrom):
        raise NotImplementedError()

    def view(self, index, bound):
        chrom = self._to_view(self.data[index])
        return (bound[1] - bound[0]) * chrom + bound[0]

    def view_best(self, bound):
        chrom = self._to_view(self.best)
        return (bound[1] - bound[0]) * chrom + bound[0]

class GenomeFloat(Genome):
    """ 实值编码基因组 """
    def __init__(self, pop_size, chrom_len):
        super().__init__(pop_size, chrom_len)
        self.data = np.random.uniform(0, 1, size=(pop_size, chrom_len))

    def select(self, fitness_array):
        """ 选择 """
        indices = np.random.choice(np.arange(self.pop_size), size=self.pop_size, p=fitness_array/fitness_array.sum())
        self.data, fitness_array = self.data[indices], fitness_array[indices]

    def cross(self, cross_prob):
        """ 交叉 """
        for idx in range(self.pop_size):
            if np.random.rand() < cross_prob:
                idx_other = np.random.choice(np.delete(np.arange(self.pop_size), idx), size=1)
                cross_rate = np.random.rand()
                cross_points = np.random.random(self.chrom_len) < cross_rate
                self.data[idx, cross_points], self.data[idx_other, cross_points] = \
                    (1-cross_prob) * self.data[idx, cross_points] + cross_prob * self.data[idx_other, cross_points], \
                    (1-cross_prob) * self.data[idx_other, cross_points] + cross_prob * self.data[idx, cross_points]

    def mutate(self, mutate_prob):
        """ 突变 """
        for idx in range(self.pop_size):
            mutate_points = np.random.choice(np.arange(self.chrom_len), size=int(self.chrom_len*mutate_prob), replace=False)
            mutate_values = np.random.uniform(0, 1, size=self.chrom_len)
            self.data[idx][mutate_points] = mutate_values[mutate_points]

    def _to_view(self, chrom):
        """ 将编码数据转换为可视化模式 """
        return chrom

class GenomeBinary(Genome):
    """ 二值编码基因组 """
    def __init__(self, pop_size, chrom_len, code_len=16):
        super().__init__(pop_size, chrom_len)
        self.code_len = code_len
        self.data = np.random.random((pop_size, chrom_len*code_len)) < 0.5
        self.binary_template = np.zeros(code_len)
        for i in range(code_len):
            self.binary_template[i] = (2**i) / 2**code_len

    def select(self, fitness_array):
        """ 选择 """
        indices = np.random.choice(np.arange(self.pop_size), size=self.pop_size, p=fitness_array/fitness_array.sum())
        self.data, fitness_array = self.data[indices], fitness_array[indices]

    def cross(self, cross_prob):
        """ 交叉 """
        for idx in range(self.pop_size):
            if np.random.rand() < cross_prob:
                idx_other = np.random.choice(np.delete(np.arange(self.pop_size), idx), size=1)
                cross_rate = np.random.rand()
                cross_points = np.random.random(self.chrom_len*self.code_len) < cross_rate
                self.data[idx, cross_points], self.data[idx_other, cross_points] = \
                    self.data[idx_other, cross_points], self.data[idx, cross_points]

    def mutate(self, mutate_prob):
        """ 突变 """
        for idx in range(self.pop_size):
            mutate_points = np.random.choice(np.arange(self.chrom_len*self.code_len),
                                            size=int(self.chrom_len*self.code_len*mutate_prob), replace=False)
            self.data[idx][mutate_points] = ~self.data[idx][mutate_points]

    def _to_view(self, chrom):
        """ 将编码数据转换为可视化模式 """
        return np.sum(chrom.reshape(self.chrom_len, self.code_len) * self.binary_template, axis=-1)

class GA():

    def __init__(self, pop_size, chrom_len, bound, calculate_fitness_func,
                    GenomeClass=GenomeFloat, cross_prob=0.8, mutate_prob=0.03):
        self.pop_size = pop_size
        self.chrom_len = chrom_len
        self.bound = bound
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        self.calculate_fitness = calculate_fitness_func

        self.fitness_array = np.zeros(pop_size)
        self.genome = GenomeClass(pop_size=pop_size, chrom_len=chrom_len)
        self.update_fitness()
        self.best_fitness = 0
        self.avg_fitness = 0
        self.update_records()

    def update_records(self):
        """ 更新最佳记录 """
        best_index = np.argmax(self.fitness_array)
        self.genome.best = self.genome[best_index]
        self.best_fitness = self.fitness_array[best_index]
        self.average_fitness = self.fitness_array.mean()

    def replace(self):
        """ 使用前代最佳替换本代最差 """
        self.genome[np.argmin(self.fitness_array)] = self.genome.best
        self.fitness_array[np.argmin(self.fitness_array)] = self.best_fitness

    def update_fitness(self):
        """ 重新计算适应度 """
        for idx in range(self.pop_size):
            self.fitness_array[idx] = self.calculate_fitness(self.genome.view(idx, self.bound))

    def genetic(self, num_gen, log=True):
        """ 开始运行遗传算法 """
        for e in range(num_gen):
            self.genome.select(self.fitness_array)
            self.genome.cross(self.cross_prob)
            self.genome.mutate(self.mutate_prob)
            self.update_fitness()

            if self.fitness_array[np.argmax(self.fitness_array)] > self.best_fitness:
                self.replace()
                self.update_records()

            if log:
                print('{} Best: {}, Average: {}'.format(
                    e+1, self.best_fitness, self.average_fitness))


if __name__ == '__main__':
    def F(x):
        return x + 10*np.sin(5*x) + 7*np.cos(4*x)

    def calculate_fitness(x):
        return F(x) + 50

    POP_SIZE = 5
    CHROM_LEN = 1
    X_BOUND = (-2, 5)
    bound = np.zeros((2, CHROM_LEN))
    bound[0] = X_BOUND[0] * np.ones(CHROM_LEN)
    bound[1] = X_BOUND[1] * np.ones(CHROM_LEN)


    ga = GA(POP_SIZE, CHROM_LEN, bound, calculate_fitness, GenomeClass=GenomeBinary, cross_prob=0.8, mutate_prob=0.3)
    ga.genetic(1000, log=True)

    import matplotlib.pyplot as plt
    x_axis = np.linspace(*X_BOUND, 200)
    plt.plot(x_axis, F(x_axis))
    plt.scatter(ga.genome.view_best(ga.bound), F(ga.genome.view_best(ga.bound)), color='r')
    plt.show()

    print(calculate_fitness(x_axis[np.argmax(F(x_axis))]))
    # print(x_axis[np.argmax(F(x_axis)+99)])
    # print(calculate_fitness(ga.best_chrom))
