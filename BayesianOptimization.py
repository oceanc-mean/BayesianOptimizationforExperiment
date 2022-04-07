# coding=utf-8
# @Time : 2022/3/2 18:14
# @Author : Ohmic Lab
# @File : BayesianOptimization.py 
# @Software: PyCharm

import warnings
import datetime
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


# # 最大最小归一化器(备用)
# class MinMaxScaler:
#
#     def fit_transform(self, x):
#         self.minimal = np.min(x, axis=0)
#         self.maximal = np.max(x, axis=0)
#         self.max_minus_min = _handle_zeros_in_scale(self.maximal - self.minimal, copy=False)
#
#         return (x - self.minimal) / self.max_minus_min
#
#     def inverse_transform(self, x):
#
#         return x * self.max_minus_min + self.minimal


# 采集函数
class AcquisitionFunction:
    def __init__(self, kind, y_max, explore):
        self.kind = kind
        self.y_max = y_max
        self.explore = explore

    def acq(self, x, gp):
        if self.kind == 'q-UCB':
            return self._ucb(x, gp, self.explore)
        if self.kind == 'q-EI':
            return self._ei(x, gp, self.y_max, self.explore)

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
            std = np.array([std]).reshape(-1, 1)

        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
            std = np.array([std]).reshape(-1, 1)

        z = (mean - y_max - xi) / std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)


class BO(object):
    def __init__(self):
        # 默认设置(外部输入)
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5) + WhiteKernel(),
            normalize_y=True,
            n_restarts_optimizer=9,
            random_state=self.ensure_rng()
        )
        self._acq_method = 'q-EI'
        self._explore_factor = float(0.100)
        self._data_filename = ''
        self._direction = '最大值'
        self._output_dirs = 'C:/Users/OHMIC/Desktop'
        # 默认设置(自定义外部输入)
        self._target_name = '已知目标值'
        self._target_value = ''
        self._factor_name = []
        self._factor_value = ''
        self._searching_start = ''
        self._searching_end = ''

        # 优先读取csv文件
        if self.data_filename != '':
            with open(self.data_filename, encoding='ANSI') as csv_file:
                lines = csv_file.readlines()
            self._target_name = lines[0].split(',')[0]
            self._target_value = np.array([float(line.split(',')[0]) for line in lines[1:-2]]).reshape(-1, 1)
            self._factor_name = lines[0][0:-1].split(',')[1::]
            self._factor_value = np.array([[float(item) for item in line[0:-1].split(',')[1::]] for line in lines[1:-2]])
            self._searching_start = np.array([[float(item) for item in lines[-2][0:-1].split(',')[1::]]])
            self._searching_end = np.array([[float(item) for item in lines[-1][0:-1].split(',')[1::]]])

        # 数据类型规定
        if self.acq_method not in ['q-EI', 'q-UCB']:
            err = '"{}"不在可选范围内，请选择"q-EI"、"q-UCB"中的一个'.format(self.acq_method)
            raise ValueError(err)
        if type(self.explore_factor) != float and type(self.explore_factor) != int:
            err = "探索因子应该是一个浮点数或整数"
            raise ValueError(err)
        if self.direction not in ['最大值', '最小值']:
            err = '"{}"不在可选范围内，请选择"最大值"、"最小值"中的一个'.format(self.direction)
            raise ValueError(err)
        if type(self.target_name) != str:
            err = "BO.target_name type must be python.str. "
            raise ValueError(err)
        if self.target_value is '' or type(self.target_value) != np.ndarray:
            err = "BO.target_value is not assigned or not an array-like of shape (n_samples, 1). "
            raise ValueError(err)
        if type(self.factor_name) != list:
            err = "BO.factor_name type must be python.list. "
            raise ValueError(err)
        if self.factor_value is '' or type(self.factor_value) != np.ndarray:
            err = "BO.factor_value is not assigned or not an array-like of shape (n_samples, n_features). "
            raise ValueError(err)
        if self.searching_start is '' or type(self.searching_start) != np.ndarray:
            err = "BO.searching_start is not assigned or not an array-like of shape (1, n_features). "
            raise ValueError(err)
        if self.searching_end is '' or type(self.searching_end) != np.ndarray:
            err = "BO.searching_end is not assigned or not an array-like of shape (1, n_features). "
            raise ValueError(err)

        # 数据行列规定
        if not (self.target_value.shape[0]
                == self.factor_value.shape[0] >= 2):
            err = "BO.target_value and BO.factor_value should have the same number of rows(≥2). "
            raise ValueError(err)
        if not (self.searching_start.shape[0]
                == self.searching_end.shape[0] == 1):
            err = "There is only one row for BO.searching_start and BO.searching_end. "
            raise ValueError(err)
        if not (20 >= self.factor_value.shape[1]
                == self.searching_start.shape[1]
                == self.searching_end.shape[1] >= 1):
            err = "BO.factor_value, BO.searching_start and BO.searching_end " \
                  "should have the same number of columns(≥1 & ≤20). "
            raise ValueError(err)
        if not (self.target_value.shape[1] == 1):
            err = "There is only one column for BO.searching_start and BO.searching_end. "
            raise ValueError(err)
        if self.factor_name != [] and len(self.factor_name) != self.factor_value.shape[1]:
            err = "If BO.factor_name is assigned, its length should be the same as BO.factor_value's columns. "
            raise ValueError(err)

        if not self.factor_name or self.factor_name == ['']:
            self.factor_name = ['影响因素' + str(i+1) for i in range(self.factor_value.shape[1])]

        if self.target_name == '' or self.target_name == ['']:
            target_name = '目标值'
        else:
            target_name = self.target_name

        # 自变量及其取值范围的归一化
        factors_normalizer = MinMaxScaler()
        normalized_xs = factors_normalizer.fit_transform(np.r_[self.factor_value,
                                                               self.searching_start,
                                                               self.searching_end])
        factor_value_norm = normalized_xs[0:-2, :]
        searching_start_norm = normalized_xs[-2, :]
        searching_end_norm = normalized_xs[-1, :]

        # 默认求最大值，当选最小值时对目标取负
        flip = 1
        if self.direction == '最小值':
            flip = -1

        self.target_value2 = flip * self.target_value

        # 根据维数设计随机种子的个数，这里只是随便设计的，具体数值待优化。
        a = self.factor_value.shape[1]
        if a <= 3:
            n_iter = 567
        elif a <= 4:
            n_iter = 2268
        elif a <= 10:
            n_iter = 2000 * a + 1000
        else:  # 10个以上一般没啥意义
            n_iter = 900 * a + 12000

        # sk_learn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = self.gp.fit(factor_value_norm, self.target_value2)

        # 布种
        x_seeds = self.ensure_rng().uniform(
            searching_start_norm,
            searching_end_norm,
            size=(n_iter, len(searching_start_norm))
        )

        # 定义采集函数
        acq = AcquisitionFunction(
                    self.acq_method,
                    np.max(self.target_value2, axis=0)[0],
                    explore=self.explore_factor
                ).acq

        # 预分配内存
        arrest_point_x = np.empty(shape=(0, len(searching_start_norm)))
        arrest_point_y = np.empty(shape=(0, 1))

        # 寻找驻点
        for x_try in x_seeds:
            # Find the local minimum of minus the acquisition function
            res = minimize(lambda x: - acq(x.reshape(1, -1), gp)[0],
                           x_try.reshape(1, -1)[0],
                           bounds=np.block([[searching_start_norm], [searching_end_norm]]).T,
                           method="L-BFGS-B")
            if res.success:
                arrest_point_x = np.append(arrest_point_x, [res.x], axis=0)
                arrest_point_y = np.append(arrest_point_y, - res.fun)

                arrest_point_y = arrest_point_y.reshape(-1, 1)

        # 对驻点的目标值维度归一化后赋权512
        target_normalizer = MinMaxScaler()
        arrest_point_y_512 = 512 * target_normalizer.fit_transform(arrest_point_y)
        # DBSCAN
        sample = np.concatenate((arrest_point_x, arrest_point_y_512), axis=1)
        db = DBSCAN(eps=1, min_samples=4).fit(sample)

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Ignoring noise if present.
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # Number of clusters in labels
        unique_labels = set(labels)
        (unique_labels.remove(-1) if -1 in unique_labels else unique_labels)

        result = np.empty(shape=(0, np.shape(sample)[1]))
        for k in unique_labels:
            class_member_mask = labels == k
            xy = sample[class_member_mask & core_samples_mask]
            result = np.append(result, [np.mean(xy, axis=0)], axis=0)
            # Sort the result.
            result = result[np.argsort(- result[:, -1])]

        classified_x = result[:, 0:-1]
        classified_y = target_normalizer.inverse_transform(result[:, -1].reshape(-1, 1) / 512)

        self.suggestions = factors_normalizer.inverse_transform(classified_x)
        self.expectations = flip * gp.predict(classified_x, return_std=False).reshape(-1, 1)

        # 设置csv保存路径
        curr_time = datetime.datetime.now()
        time_str = curr_time.strftime('%Y%m%d%H%M%S')
        filename = 'BO_result_' + time_str + str(self.direction) + str(self.acq_method)
        self.output_dirs = self.output_dirs + '/' + filename + '.csv'
        self.scores = ''
        if self.acq_method == 'q-EI':
            self.scores = classified_y * 100  # % 采集价值，优化的潜在可能性,有实际意义，一般低于5%就没啥优化的必要了
            self.num_str = np.concatenate((self.suggestions, self.expectations, self.scores), axis=1)
            self.header_list = self.factor_name + [target_name + '预期值', 'EI采集评分(%)']
        if self.acq_method == 'q-UCB':
            self.scores = 100 * result[:, -1].reshape(-1, 1) / 512  # % 一定置信度下样本可能的最好表现，没啥实际意义
            self.num_str = np.concatenate((self.suggestions, self.expectations), axis=1)
            self.header_list = self.factor_name + [target_name + '预期值']
        # np.savetxt(output_dirs, self.num_str, delimiter=',', header=','.join(self.header_list))

        # 获取画图所需
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x_seeds, return_std=True)
            std = np.array([std]).reshape(-1, 1)
        acq_value1 = acq(x_seeds, gp) * 100

        self.seads = np.concatenate((factors_normalizer.inverse_transform(x_seeds), acq_value1,
                                     flip * mean, flip * mean + 1.96 * std, flip * mean - 1.96 * std), axis=1)

        self.big = []
        x_tries = np.linspace(start=searching_start_norm, stop=searching_end_norm, num=1000)
        inversed_x_tries = factors_normalizer.inverse_transform(x_tries)
        for i in range(factor_value_norm.shape[1]):
            factor = factor_value_norm[:, i].reshape(-1, 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _gp = self.gp.fit(factor, self.target_value2)
                mean1, std1 = _gp.predict(x_tries[:, i].reshape(-1, 1), return_std=True)
                std1 = np.array([std1]).reshape(-1, 1)
            acq_value2 = acq(x_tries[:, i].reshape(-1, 1), _gp) * 100
            one_factor = np.concatenate(
                (inversed_x_tries[:, i].reshape(-1, 1), flip * mean1,
                 flip * mean1 + 1.96 * std1, flip * mean1 - 1.96 * std1, acq_value2), axis=1
            )
            self.big.append(one_factor)

    # 属性的查看与修改
    @property
    def gp(self):  # 查看属性
        return self._gp

    @gp.setter  # 添加或设置属性（属性名.setter）
    def gp(self, _gp):
        self._gp = _gp

    @property
    def acq_method(self):  # 查看属性
        return self._acq_method

    @acq_method.setter  # 添加或设置属性（属性名.setter）
    def acq_method(self, _acq_method):
        self._acq_method = _acq_method

    @property
    def explore_factor(self):  # 查看属性
        return self._explore_factor

    @explore_factor.setter  # 添加或设置属性（属性名.setter）
    def explore_factor(self, _explore_factor):
        self._explore_factor = _explore_factor

    @property
    def data_filename(self):  # 查看属性
        return self._data_filename

    @data_filename.setter  # 添加或设置属性（属性名.setter）
    def data_filename(self, _data_filename):
        self._data_filename = _data_filename

    @property
    def direction(self):  # 查看属性
        return self._direction

    @direction.setter  # 添加或设置属性（属性名.setter）
    def direction(self, _direction):
        self._direction = _direction

    @property
    def output_dirs(self):  # 查看属性
        return self._output_dirs

    @output_dirs.setter  # 添加或设置属性（属性名.setter）
    def output_dirs(self, _output_dirs):
        self._output_dirs = _output_dirs

    @property
    def target_name(self):  # 查看属性
        return self._target_name

    @target_name.setter  # 添加或设置属性（属性名.setter）
    def target_name(self, _target_name):
        self._target_name = _target_name

    @property
    def target_value(self):  # 查看属性
        return self._target_value

    @target_value.setter  # 添加或设置属性（属性名.setter）
    def target_value(self, _target_value):
        self._target_value = _target_value

    @property
    def factor_name(self):  # 查看属性
        return self._factor_name

    @factor_name.setter  # 添加或设置属性（属性名.setter）
    def factor_name(self, _factor_name):
        self._factor_name = _factor_name

    @property
    def factor_value(self):  # 查看属性
        return self._factor_value

    @factor_value.setter  # 添加或设置属性（属性名.setter）
    def factor_value(self, _factor_value):
        self._factor_value = _factor_value

    @property
    def searching_start(self):  # 查看属性
        return self._searching_start

    @searching_start.setter  # 添加或设置属性（属性名.setter）
    def searching_start(self, _searching_start):
        self._searching_start = _searching_start

    @property
    def searching_end(self):  # 查看属性
        return self._searching_end

    @searching_end.setter  # 添加或设置属性（属性名.setter）
    def searching_end(self, _searching_end):
        self._searching_end = _searching_end

    @staticmethod
    def ensure_rng(random_state=None):
        """
        Creates a random number generator based on an optional seed.  This can be
        an integer or another random state for a seeded rng, or None for an
        unseeded rng.
        """
        if random_state is None:
            random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        else:
            assert isinstance(random_state, np.random.RandomState)
        return random_state

    # @input_filename.deleter  # 删除属性（属性名.deleter） 注意：属性一旦删除，就无法设置和获取
    # def input_filename(self):
    #     del self._input_filename
