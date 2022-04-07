# coding=utf-8
# @Time : 2022/3/2 18:15
# @Author : Ohmic Lab
# @File : TestBO.py 
# @Software: PyCharm

from BayesianOptimization import BO
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np


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


def experiment_function(x_tries, optimal_solution1):
    # 输出实验结果和最优实验值
    x, y = x_tries[:, 0], x_tries[:, 1]
    experiment_result = x**2 + y**2
    experiment_result = - experiment_result
    # experiment_result = 5 * x_tries[:, 0] + 5 * x_tries[:, 1] + 10
    experiment_result = experiment_result.reshape(-1, 1)
    concatenate_tries_result = np.concatenate((x_tries, experiment_result), axis=1)
    optimal_solution2 = concatenate_tries_result[np.argmax(concatenate_tries_result[:, -1])]
    optimal_solution = optimal_solution2 if optimal_solution2[-1] > optimal_solution1[-1] else optimal_solution1

    return experiment_result, optimal_solution


def main():
    # Initial Parameters
    BO.gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        alpha=0.1,
        normalize_y=True,
        n_restarts_optimizer=9,
        random_state=ensure_rng()
    )
    searching_start = np.array([[-10, -10]])
    searching_end = np.array([[10, 10]])
    # b = np.array([[101]])
    # c = np.sum((searching_start - searching_end) ** 2 / 10000, axis=1)
    first_experiment_num = 3
    times = 0
    all_experiment_num = first_experiment_num
    max_parallel = 3
    scores = np.array([[101]])
    optimal_solution = np.array([[- 5e10]])

    # 初始随机布种
    x_tries = ensure_rng().uniform(
        searching_start,
        searching_end,
        size=(first_experiment_num, searching_start.shape[1])
    )

    # 获得实验结果和最优实验值
    experiment_result, optimal_solution = experiment_function(x_tries, optimal_solution)

    # print(x_tries)
    # print(experiment_result)
    # print(optimal_solution)

    while np.max(scores) > 5 and experiment_result[-1] != experiment_result[-2] != experiment_result[-3]:
        BO.searching_start = searching_start
        BO.searching_end = searching_end
        BO.factor_value = x_tries
        BO.target_value = experiment_result
        # BO.direction = '最小值'
        bo = BO()

        suggestions = bo.suggestions
        scores = bo.scores
        if suggestions[np.argwhere(scores > 5)[:, 0]].shape[0] > max_parallel:
            x_tries = np.concatenate((x_tries, suggestions[np.argwhere(scores > 5)[:, 0]][0:max_parallel, :]), axis=0)
        else:
            x_tries = np.concatenate((x_tries, suggestions[np.argwhere(scores > 5)[:, 0]]), axis=0)
        experiment_result, optimal_solution = experiment_function(x_tries, optimal_solution)
        searching_start = searching_start
        searching_end = searching_end
        times += 1
        all_experiment_num += suggestions[np.argwhere(scores > 5)[:, 0]].shape[0]
        # a = np.sum((x_tries - optimal_solution[0:-1])**2, axis=1)
        # b = a[a != 0]

        print(x_tries)
        print(experiment_result)
        # print(optimal_solution)
        # print(scores)
        print(np.max(scores))
        print(optimal_solution)
        print(times)

    print(all_experiment_num)






    # bo = BO()
    # print(bo.num_str[0, 1])


if __name__ == '__main__':
    main()

