"""
TO solve param
or use SPSSPRO
"""

import numpy as np


# 定义损失函数
def compute_error(a, b, c, x, y):
    return np.sum((y - (a * x**2 + b * x + c)) ** 2) / len(x)

# 定义梯度下降函数
def gradient_descent(a_current, b_current, c_current, x, y, learning_rate):
    N = float(len(x))
    a_gradient = -(2/N) * np.sum(x**2 * (y - (a_current * x**2 + b_current * x + c_current)))
    b_gradient = -(2/N) * np.sum(x * (y - (a_current * x**2 + b_current * x + c_current)))
    c_gradient = -(2/N) * np.sum(y - (a_current * x**2 + b_current * x + c_current))
    new_a = a_current - (learning_rate * a_gradient)
    new_b = b_current - (learning_rate * b_gradient)
    new_c = c_current - (learning_rate * c_gradient)
    return [new_a, new_b, new_c]

# 初始化参数
a = 0
b = 0
c = 0

# 定义学习率和迭代次数
learning_rate = 0.0001
num_iterations = 1000

# 假设我们有一些真实的输入和输出数据
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([3, 7, 13, 21, 31])

# 使用梯度下降法求解参数
for i in range(num_iterations):
    a, b, c = gradient_descent(a, b, c, x_data, y_data, learning_rate)

print("After {0} iterations a = {1}, b = {2}, c = {3}, error = {4}".format(num_iterations, a, b, c,
                                                                            compute_error(a, b, c,
                                                                                          x_data,
                                                                                          y_data)))
