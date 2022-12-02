#  __author__ = 'czx'
# coding=utf-8
import numpy as np
from numpy import *
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

x = np.arange(-10, 10, 0.01)
plt.tick_params(labelsize=14)  # 刻度字体大小14
y_sigmoid = sigmoid(x)

plt.plot(x, y_sigmoid, color = '#87CEFA', linewidth=2.5, label=u'sigmoid')
# plt.legend(loc='upper left', fontsize=16, frameon=False)  # 图例字体大小16
# plt.tight_layout()  # 去除边缘空白
plt.xlabel('x',fontsize=14)
plt.ylabel('Sigmoid(x)',fontsize=14)
plt.savefig("D:/My World/毕业设计/毕业论文/sigmoid.svg", dpi=600, format="svg")
# savefig要写在show前面,不然保存的就是空白图片
plt.show()
