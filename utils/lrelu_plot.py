#  __author__ = 'czx'
# coding=utf-8
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist


def leakyrelu(x):
    y = x.copy()
    for i in range(y.shape[0]):
        if y[i] < 0:
            y[i] = 0.2 * y[i]
    return y

x = np.arange(-10, 10, 0.01)
plt.tick_params(labelsize=14)  # 刻度字体大小14
y_relu = leakyrelu(x)

plt.plot(x, y_relu, color = '#87CEFA', linewidth=2.5, label=u'leakyrelu')
# plt.tight_layout()  # 去除边缘空白
plt.xlabel('x',fontsize=14)
plt.ylabel('LReLU(x)',fontsize=14)
plt.savefig("D:/My World/毕业设计/毕业论文/lrelu.svg", dpi=600, format="svg")
# savefig要写在show前面,不然保存的就是空白图片
plt.show()

