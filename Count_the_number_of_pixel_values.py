from PIL import Image
from skimage.io import imread, imsave, imshow
import matplotlib.pyplot as plt
import numpy as np


data = np.array(Image.open('E:/oxford-iiit-pet/processed/trainval_label/0001.png'))
mask = np.unique(data)
tmp = []

for v in mask:

    tmp.append(np.sum(data==v))

ts = np.max(tmp)
tm = np.min(tmp)

max_v = mask[np.argmax(tmp)]
min_v = mask[np.argmin(tmp)]

print(f'这个值：{max_v}出现的次数最多，为{ts}次')
print(f'这个值：{min_v}出现的次数最少，为{tm}次')
print(tmp)