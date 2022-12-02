from PIL import Image
from skimage.io import imread, imsave, imshow
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def letterbox_image(image, size):
    image = image.convert("RGB")
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image, nw, nh

image = Image.open('E:/oxford-iiit-pet/images/Abyssinian_1.jpg')
# image = imread('E:/oxford-iiit-pet/images/Abyssinian_1.jpg')
new_image, nw, nh = letterbox_image(image, (224, 224))
# res = new_image[int((256-nh)//2):int((256-nh)//2+nh), int((256-nw)//2):int((256-nw)//2+nw)]
print(type(new_image))
new_image.save("E:/oxford-iiit-pet/processed/images/" + 'file.jpg')
# plt.imshow(new_image)
# plt.savefig("E:/oxford-iiit-pet/processed/images/" + 'file.png')
# plt.show()


# data = np.array(Image.open('E:/oxford-iiit-pet/annotations/trimaps/Abyssinian_1.png'))
# mask = np.unique(data)
# tmp = []
#
# for v in mask:
#
#     tmp.append(np.sum(data==v))
#
# ts = np.max(tmp)
#
# max_v = mask[np.argmax(tmp)]
#
# print(f'这个值：{max_v}出现的次数最多，为{ts}次')
# print(tmp)
# print(np.unique(mask))