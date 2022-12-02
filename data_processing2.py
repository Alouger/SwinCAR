import matplotlib.pyplot as plt
from IPython.display import clear_output
plt.rcParams['figure.dpi'] = 200
from PIL import Image
import os
import io
from time import time

import numpy as np


from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from tqdm.auto import tqdm


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


def letterbox_mask(image, size):
    # image = image.convert("RGB")
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('L', size, 2)  # 2 is the background
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image, nw, nh



ROOT_DATA_PATH = 'E:\\oxford-iiit-pet'
IMAGES_PATH = os.path.join(ROOT_DATA_PATH, 'images')
MASKS_PATH = os.path.join(ROOT_DATA_PATH, 'annotations/trimaps')

# INPUT_SIZE = (256, 256)

images = []
masks = []
for file in tqdm(os.listdir(IMAGES_PATH)):
    image_path = os.path.join(IMAGES_PATH, file)
    mask_path = os.path.join(MASKS_PATH, file)[:-4] + '.png'
    # print(file)
    image = None
    mask = None

    try:
        image = Image.open(image_path)
        new_image, nw, nh = letterbox_image(image, (224, 224))
        # image = imread(image_path)  # read the image from disc
        # image = resize(image, INPUT_SIZE, mode='constant', anti_aliasing=True)  # resize and normilize the image
        # image = np.rollaxis(image, 2, 0)  # turn dimensions (256,256,3) -> (3, 256, 256)
        # im = Image.fromarray(image)
        new_image.save("E:/oxford-iiit-pet/processed/images/" + file[:-4] + ".png")
        # plt.imsave("E:/oxford-iiit-pet/processed/images/" + file, new_image)
        # im.save("E:/oxford-iiit-pet/processed/images/" + file)
    except:
        print('Error reading image: ' + image_path)
        continue

    # # try:
    # mask = Image.open(mask_path)
    # new_mask, nw, nh = letterbox_mask(mask, (224, 224))
    #     # image = imread(image_path)  # read the image from disc
    #     # image = resize(image, INPUT_SIZE, mode='constant', anti_aliasing=True)  # resize and normilize the image
    #     # image = np.rollaxis(image, 2, 0)  # turn dimensions (256,256,3) -> (3, 256, 256)
    #     # im = Image.fromarray(image)
    # new_mask.save("E:/oxford-iiit-pet/processed/annotations/" +  file[:-4] + ".png")
    #     # plt.imsave("E:/oxford-iiit-pet/processed/images/" + file, new_image)
    #     # im.save("E:/oxford-iiit-pet/processed/images/" + file)
    # # except:
    # #     print('Error reading image: ' + mask_path)
    # #     continue