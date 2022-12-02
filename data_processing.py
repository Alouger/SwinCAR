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


def display(display_X, display_Y):
    n_images = len(display_X)
    plt.figure(figsize=(18, n_images))
    for i in range(n_images):
        plt.subplot(2, n_images, i+1)
        plt.axis("off")
        plt.imshow(np.rollaxis(display_X[i], 0, 3))

        plt.subplot(2, n_images, i+n_images+1)
        plt.axis("off")
        plt.imshow(display_Y[i][0])
    plt.show()


ROOT_DATA_PATH = 'E:\\oxford-iiit-pet'
IMAGES_PATH = os.path.join(ROOT_DATA_PATH, 'images')
MASKS_PATH = os.path.join(ROOT_DATA_PATH, 'annotations/trimaps')

INPUT_SIZE = (256, 256)

images = []
masks = []
for file in tqdm(os.listdir(IMAGES_PATH)):
    image_path = os.path.join(IMAGES_PATH, file)
    mask_path = os.path.join(MASKS_PATH, file)[:-4] + '.png'
    print(file)
    image = None
    mask = None

    try:
        image = imread(image_path)  # read the image from disc
        image = resize(image, INPUT_SIZE, mode='constant', anti_aliasing=True)  # resize and normilize the image
        # image = np.rollaxis(image, 2, 0)  # turn dimensions (256,256,3) -> (3, 256, 256)
        # im = Image.fromarray(image)

        plt.imsave("E:/oxford-iiit-pet/processed/images/" + file, image)
        # im.save("E:/oxford-iiit-pet/processed/images/" + file)
    except:
        print('Error reading image: ' + image_path)
        continue

    # try:
        # read the trimap
    with open(mask_path, "rb") as fid:
        encoded_mask_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_mask_png)
    mask = imread(encoded_png_io)

    # select only object mask for now
    tmp_mask = np.zeros(mask.shape, dtype=float)
    tmp_mask[mask == 1] = 1
    mask = tmp_mask

    # resize the mask
    mask = resize(mask, INPUT_SIZE, mode='constant', anti_aliasing=False)

    # our UNet outputs (1, 256, 256) mask shall have same dimensions
    tmp_mask = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1]))
    tmp_mask[0] = mask
    mask = tmp_mask

    plt.imsave("E:/oxford-iiit-pet/processed/annotations/" + file[:-4] + ".png", mask)

        # ma = Image.fromarray(mask)
        # ma.save("E:/oxford-iiit-pet/processed/annotations/" + file[:-4] + ".png")
    # except:
    #     print('Error reading mask: ' + mask_path)
    #     continue

    if image is None or mask is None:
        continue

    if len(image.shape) != 3 or image.shape[0] != 3:
        print('Expected 3 channels. But got ' + str(image.shape) + ' in ' + image_path)
        continue

    # matplotlib.imsave('"E:/oxford-iiit-pet/processed/images", array)
    # cv2.imwrite("E:/oxford-iiit-pet/processed/images", image)
    # cv2.imwrite("E:/oxford-iiit-pet/processed/annotations", mask)
    images.append(image)
    masks.append(mask)

display(images[:6], masks[:6])