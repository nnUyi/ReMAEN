# -*- coding=utf-8 -*-

'''
    author: Youzhao Yang
    date: 05/01/2019
    github: https://github.com/nnuyi
'''

import numpy as np
import cv2
import os

# customer libraries
from settings import *

def save_images(images, size, file_name):
    return cv2.imwrite(file_name, merge_images(images, size))

def merge_images(images, size):
    shape = images.shape
    h, w = shape[1], shape[2]
    imgs = np.zeros([h*size[0], w*size[1], 3])
    for idx, img in enumerate(images):
        i = idx % size[1]
        j = idx // size[0]
        imgs[j*h:j*h+h, i*w:i*w+w, :] = img
    return imgs

def read_data(dataset, data_path, batch_size, patch_size, dataset_size):
    try:
        img_path = os.path.join(data_path.format(dataset), train_dic[dataset][0])
        label_path = os.path.join(data_path.format(dataset), train_dic[dataset][1])
    except:
        print('no training dataset named {}'.format(dataset))
        return

    img_batch = np.zeros([batch_size, patch_size, patch_size, 3])
    label_batch = np.zeros([batch_size, patch_size, patch_size, 3])

    for i in range(batch_size):
        img_idx = np.random.randint(1, dataset_size+1)
        img = cv2.imread(img_path.format(img_idx))
        label = cv2.imread(label_path.format(img_idx))
        
        w = np.random.randint(0, img.shape[0]-patch_size)
        h = np.random.randint(0, img.shape[1]-patch_size)
        
        if np.max(img) > 1:
            img = img/255.0
        if np.max(label) > 1:
            label = label/255.0

        img_batch[i, :,:,:] = img[w:w+patch_size, h:h+patch_size,:]
        label_batch[i, :,:,:] = label[w:w+patch_size, h:h+patch_size,:]
    return img_batch, label_batch
