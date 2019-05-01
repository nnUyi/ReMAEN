# -*- coding=utf-8 -*-

'''
	author: YouzhaoYang
	date: 05/01/2019
	github: https://github.com/nnuyi
'''

import tensorflow as tf
import argparse

# configuration
parser = argparse.ArgumentParser()
##################################### learning params ####################################
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
##################################### training params ####################################
parser.add_argument('--patch_size', type=int, default=64, help='image size')
parser.add_argument('--channel_dim', type=int, default=32, help='numbers of feature maps')
parser.add_argument('--input_channels', type=int, default=3, help='numbers of image channels')
parser.add_argument('--is_training', type=bool, default=False, help='training or testing')
parser.add_argument('--is_testing', type=bool, default=False, help='training or testing')
parser.add_argument('--iterations', type=int, default=40000, help='training iterations')
parser.add_argument('--batch_size', type=int, default=32, help='training batchsize')
parser.add_argument('--phase', type=str, default='train', help='indicating training or testing phase')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='checkpoint directory')
parser.add_argument('--test_dir', default='test', type=str, help='testing directory')
parser.add_argument('--sample_dir', type=str, default='sample', help='sample directory')
parser.add_argument('--logs_dir', type=str, default='logs', help='log directory')
################################# training and testing set ################################
#parser.add_argument('--data_path', type=str, default='/home/yyz/yyz_pj/datasets/Derain/train/{}/origin_data/', help='training data path')
#parser.add_argument('--test_path', type=str, default='/home/yyz/yyz_pj/datasets/Derain/test/{}/', help='testing data path')
parser.add_argument('--data_path', type=str, default='/YyzData/Derain/datasets/train/{}/', help='training data path')
parser.add_argument('--test_path', type=str, default='/YyzData/Derain/datasets/test/{}/', help='testing data path')
parser.add_argument('--train_dataset', type=str, required=True, help='dataset name')
parser.add_argument('--test_dataset', type=str, required=True, help='dataset name')
parser.add_argument('--trainset_size', type=int, required=True, help='number of images in training set')
parser.add_argument('--testset_size', type=int, required=True, help='number of images in testing set')

args = parser.parse_args()

# kernel of edge loss
kernel=tf.constant([
            [[[-1.]], [[-1.]], [[-1.]]],
            [[[-1.]], [[8.]], [[-1.]]],
            [[[-1.]], [[-1.]], [[-1.]]]])

# training datasets
train_dic = {
             'Rain100L':('rain-{}.png', 'norain-{}.png'),
             'Rain100H':('rain-{}.png', 'norain-{}.png'),
             'Rain800':('rain-{}.jpg', 'norain-{}.jpg'),
             'Rain1200':('rain-{}.jpg', 'norain-{}.jpg'),
             'Rain1400':('rain-{}.jpg', 'norain-{}.jpg'),
            }

# testing datasets
test_dic = {
            'Rain100L':(100, 'rain-{:03d}.png', 'norain-{:03d}.png'),
            'Rain100H':(100, 'rain-{:03d}.png', 'norain-{:03d}.png'),
            'Rain800':(100, 'rain-{:03d}.png', 'norain-{:03d}.png'),
            'Rain1200':(1200, 'rain-{:03d}.jpg', 'norain-{:03d}.jpg'),
            'Rain1400':(1400, 'rain-{:03d}.jpg', 'norain-{:03d}.jpg'),
            'Practical':(67, 'rain-{:03d}.jpg', 'norain-{:03d}.jpg')
           }

