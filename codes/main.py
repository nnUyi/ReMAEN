# -*- coding=utf-8 -*-

'''
    author: Youzhao Yang
    date: 05/01/2019
    github: https://github.com/nnuyi
'''

import tensorflow as tf
import os

# customer libraries
from ReMAEN import DerainNet
from settings import *

# create directory
def check_dir():
    if not os.path.exists('sample'):
        os.mkdir('sample')
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.exists('test'):
        os.mkdir('test')

# print training configuration
def print_config():
    print('ConfigProto:')
    print('-----------------------------------------------------')
    print('train_dataset:{}'.format(args.train_dataset))
    print('batch_size:{}'.format(args.batch_size))
    print('iterations:{}'.format(args.iterations))
    print('trainset_size:{}'.format(args.trainset_size))
    print('testset_size:{}'.format(args.testset_size))
    print('phase:{}'.format(args.phase))
    print('-----------------------------------------------------')

# training entry
def main(_):
    check_dir()
    print_config()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    gpu_options.allow_growth = True
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=config) as sess:        
        derainnet = DerainNet(args, sess=sess)
        if args.is_training:
            derainnet.build()
            derainnet.train()
        if args.is_testing:
            derainnet.test()

if __name__=='__main__':
    tf.app.run()
