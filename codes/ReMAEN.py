# -*- coding=utf-8 -*-

'''
	author: YouzhaoYang
	date: 05/01/2019
	github: https://github.com/nnuyi
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
import cv2

# customer libraries
from utils import save_images, read_data
from metrics import SSIM, PSNR
from settings import *

class DerainNet:
    model_name = 'DerainNet'
    
    '''Derain Net: all the implemented layer are included (e.g. MAEB,
                                                                convGRU
                                                                shared channel attention,
                                                                channel attention).

        Params:
            config: the training configuration
            sess: runing session
    '''
    
    def __init__(self, config, sess=None):
        # config proto
        self.config = config
        self.channel_dim = self.config.channel_dim
        self.batch_size = self.config.batch_size
        self.patch_size = self.config.patch_size
        self.input_channels = self.config.input_channels
        
        # metrics
        self.ssim = SSIM(max_val=1.0)
        self.psnr = PSNR(max_val=1.0)

        # create session
        self.sess = sess
    
    # global average pooling
    def globalAvgPool2D(self, input_x):
        global_avgpool2d = tf.contrib.keras.layers.GlobalAvgPool2D()
        return global_avgpool2d(input_x)
    
    # leaky relu
    def leakyRelu(self, input_x):
        leaky_relu = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)
        return leaky_relu(input_x)

    # squeeze-and-excitation block
    def SEBlock(self, input_x, input_dim=32, reduce_dim=8, scope='SEBlock'):
        with tf.variable_scope(scope) as scope:
            # global scale
            global_pl = self.globalAvgPool2D(input_x)
            reduce_fc1 = slim.fully_connected(global_pl, reduce_dim, activation_fn=tf.nn.relu)
            reduce_fc2 = slim.fully_connected(reduce_fc1, input_dim, activation_fn=None)
            g_scale = tf.nn.sigmoid(reduce_fc2)
            g_scale = tf.expand_dims(g_scale, axis=1)
            g_scale = tf.expand_dims(g_scale, axis=1)
            gs_input = input_x*g_scale
            return gs_input

    # GRU with convolutional version
    def convGRU(self, input_x, h, out_dim, scope='convGRU'):
        with tf.variable_scope(scope):
            if h is None:
                self.conv_xz = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xz')
                self.conv_xn = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xn')
                z = tf.nn.sigmoid(self.conv_xz)
                f = tf.nn.tanh(self.conv_xn)
                h = z*f
            else:
                self.conv_hz = slim.conv2d(h, out_dim, 3, 1, scope='conv_hz')
                self.conv_hr = slim.conv2d(h, out_dim, 3, 1, scope='conv_hr')

                self.conv_xz = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xz')
                self.conv_xr = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xr')
                self.conv_xn = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xn')
                r = tf.nn.sigmoid(self.conv_hr+self.conv_xr)
                z = tf.nn.sigmoid(self.conv_hz+self.conv_xz)
                
                self.conv_hn = slim.conv2d(r*h, out_dim, 3, 1, scope='conv_hn')
                n = tf.nn.tanh(self.conv_xn + self.conv_hn)
                h = (1-z)*h + z*n

        # shared channel attention block
        se = self.SEBlock(h, out_dim, reduce_dim=int(out_dim/4))
        h = self.leakyRelu(se)
        return h, h

    # multi-scale aggregation and enhancement block(MAEB)
    def MAEB(self, input_x, scope_name, dilated_factors=3):
        '''MAEB: multi-scale aggregation and enhancement block
            Params:
                input_x: input data
                scope_name: the scope name of the MAEB (customer definition)
                dilated_factor: the maximum number of dilated factors(default=3, range from 1 to 3)

            Return:
                return the output the MAEB
                
            Input shape:
                4D tensor with shape '(batch_size, height, width, channels)'
                
            Output shape:
                4D tensor with shape '(batch_size, height, width, channels)'
        '''
        dilate_c = []  
        with tf.variable_scope(scope_name):
            for i in range(1,dilated_factors+1):
                d1 = self.leakyRelu(slim.conv2d(input_x, self.channel_dim, 3, 1, rate=i, activation_fn=None, scope='d1'))
                d2 = self.leakyRelu(slim.conv2d(d1, self.channel_dim, 3, 1, rate=i, activation_fn=None, scope='d2'))
                dilate_c.append(d2)

            add = tf.add_n(dilate_c)
            shape = add.get_shape().as_list()
            output = self.SEBlock(add, shape[-1], reduce_dim=int(shape[-1]/4))
            return output

    # multi-scale aggregation and enhancement network
    def derainNet(self, input_x, scope_name='derainNet'):    
        '''ReMAEN: recurrent multi-scale aggregation and enhancement network
            Params:
                input_x: input data
                scope_name: the scope name of the ReMAEN (customer definition, default='derainnet')
            Return:
                return the derained results

            Input shape:
                4D tensor with shape '(batch_size, height, width, channels)'
                
            Output shape:
                4D tensor with shape '(batch_size, height, width, channels)'            
        '''
        # reuse: tf.AUTO_REUSE(such setting will enable the network to reuse parameters automatically)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d,slim.conv2d_transpose], weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              normalizer_fn = None,
                                              activation_fn = None,
                                              padding='SAME'):
                old_states = [None for _ in range(7)]
                stages = 3
                derain = input_x

                for i in range(stages):
                    cur_states = []
                    with tf.variable_scope('ReMAEN'):
                        with tf.variable_scope('extracting_path'):
                            MAEB1 = self.MAEB(derain, scope_name='MAEB1')
                            gru1, h1 = self.convGRU(MAEB1, old_states[0], self.channel_dim, scope='convGRU1')
                            cur_states.append(h1)

                            MAEB2 = self.MAEB(gru1, scope_name='MAEB2')
                            gru2, h2 = self.convGRU(MAEB2, old_states[1], self.channel_dim, scope='convGRU2')
                            cur_states.append(h2)
                            
                            MAEB3 = self.MAEB(gru2, scope_name='MAEB3')
                            gru3, h3 = self.convGRU(MAEB3, old_states[2], self.channel_dim, scope='convGRU3')
                            cur_states.append(h3)

                            MAEB4 = self.MAEB(gru3, scope_name='MAEB4')
                            gru4, h4 = self.convGRU(MAEB4, old_states[3], self.channel_dim, scope='convGRU4')
                            cur_states.append(h4)
                            
                        with tf.variable_scope('responding_path'):
                            up5 = slim.conv2d(gru4, self.channel_dim, 3, 1, activation_fn=tf.nn.relu, scope='conv5')
                            add5 = tf.add(up5, MAEB3)
                            gru5, h5 = self.convGRU(add5, old_states[4], self.channel_dim, scope='convGRU5')
                            cur_states.append(h5)
                            
                            up6 = slim.conv2d(gru5, self.channel_dim, 3, 1, activation_fn=tf.nn.relu, scope='conv6')
                            add6 = tf.add(up6, MAEB2)
                            gru6, h6 = self.convGRU(add6, old_states[5], self.channel_dim, scope='convGRU6')
                            cur_states.append(h6)
                            
                            up7 = slim.conv2d(gru6, self.channel_dim, 3, 1, activation_fn=tf.nn.relu, scope='conv7')
                            add7 = tf.add(up7, MAEB1)
                            gru7, h7 = self.convGRU(add7, old_states[6], self.channel_dim, scope='convGRU7')
                            cur_states.append(h7)
                        
                    # residual map generator
                    with tf.variable_scope('RMG'):
                        rmg_conv = slim.conv2d(gru7, self.channel_dim, 3, 1)
                        rmg_conv_se = self.leakyRelu(self.SEBlock(rmg_conv, self.channel_dim, reduce_dim=int(self.channel_dim/4)))
                        residual = slim.conv2d(rmg_conv_se, self.input_channels, 3, 1)
                    
                    derain = derain - residual
                    old_states = [tf.identity(s) for s in cur_states]

        return derain, residual
    
    def build(self):
        # placeholder
        self.rain = tf.placeholder(tf.float32, [None, None, None, self.input_channels], name='rain')
        self.norain = tf.placeholder(tf.float32, [None, None, None, self.input_channels], name='norain')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        
        # derainnet
        self.out, self.residual = self.derainNet(self.rain)
        self.finer_out = tf.clip_by_value(self.out, 0, 1.0)
        self.finer_residual = tf.clip_by_value(tf.abs(self.residual), 0, 1)
        
        # metrics
        self.ssim_finer_tensor = tf.reduce_mean(self.ssim._ssim(self.norain, self.out, 0, 0))
        self.psnr_finer_tensor = tf.reduce_mean(self.psnr.compute_psnr(self.norain, self.out))
        self.ssim_val = tf.reduce_mean(self.ssim._ssim(self.norain, self.finer_out, 0, 0))
        self.psnr_val = tf.reduce_mean(self.psnr.compute_psnr(self.norain, self.finer_out))
        
        # loss function
        # MSE loss
        self.l2_loss = tf.reduce_mean(tf.square(self.out - self.norain))
        # edge loss, kernel is imported from settings
        self.norain_edge = tf.nn.relu(tf.nn.conv2d(tf.image.rgb_to_grayscale(self.norain), kernel, [1,1,1,1],padding='SAME'))
        self.derain_edge = tf.nn.relu(tf.nn.conv2d(tf.image.rgb_to_grayscale(self.out), kernel, [1,1,1,1],padding='SAME'))
        self.edge_loss = tf.reduce_mean(tf.square(self.norain_edge-self.derain_edge))
        # total loss
        self.total_loss = self.l2_loss + 0.1*self.edge_loss
        
        # optimization
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if 'derainNet' in var.name]
        self.train_ops = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.config.beta1, beta2=self.config.beta2).minimize(self.total_loss, var_list=g_vars)
        
        # summary
        self.l2_loss_summary = tf.summary.scalar('l2_loss', self.l2_loss)
        self.total_loss_summary = tf.summary.scalar('total_loss', self.total_loss)
        self.edge_loss_summary = tf.summary.scalar('edge_loss', self.edge_loss)
        self.ssim_summary = tf.summary.scalar('ssim', self.ssim_val)
        self.psnr_summary = tf.summary.scalar('psnr', self.psnr_val)
        self.summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.config.logs_dir, self.sess.graph)
        
        # saver
        global_variables = tf.global_variables()
        var_to_store = [var for var in global_variables if 'derainNet' in var.name]
        self.saver = tf.train.Saver(var_list=var_to_store)

        # trainable variables
        num_params = 0
        for var in g_vars:
            tmp_num = 1
            for i in var.get_shape().as_list():
                tmp_num = tmp_num*i
            num_params = num_params + tmp_num
        print('numbers of trainable parameters:{}'.format(num_params))

    # training phase
    def train(self):
        # initialize variables
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # load training model
        check_bool = self.load_model()
        if check_bool:
            print('[!!!] load model successfully')
        else:
            print('[***] fail to load model')
        
        lr_ = self.config.lr
        start_time = time.time()
        for counter in range(self.config.iterations):
            if counter == 30000:
                lr_ = 0.1*lr_

            # obtain training image pairs
            img, label = read_data(self.config.train_dataset, self.config.data_path, self.batch_size, self.patch_size, self.config.trainset_size)
            _, total_loss, summaries, ssim, psnr = self.sess.run([self.train_ops,
                                                               self.total_loss,
                                                               self.summaries,
                                                               self.ssim_val,
                                                               self.psnr_val], feed_dict={self.rain:img,
                                                                                           self.norain:label,
                                                                                           self.lr:lr_})

            print('Iteration:{}, phase:{}, loss:{:.4f}, ssim:{:.4f}, psnr:{:.4f}, lr:{}, iterations:{}'.format(counter,
                                                                                                                 self.config.phase,
                                                                                                                 total_loss,
                                                                                                                 ssim,
                                                                                                                 psnr,
                                                                                                                 lr_,
                                                                                                                 self.config.iterations))
                                
            self.summary_writer.add_summary(summaries, global_step=counter)
            if np.mod(counter, 100)==0:
                self.sample(self.config.sample_dir, counter)

            if np.mod(counter, 500)==0:
                self.save_model()
        
        # save final model
        if counter == self.config.iterations-1:
            self.save_model()

        # training time
        end_time = time.time()
        print('training time:{} hours'.format((end_time-start_time)/3600.0))

    # sampling phase
    def sample(self, sample_dir, iterations):
        # obtaining sampling image pairs
        test_img, test_label = read_data(self.config.test_dataset, self.config.data_path, self.batch_size, self.patch_size, self.config.testset_size)
        finer_out, finer_residual = self.sess.run([self.finer_out, self.finer_residual], feed_dict={self.rain:test_img})
        
        # save sampling images
        test_img_uint8 = np.uint8(test_img*255.0)
        test_label_uint8 = np.uint8(test_label*255.0)
        finer_out_uint8 = np.uint8(finer_out*255.0)
        finer_residual = np.uint8(finer_residual*255.0)
        sample = np.concatenate([test_img_uint8, test_label_uint8, finer_out_uint8, finer_residual], 2)
        save_images(sample, [int(np.sqrt(self.batch_size))+1, int(np.sqrt(self.batch_size))+1], '{}/{}_{}_{:04d}.jpg'.format(self.config.sample_dir,
                                                                                                                             self.config.test_dataset,
                                                                                                                             self.config.phase,
                                                                                                                             iterations))
    
    # testing phase
    def test(self):
        rain = tf.placeholder(tf.float32, [None, None, None, self.input_channels], name='test_rain')
        norain = tf.placeholder(tf.float32, [None, None, None, self.input_channels], name='test_norain')
        
        out, residual = self.derainNet(rain)
        finer_out = tf.clip_by_value(out, 0, 1.0)
        finer_residual = tf.clip_by_value(tf.abs(residual), 0, 1.0)

        ssim_val = tf.reduce_mean(self.ssim._ssim(norain, finer_out, 0, 0))
        psnr_val = tf.reduce_mean(self.psnr.compute_psnr(norain, finer_out))

        # load model
        self.saver = tf.train.Saver()
        check_bool = self.load_model()
        if check_bool:
            print('[!!!] load model successfully')
        else:
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()
            print('[***] fail to load model')

        try:
            test_num, test_data_format, test_label_format = test_dic[self.config.test_dataset]
        except:
            print('no testing dataset named {}'.format(self.config.test_dataset))
            return

        ssim = []
        psnr = []
        for index in range(1, test_num+1):
            test_data_fn = test_data_format.format(index)
            test_label_fn = test_label_format.format(index)
            
            test_data_path = os.path.join(self.config.test_path.format(self.config.test_dataset), test_data_fn)
            test_label_path = os.path.join(self.config.test_path.format(self.config.test_dataset), test_label_fn)

            test_data_uint8 = cv2.imread(test_data_path)
            test_label_uint8 = cv2.imread(test_label_path)

            test_data_float = test_data_uint8/255.0
            test_label_float = test_label_uint8/255.0
            
            test_data = np.expand_dims(test_data_float, 0)
            test_label = np.expand_dims(test_label_float, 0)
            
            t = 0
            s_t = time.time()
            finer_out_val, finer_residual_val, tmp_ssim, tmp_psnr = self.sess.run([finer_out,
                                                                                   finer_residual,
                                                                                   ssim_val,
                                                                                   psnr_val] , feed_dict={rain:test_data,
                                                                                                          norain:test_label})

            e_t = time.time()            
            total_t = e_t - s_t
            t = t + total_t

            # save psnr and ssim metrics
            ssim.append(tmp_ssim)
            psnr.append(tmp_psnr)
            # save testing image
            test_label = np.uint8(test_label*255)
            finer_out_val = np.uint8(finer_out_val*255)
            finer_residual_val = np.uint8(finer_residual_val*255)
            save_images(finer_out_val, [1,1], '{}/{}_{}'.format(self.config.test_dir, self.config.test_dataset, test_data_fn))
            save_images(test_label, [1,1], '{}/{}'.format(self.config.test_dir, test_data_fn))
            save_images(finer_residual_val, [1,1], '{}/residual_{}'.format(self.config.test_dir, test_data_fn))
            print('test image {}: ssim:{}, psnr:{} time:{:.4f}'.format(test_data_fn, tmp_ssim, tmp_psnr, total_t))
        
        mean_ssim = np.mean(ssim)
        mean_psnr = np.mean(psnr)
        print('Test phase: ssim:{}, psnr:{}'.format(mean_ssim, mean_psnr))
        print('Average time:{}'.format(t/(test_num-1)))

    # save model            
    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.config.train_dataset,
            self.batch_size)
    @property
    def model_pos(self):
        return '{}/{}/{}'.format(self.config.checkpoint_dir, self.model_dir, self.model_name)

    def save_model(self):
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)
        self.saver.save(self.sess, self.model_pos)
        
    def load_model(self):
        if not os.path.isfile(os.path.join(self.config.checkpoint_dir, self.model_dir,'checkpoint')):
            return False
        else:
            self.saver.restore(self.sess, self.model_pos)
            return True
