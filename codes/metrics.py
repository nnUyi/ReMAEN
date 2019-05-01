# -*- coding=utf-8 -*-

'''
	author: YouzhaoYang
	date: 05/01/2019
	github: https://github.com/nnuyi
'''

import tensorflow as tf
import numpy as np
import cv2

class MSE:
	def __init__(self, L):
		pass

	def compute_mse(self, img1, img2):
		# RGB image		
		shape = img1.get_shape().as_list()
		if len(shape) < 3:
			raise Exception('error image type')
		diff = tf.square(img1-img2)
		mse = tf.reduce_mean(diff)
		return mse

class PSNR:
	def __init__(self, max_val=255):
		self.max_val = max_val
        
	def compute_mse(self, img1, img2):
		# RGB image
		shape = img1.get_shape().as_list()
		if len(shape) < 3:
			raise Exception('error image type')
		diff = tf.square(img1-img2)
		mse = tf.reduce_mean(diff, [-3,-2,-1])
		return mse

	def compute_rmse(self, img1, img2):
		shape = img1.get_shape().as_list()
		if len(shape) < 3:
			raise Exception('error image type')
		diff = tf.square(img1-img2)
		rmse = tf.sqrt(tf.reduce_mean(diff))
		return rmse

	def compute_psnr(self, img1, img2):
		# type transformation
		mse = self.compute_mse(img1, img2)

		if mse == 0:
			return 100.0, 100.0, 100.0
		
		psnr = 10*((tf.log(tf.square(self.max_val)/mse))/tf.log(tf.constant(10.0, dtype=tf.float32)))
		# psnr1 = 10*np.log10(float(max_val**2)/float(mse))
		# psnr2 = 20*np.log10(float(max_val)/float(np.sqrt(mse)))
		# psnr3 = 20*np.log10(float(max_val))-10*np.log10(mse)

		return psnr

class SSIM:
	def __init__(self, max_val=255):
		# hyper-parameters
		self.k1 = 0.01
		self.k2 = 0.03
		self.L = max_val
		self.c1 = (self.k1*self.L)**2
		self.c2 = (self.k2*self.L)**2
                
	def _ssim_per_channel(self, box1, box2, size=11, sigma=1.5):
		# obtain gaussian filter
		shape1 = box1.get_shape().as_list()
		
		sigma = float(sigma)
		
		g_filter = self._fspecial_gaussian(size, sigma)
		g_filter = tf.tile(g_filter, [1, 1, shape1[-1], 1])
		# lumination
		mu_x = tf.nn.depthwise_conv2d(box1, g_filter, strides=[1,1,1,1], padding='VALID')
		mu_y = tf.nn.depthwise_conv2d(box2, g_filter, strides=[1,1,1,1], padding='VALID')
		mu_x_2 = mu_x*mu_x
		mu_y_2 = mu_y*mu_y
		mu_x_y = mu_x*mu_y
		lumination = (2*mu_x_y+self.c1)/(mu_x_2+mu_y_2+self.c1)

		# contrast-structure
		mu_x2_y2 = tf.nn.depthwise_conv2d(tf.square(box1)+tf.square(box2), g_filter, strides=[1,1,1,1], padding='VALID')
		mu_xy = tf.nn.depthwise_conv2d(box1*box2, g_filter, strides=[1,1,1,1], padding='VALID')
		
		cs = (2*mu_xy - 2*mu_x_y + self.c2)/(mu_x2_y2 - mu_x_2 - mu_y_2 + self.c2)

		ssim_map = lumination*cs

		return tf.reduce_mean(ssim_map, [1,2])

	def _ssim(self, img1, img2, b_row, b_col):
		b,h,w,c = img1.get_shape().as_list()
		
		# check image format
		bool_check = self._check_image(img1, img2)
		if not bool_check:
			raise Exception('no compatible image format')
		#ssim_val = tf.reduce_mean(self._ssim_per_channel(img1[:, b_row:h-b_row, b_col:w-b_col, :], img2[:, b_row:h-b_row, b_col:w-b_col,:]), -1)
		ssim_val = tf.reduce_mean(self._ssim_per_channel(img1, img2), -1)
		
		return ssim_val

	def _fspecial_gaussian(self, size=11, sigma=1.5):
		size = tf.convert_to_tensor(size, dtype=tf.int32)
		sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
	        
		coords = tf.range(size)
		coords = tf.cast(coords, tf.float32) - tf.cast((size-1), tf.float32)/2.0
		
		g_filter = tf.square(coords)
		g_filter = g_filter*(-0.5/tf.square(sigma))
		g_filter = tf.reshape(g_filter, [1,-1])+tf.reshape(g_filter, [-1,1])
		g_filter = tf.reshape(g_filter, [1,-1])
		g_filter = tf.nn.softmax(g_filter, -1)
		g_filter = tf.reshape(g_filter, [size, size, 1, 1])
		
		return g_filter

	def _check_image(self, img1, img2):
		return img1.get_shape().as_list() == img2.get_shape().as_list()

if __name__=='__main__':
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            img1_list = []
            img2_list = []
            for i in range(1,13):
                test_data_fn = './test/Rain100L_{:03d}_in.png'.format(i)
                test_label_fn = './test/{:03d}_GT.png'.format(i)
                
                img1 = cv2.imread(test_data_fn)/255.0
                img2 = cv2.imread(test_label_fn)/255.0
                
                img1_list.append(img1)
                img2_list.append(img2)
                
            img1 = np.array(img1_list)
            img2 = np.array(img2_list)
            
            img1_tensor = tf.convert_to_tensor(img1, dtype='float32', name='img1_tensor')
            img2_tensor = tf.convert_to_tensor(img2, dtype='float32', name='img2_tensor')

            psnr = PSNR(max_val=1.0)
            psnr_list = sess.run(psnr.compute_psnr(img1_tensor,img2_tensor))
            print(psnr_list)
            psnr_val = np.mean(psnr_list)
            print('psnr:{}'.format(psnr_val))

            # test SSIM
            ssim = SSIM(max_val=1.0)
            ssim_list = sess.run(ssim._ssim(img1_tensor, img2_tensor, 0, 0))
            print(ssim_list)
            ssim_val = np.mean(ssim_list)
            print('ssim value:{}'.format(ssim_val))
