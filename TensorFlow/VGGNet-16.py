# -*- coding: utf-8 -*-
"""
Created on Sun May  6 14:35:04 2018

@author: zhu_lin
"""

from datetime import datetime
import math
import time
import tensorflow as tf



"""
# 创建卷积层
 input_op    输入的tensor
 name        卷基层的名称
 kh,kw       卷积核大小
 n_out       卷积核个数
 sh,sw       卷积核步长
 parameters  训练参数列表
"""
def conv_op(input_op, name, kh, kw, n_out, sh, sw, parameters):
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'weights', shape=[kh,kw,n_in,n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True, name='biases')
        conv = tf.nn.conv2d(input=input_op, filter=kernel, strides=[1,sh,sw,1], padding='SAME')
        activation = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        parameters += [kernel, biases]
        return activation
    

"""
# 创建全连接层
 input_op   输入的tensor
 name       全连接层名称
 n_out      神经元个数
 parameters 训练参数列表
"""
def fc_op(input_op, name, n_out, parameters):
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        weights = tf.get_variable(scope+'weights', shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='biases')
        activation = tf.nn.relu_layer(input_op, weights, biases, name=scope)
        parameters += [weights, biases]
        return activation
    

"""
# 创建池化层
 input_op  输入的tensor
 name      池化层名称
 kh,kw     池化卷积核大小
 sh,sw     池化卷积核步长
"""
def maxpool_op(input_op, name, kh, kw, sh, sw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, sh, sw, 1],
                          padding='SAME',
                          name=name)


"""
# 构建VGG-16卷积网络
 input_op  输入的tensor
 keep_prob dropout大小
"""
def inference_op(input_op, keep_prob):
    parameters = []
    
    conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3, n_out=64, sh=1, sw=1, parameters=parameters)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3, n_out=64, sh=1, sw=1, parameters=parameters)
    pool1 = maxpool_op(conv1_2, name='pool1', kh=2, kw=2, sh=2, sw=2)
    
    conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, sh=1, sw=1, parameters=parameters)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=128, sh=1, sw=1, parameters=parameters)
    pool2 = maxpool_op(conv2_2, name='pool2', kh=2, kw=2, sh=2, sw=2)
    
    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, sh=1, sw=1, parameters=parameters)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, sh=1, sw=1, parameters=parameters)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, sh=1, sw=1, parameters=parameters)
    pool3 = maxpool_op(conv3_3, name='pool3', kh=2, kw=2, sh=2, sw=2)
    
    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, sh=1, sw=1, parameters=parameters)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, sh=1, sw=1, parameters=parameters)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, sh=1, sw=1, parameters=parameters)
    pool4 = maxpool_op(conv4_3, name='pool4', kh=2, kw=2, sh=2, sw=2)
    
    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, sh=1, sw=1, parameters=parameters)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, sh=1, sw=1, parameters=parameters)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, sh=1, sw=1, parameters=parameters)
    pool5 = maxpool_op(conv5_3, name='pool5', kh=2, kw=2, sh=2, sw=2)
    
    # 将[batches, height, weight, tunnels]数据进行扁平化处理
    shape = pool5.get_shape()
    flatteneda_shape = shape[1].value * shape[2].value * shape[3].value
    resh1 = tf.reshape(pool5, [-1, flatteneda_shape], name='resh1')
    
    fc6 = fc_op(resh1, name='fc6', n_out=4096, parameters=parameters)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')
    
    fc7 = fc_op(fc6_drop, name='fc7', n_out=4096, parameters=parameters)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc7_drop')
    
    fc8 = fc_op(fc7_drop, name='fc8', n_out=1000, parameters=parameters)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    
    return predictions, softmax, fc8, parameters


"""
# 网络模型性能评估
 session      Session
 target       需要计算的运算图
 feed         传入的tf变量
 info_string  进行测试的名称
"""
def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i%10:
                print('%s: step %i, duration=%.3f' % (datetime.now(), i-num_steps_burn_in, duration))               
            total_duration += duration
            total_duration_squared += duration * duration
            
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec/batch' % (datetime.now(), info_string, num_batches, mn, sd))
    

"""
# 模型测试主函数
"""   
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],
                                              dtype=tf.float32,
                                              stddev=1e-1))
        
        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, parameters = inference_op(images, keep_prob)
                
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        time_tensorflow_run(sess, predictions, {keep_prob:1.0}, "Forward")      
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, {keep_prob:0.5}, "Backward")
        

batch_size = 32
num_batches = 100
run_benchmark()
    
    
    
    
    
    
        
        