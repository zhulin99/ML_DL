# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:48:30 2018

@author: zhu_lin
"""

import cifar10
import cifar10_input
import tensorflow as tf
import numpy as np
import time
import math  


max_steps = 3000
batch_size = 128
data_dir = 'cifar10_data/cifar-10-batches-bin'


# 初始化权重参数，并给权重加入L2损失，相当于做了L2正则化处理
# 其中tf.nn.l2_loss()计算的是没有开方的L2范数值的一半：sum(var ** 2)/2
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weigt_lose')
        tf.add_to_collection('losses', weight_loss)
    return var


# 从CIFAR10读取数据
# cifar10.maybe_download_and_extract()
# 读取训练数据并进行数据增强（反转、剪裁、对比度），以及数据标准化
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)


images_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])


# 第一层 卷积层(Conv1 + Maxpool + LRN)
weight1 = variable_with_weight_loss(shape=[5,5,3,64], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(input=images_holder, filter=weight1, strides=[1,1,1,1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)


# 第二层 卷积层(Conv1 + LRN + Maxpool)
weight2 = variable_with_weight_loss(shape=[5,5,64,64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(input=norm1, filter=weight2, strides=[1,1,1,1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')


# 第三层 全连接层FC1
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 第四层 全连接层FC2
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 第五层 输出层
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)


# 损失函数
# tf.nn.softmax_cross_entropy_with_logits         数据需要one-hot编码
# tf.nn.sparse_softmax_cross_entropy_with_logits  数据不需要one-hot编码
def loss(logits, labels):
    # 非稀疏表示的label，type类型必须为int型
    labels = tf.cast(labels, tf.int64)
    # 将softmax和交叉熵计算放在一起，先算softmax,再算交叉熵（原来需要两步计算，现在一步就可以）
    # 1、y=tf.nn.softmax(logits)    2、cross_entropy=-tf.reduce_sum(y_*tf.log(y))  
    # tf.nn.sparse_softmax_cross_entropy_with_logits()  结果返回的是向量
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                                                                   labels=labels,
                                                                   name='cross_entropy_per_eg')
    # 如果要求交叉熵，我们要再做一步tf.reduce_sum操作
    # 如果求loss，则要做一步tf.reduce_mean操作
    # 因为sparse_softmax_cross_entropy_with_logits() 求得是：-tf.reduce_sum(y_ * tf.log(y_conv))
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

loss = loss(logits, label_holder)


# 反向传播，梯度下降
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


# 计算测试集准确率
# 判断预测结果最大的那个数的下标是否和标签集中对应，返回bool型向量
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)


# 开始模型训练
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss], feed_dict={images_holder:image_batch, label_holder:label_batch})
    duration = time.time() - start_time
    
    if step%10 == 0:
        examples_per_sec = batch_size / duration    # 计算每秒处理多少条数据
        sec_per_batch = float(duration)             # 计算处理一个batch需要多少秒
        
        format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
        

# 评测模型在测试集上准确率
num_examples = 10000
num_iter = int(math.ceil(num_examples/batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={images_holder:image_batch, label_holder:label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print('precision @ 1=%.3f' % precision)
        
    
    










