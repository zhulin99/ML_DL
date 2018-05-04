# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 10:30:32 2018

@author: zhu_lin
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 导入数据集
minist = input_data.read_data_sets("MNIST_data/", one_hot=True)

in_units = 784
h1_units = 200

# 初始化权重
w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
w2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 向前传播
hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y_hat = tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2)

# 损失函数
# reduction_indices=[1] 指定求平均值的维度，0-每一列求均值，1-每一行求均值
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]))

# 反向传播梯度下降
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# 开始训练模型
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = minist.train.next_batch(128)
    sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.75})
     
# 计算准确度
correct_pridection = tf.equal(tf.arg_max(y,1), tf.arg_max(y_hat,1))
accuracy = tf.reduce_mean(tf.cast(correct_pridection, tf.float32))

# 测试集准确度
test_accuracy = accuracy.eval(session=sess, feed_dict={x:minist.test.images, y:minist.test.labels, keep_prob:1})

print ("测试集准确度为：%f" % test_accuracy)


