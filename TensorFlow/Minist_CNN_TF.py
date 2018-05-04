# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time


# 导入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# *************
# 权重初始化
# *************
# 加入噪声，初始化为很小的数，避免全为0，出现0梯度
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# ****************
# 卷积层和池化设置
# ****************
# 卷积f=5, s=1, p=same
# input = [batch, in_height, in_width, in_channels]
# filter = [filter_height, filter_width, in_channels, out_channels]
# strides =  [1, stride, stride, 1]  1-D int list
# return = input = [batch, height, width, out_channels]
def conv2d(x, W):
  return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

# 池化f=2, s=2, MAX
# ksize = [1, height, width, 1]   1-D int list
# strides = [1, stride, stride, 1]  1-D int list
def max_pool_2x2(x):
  return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#计算开始时间
start = time.clock()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", [None,10])

# *************
# 前向传播
# *************
# 第一个卷积层
# 卷积filter必须为4维张量[filter_height, filter_width, in_channels, out_channels]
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 输入数据必须是4维张量[batch, in_height, in_width, in_channels]
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)     # 卷积
h_pool1 = max_pool_2x2(h_conv1)                              # 池化

# 第二个卷积层
# 第二个卷积层输出为 7*7*64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层FC1
# 第二个卷积层输出为 7*7*64，FC1设置1024个神经元
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# 前向传播ReLU
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为防止过拟合在输出层之前加上Dropout，训练过程中开启dropout，测试过程中关闭
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层, softmax
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# *************
# 计算损失函数
# *************
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

# ******************
# 反向传播，梯度下降
# ******************
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# ********************
# 预测模型，准确度计算
# ********************
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# **************
# 开始训练模型
# **************
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    _, minibatch_cost = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    # 每训练100次对模型进行一次准确率的检测
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("循环： %d 次, 训练集准确度： %g" % (i, train_accuracy))

# 测试集不开启Dropout
test_accuracy = accuracy.eval(session=sess, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
print("测试集准确度： %g" % test_accuracy)

# 计算程序结束时间
end = time.clock()
print("运行时间为： %g s" %(end-start))


