# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 导入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 初始化参数
W = tf.Variable(tf.zeros([mnist.train.images.shape[1], 10]))
b = tf.Variable(tf.zeros([10]))
costs = []

# 建立模型
X = tf.placeholder(tf.float32, [None, mnist.train.images.shape[1]])
Y = tf.placeholder(tf.float32, [None, 10])

# 向前传播（softmax激活）
Y_hat = tf.nn.softmax(tf.matmul(X, W) + b)

# 成本函数
# reduction_indices=[1] 指定求平均值的维度，0-每一列求均值，1-每一行求均值
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(Y_hat), reduction_indices=[1]))

# 反向传播（梯度下降，最小化成本函数）
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# 创建session
sess = tf.InteractiveSession()
# 初始化变量（声明了变量，就必须初始化才能用）
tf.global_variables_initializer().run()

# mini-batch 训练模型
for epoch in range(1000):
    # 每次使用100个小批量数据(random-mini-batch)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run([train_step, cost], feed_dict={X: batch_xs, Y: batch_ys})

# 计算准确率
correct_prediction = tf.equal(tf.argmax(Y_hat,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

