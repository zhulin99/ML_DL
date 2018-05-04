# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
from tensorflow.python.framework import ops
import math


# 导入数据
def load_dataset():
    train_dataset = h5py.File('Sign_data/train_signs.h5', "r")  # 训练集
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('Sign_data/test_signs.h5', "r")    # 测试集
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# ong-hot编码
def convert_to_one_hot(Y, C):
    # np.eye():生成单位矩阵
    # np.eye(C)[n] 返回下标为n的指定行
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# 创建X，Y的占位符
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None])
    Y = tf.placeholder(tf.float32, [n_y, None])
    return X, Y


# 参数W进行xavier初始化，b初始化为0
def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


# 前向传播
def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

# 成本计算
def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


# 随机小批量
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[1]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    num_complete_minibatches = math.floor(m / mini_batch_size)  # 小批量数量
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:  # 剩余的小批量
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# 模型训练
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
    # 使重新运行时不覆盖tf的变量
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)  # 占位符
    # 初始化参数
    parameters = initialize_parameters()
    # 前向传播
    Z3 = forward_propagation(X, parameters)
    # 损失计算
    cost = compute_cost(Z3, Y)
    # 反向传播（使用Adam梯度下降）
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()  # 变量初始化
    # 训练模型参数
    with tf.Session() as sess:
        sess.run(init)  # 运行变量初始化
        for epoch in range(num_epochs):
            epoch_cost = 0.  # 每epoch一次后的成本值
            num_minibatches = int(m / minibatch_size)  # 小批量数目
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)  # 随机小批量

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 100 == 0:
                print("%i epoch 后的成本值 : %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print("参数训练完毕")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("训练集准确度:", accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集准确度:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T  # 转置
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    X_train = X_train_flatten / 255.  # 标准化
    X_test = X_test_flatten / 255.
    Y_train = convert_to_one_hot(Y_train_orig, 6)  # 变为one-hot形式
    Y_test = convert_to_one_hot(Y_test_orig, 6)

    parameters = model(X_train, Y_train, X_test, Y_test)



