# -*- coding:utf-8 -*-
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Xavier初始化器，生成（low，high）之间的均匀分布
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class AdditiveGaussianNoiseAutoenconder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, 
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        
        networks_weights = self._initialize_weights()
        self.weights = networks_weights
        
        # 前向传播
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 加入高斯噪声
        self.addGNx = self.x + scale * tf.random_normal((n_input,))
        # 隐含层
        self.hidden = self.transfer(tf.add(tf.matmul(self.addGNx, self.weights['w1']), self.weights['b1']))
        # 输出层
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        
        # 计算损失
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x) ,2.0))
        # 梯度下降Adam优化
        self.optimizer = optimizer.minimize(self.cost)
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
     
    # 初始化参数
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights
    
    # 训练函数
    def partial_fit(self, X):
        cost, optimize = self.sess.run([self.cost, self.optimizer], 
                                       feed_dict={self.x:X, self.scale:self.training_scale})
        return cost
    
    # 测试集损失计算
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x:X, self.scale:self.training_scale})
    
    # 获取隐含层输出结果（即学习的高阶特征）
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x:X, self.scale:self.training_scale})
    
    # 获取输出结果数据（隐含层复原数据）
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden:hidden})
    
    
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x:X, self.scale:self.training_scale})
    
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    
    def getBiases(self):
        return self.sess.run(self.weights['b1'])
    


# 对数据进行标准化处理(即均值为0，标准差为1的分布)
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)    # 计算数据的均值和标准差
    X_train = preprocessor.transform(X_train)            # 将数据转化为标准化分布
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


# 随机获取mini-batch数据(不放回抽样)
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


# 训练模型
def model(X_train, X_test, n_samples, num_epochs=20, batch_size=128, hidden_num=200, 
          learning_rate=0.001, scale=0.01):
    autoencoder = AdditiveGaussianNoiseAutoenconder(n_input = 784, 
                                                n_hidden = hidden_num, 
                                                transfer_function = tf.nn.softplus, 
                                                optimizer = tf.train.AdamOptimizer(learning_rate), 
                                                scale = scale)
    for epoch in range(num_epochs):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / total_batch
        
        if epoch % 1 == 0:
            print("Epoch %i 后的成本值 : %f" % (epoch+1, avg_cost))
    
    print("Test Total cost:" + str(autoencoder.calc_total_cost(X_test)))

       
if __name__ == '__main__':      
    # 读取MINIST数据
    minist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # 输入数据标准化处理
    X_train, X_test = standard_scale(minist.train.images, minist.test.images)
    n_samples = int(minist.train.num_examples)
    model(X_train, X_test, n_samples, num_epochs=50, batch_size=128, hidden_num=300, 
          learning_rate=0.002, scale=0.01)
        










        
        