# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


# 导入数据
def load_dataset(trainDataDir, testDataDir):
    # hdf5 读取数据
    train_dataset = h5py.File(trainDataDir, "r")                # 读取训练数据，共1113张图片
    test_dataset = h5py.File(testDataDir, "r")                  # 读取测试数据，共100张图片
    # 训练集数据
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 原始训练集（1113*128*128*3）
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 原始训练集的标签集（y=0是80年代,y=1是90年代）（1113*1）
    # 测试集数据
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])     # 原始测试集（100*128*128*3)
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])     # 原始测试集的标签集（y=0是80年代,y=1是90年代）（100*1）
    # 将标签数据集转换为行向量
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # 原始训练集的标签集设为（1*1113）
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))     # 原始测试集的标签集设为（1*100）
    classes = np.array(test_dataset["list_classes"][:])
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# 激活函数
# 1、sigmoid函数
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# sigmoid 导数
def sigmoid_prime(z):
    s = sigmoid(z) * (1 - sigmoid(z))
    return s

# 2、tanh 函数
def tanh(z):
    s = (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
    return s

# tanh 导数
def tanh_prime(z):
    s = 1 - np.power(tanh(z), 2)
    return s

# 3、ReLU 函数
def relu(z):
    s = np.maximum(0, z)
    return s

# ReLU 导数
def relu_prime(z):
    s = z
    s[z <= 0] = 0
    s[z > 0] = 1
    return s

# 随机初始化参数w1,b1,w2,b2
def initial_parameters(a0, a1, a2):
    np.random.seed(1)
    w1 = np.random.randn(a1,a0) * 0.01
    b1 = np.zeros((a1,1))
    w2 = np.random.randn(a2,a1) * 0.01
    b2 = np.zeros((a2, 1))
    return w1,b1,w2,b2

# 训练神经网络
# hidden_neuron-隐层神经元个数 num_iterations-梯度下降次数 learning_rate-学习率，即参数ɑ
def train(X_train, Y_train, hidden_neuron, num_iterations, learning_rate, print_cost):
    m = X_train.shape[1]                    # 训练样本数
    a0 = X_train.shape[0]                   # a0 （输入层）特征数
    a1 = hidden_neuron                      # a1 （隐含层）神经元个数
    a2 = Y_train.shape[0]                   # a2 = 1 (输出层) 神经元个数
    w1, b1, w2, b2 = initial_parameters(a0, a1, a2)
    costs = []                              # 存储损失值
    for i in range(num_iterations):
        # 前向传播
        Z1 = np.dot(w1, X_train) + b1
        A1 = tanh(Z1)                       #(a1,m)
        Z2 = np.dot(w2, A1) + b2
        A2 = sigmoid(Z2)                    #(a2,m)
        cost = -(1.0/m) * np.sum(np.dot(Y_train, np.log(A2).T) + np.dot((1-Y_train), np.log(1-A2).T))
        cost = np.squeeze(cost)             # 压缩维度(删除维度为1的维数)

        # 向后传播
        dZ2 = A2 - Y_train
        dw2 = np.dot(dZ2, A1.T)/m
        db2 = np.sum(dZ2, axis=1, keepdims=True)/m

        dZ1 = np.dot(w2.T, dZ2) * tanh_prime(Z1)
        dw1 = np.dot(dZ1, X_train.T)/m
        db1 = np.sum(dZ1, axis=1, keepdims=True)/m

        # 更新权重
        w1 = w1 - learning_rate * dw1
        b1 = b1 - learning_rate * db1
        w2 = w2 - learning_rate * dw2
        b2 = b2 - learning_rate * db2

        if i % 100 == 0:  # 每100次记录一次成本值
            costs.append(cost)
        if print_cost and i % 100 == 0:  # 打印成本值
            print("循环%i次后的成本值: %f" % (i, cost))

    param = {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "costs": costs, "learning_rate": learning_rate}
    return param

# 模型预测
def pridect(X, Y, param, datatype):
    w1 = param["w1"]
    b1 = param["b1"]
    w2 = param["w2"]
    b2 = param["b2"]
    m = X.shape[1]                  # 测试集样本数
    Y_prediction = np.zeros((1,m))  # 初始化测试结果
    Z1 = np.dot(w1, X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(w2, A1) + b2
    Y_hat = sigmoid(Z2)             # 预测值
    for i in range(Y_hat.shape[1]):
        if Y_hat[:, i] > 0.5:
            Y_prediction[:, i] = 1
        else:
            Y_prediction[:, i] = 0

    # 识别准确度
    accuracy = 100 - np.mean(np.abs(Y_prediction - Y)) * 100
    if datatype == 0:
        print("训练集识别准确度: {} %".format(accuracy))
    else:
        print("测试集识别准确度: {} %".format(accuracy))
    return Y_prediction

# 绘制成本曲线
def plotcost(param):
    plt.plot(param["costs"])
    plt.ylabel('cost')
    plt.xlabel('Iteration(per hundreds)')
    plt.title("Learning rate=" + str(param["learning_rate"]))
    plt.show()

if __name__ == '__main__':
    # 初始化数据
    trainDataDir = "F:\\imagefile\\hdf5\\train.h5"
    testDataDir = "F:\\imagefile\\hdf5\\test.h5"
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset(trainDataDir, testDataDir)

    # 获取数据集参数
    m_train = train_set_x_orig.shape[0]  # 训练集中样本个数
    m_test = test_set_x_orig.shape[0]    # 测试集总样本个数
    num_px = test_set_x_orig.shape[1]    # 图片的像素大小

    # 标准化数据集
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  # 原始训练集转置设为（49152*1113）
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T     # 原始测试集转置设为（49152*100）
    train_set_x = train_set_x_flatten / 255.  # 将训练集矩阵标准化
    test_set_x = test_set_x_flatten / 255.    # 将测试集矩阵标准化

    # 训练模型
    model = train(train_set_x, train_set_y, hidden_neuron=20, num_iterations=4000, learning_rate=0.005, print_cost=True)

    # 测试模型
    train_y_hat = pridect(train_set_x, train_set_y, model, datatype=0)
    test_y_hat = pridect(test_set_x, test_set_y, model, datatype=1)

    # 绘制成本曲线
    plotcost(model)
