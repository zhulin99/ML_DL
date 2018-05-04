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
# 1、sigmoid 函数
def sigmoid(Z):
    s = 1.0 / (1.0 + np.exp(-Z))
    return s

def sigmoid_prime(Z):
    s = sigmoid(Z) * (1 - sigmoid(Z))
    return s

# 2、tanh 函数
def tanh(x):
    s = (np.exp(x) - np.exp(-x)) / ((np.exp(x) + np.exp(-x)))
    return s

def tanh_prime(x):
    s = 1 - np.power(tanh(x), 2)
    return s

# 3、relu 函数
def relu(Z):
    s = np.maximum(0, Z)
    return s

def relu_prime(Z):
    s = Z
    s[Z <= 0] = 0
    s[Z > 0] = 1
    return s

# 初始化参数
def initial_weights(layer):
    np.random.seed(2)
    L = len(layer)
    parameters = {}
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(layer[l], layer[l-1])*np.sqrt(2.0/layer[l-1])
        parameters["b"+str(l)] = np.zeros((layer[l],1))
    return parameters

# 训练模型
def train(X, Y, layer, itr_num, learning_rate,  isprint=True):
    m = Y.shape[1]                          # 训练集数量
    Fcahe = {}                              # 缓存前向传播计算过程中变量(Z,A)
    Bcahe = {}                              # 缓存反向传播计算过程中变量(dZ,dW,db)
    costs = []

    Fcahe["A"+str(0)] = X                   # 初始化 A0=X
    parameters = initial_weights(layer)     # 初始化参数
    L = len(parameters) // 2                # 获取网络层数

    for itr in range(itr_num):
        # 前向传播
        for l in range(1,L):
            Fcahe["Z"+str(l)] = np.dot(parameters["W"+str(l)],  Fcahe["A"+str(l-1)]) + parameters["b"+str(l)]
            Fcahe["A"+str(l)] = relu(Fcahe["Z"+str(l)])
        Fcahe["Z"+str(L)] = np.dot(parameters["W"+str(L)], Fcahe["A"+str(L-1)]) + parameters["b"+str(L)]
        Fcahe["A"+str(L)] = sigmoid(Fcahe["Z"+str(L)])
        cost = -(1.0 / m) * np.sum(Y * np.log(Fcahe["A" + str(L)]) + (1 - Y) * np.log(1 - Fcahe["A" + str(L)]))
        cost = np.squeeze(cost)

        # 反向传播
        Bcahe["dZ"+str(L)] = (-(np.divide(Y,Fcahe["A"+str(L)])-np.divide(1-Y,1-Fcahe["A"+str(L)]))) * sigmoid_prime(Fcahe["Z"+str(L)])
        Bcahe["dW"+str(L)] = (1.0/m) * np.dot(Bcahe["dZ"+str(L)], Fcahe["A"+str(L-1)].T)
        Bcahe["db"+str(L)] = (1.0/m) * np.sum(Bcahe["dZ"+str(L)], axis=1, keepdims=True)
        for l in reversed(range(L-1)):
            Bcahe["dZ"+str(l+1)] = np.dot(parameters["W"+str(l+2)].T, Bcahe["dZ"+str(l+2)]) * relu_prime(Fcahe["Z"+str(l+1)])
            Bcahe["dW"+str(l+1)] = (1.0/m) * np.dot(Bcahe["dZ"+str(l+1)], Fcahe["A"+str(l)].T)
            Bcahe["db"+str(l+1)] = (1.0/m) * np.sum(Bcahe["dZ"+str(l+1)], axis=1, keepdims=True)

        # 更新参数
        for l in range(L):
            parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*Bcahe["dW"+str(l+1)]
            parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*Bcahe["db"+str(l+1)]

        if isprint and itr%200 == 0:
            print('cost after ' + str(itr) + ' iteration is :' + str(cost))
            costs.append(cost)

    # 绘制成本曲线
    plt.plot(np.squeeze(costs))
    plt.xlabel("itr_num(per hundred)")
    plt.ylabel("cost")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

# 模型测试
def predict(X,Y,parameters):
    num = X.shape[1]
    pred = np.zeros((1, num))
    L = len(parameters) // 2
    A0 = X
    for l in range(1, L):
        Z = np.dot(parameters["W" + str(l)], A0) + parameters["b" + str(l)]
        A = relu(Z)
        A0 = A
    Z = np.dot(parameters["W" + str(L)], A0) + parameters["b" + str(L)]
    A = sigmoid(Z)

    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            pred[0, i] = 0
        else:
            pred[0, i] = 1
    print("accuracy:{}%".format(100 - np.mean(np.abs(pred - Y)) * 100))
    return pred

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

    # 定义网络层数及神经元个数
    n_x = train_set_x_flatten.shape[0]
    layer_dims = [n_x, 10, 5, 3, 1]
    # 训练网络
    parameters = train(train_set_x, train_set_y, layer_dims, itr_num=5000, learning_rate=0.005, isprint=True)
    # 测试模型
    train_y_hat = predict(train_set_x, train_set_y, parameters)
    test_y_hat = predict(test_set_x, test_set_y, parameters)