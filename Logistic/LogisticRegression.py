# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

# 导入数据
def load_dataset(trainDataDir,testDataDir):
    train_dataset = h5py.File(trainDataDir, "r")                # 读取训练数据，共1113张图片
    test_dataset = h5py.File(testDataDir, "r")                  # 读取测试数据，共100张图片

    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 原始训练集（1113*128*128*3）
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 原始训练集的标签集（y=0是80年代,y=1是90年代）（1113*1）

    test_set_x_orig = np.array(test_dataset["test_set_x"][:])     # 原始测试集（100*128*128*3)
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])     # 原始测试集的标签集（y=0是80年代,y=1是90年代）（100*1）

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # 原始训练集的标签集设为（1*1113）
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))     # 原始测试集的标签集设为（1*100）

    classes = np.array(test_dataset["list_classes"][:])
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# sigmoid函数
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# 初始化参数w,b（初始化设置为0）
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))   #w为一个dim*1矩阵
    b = 0
    return w, b

# 计算Y_hat（预测值）,成本函数J 以及梯度（dw，db）
def propagate(w, b, X, Y):
    m = X.shape[1]                       # 样本个数
    Y_hat = sigmoid(np.dot(w.T,X) + b)   # 预测值
    cost = -(np.sum(np.dot(Y,np.log(Y_hat).T) + np.dot((1-Y),np.log(1-Y_hat).T)))/m  # 成本函数
    dw = (np.dot(X,(Y_hat - Y).T))/m
    db = (np.sum(Y_hat - Y))/m
    cost = np.squeeze(cost)  # 压缩维度(删除维度为1的维数)
    grads = {"dw": dw,
             "db": db}       # 梯度
    return grads, cost

# 梯度下降找出最优解
# num_iterations-梯度下降次数 learning_rate-学习率，即参数ɑ
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []  # 记录成本值
    for i in range(num_iterations):  # 循环进行梯度下降
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:     # 每100次记录一次成本值
            costs.append(cost)
        if print_cost and i % 100 == 0:  # 打印成本值
            print ("循环%i次后的成本值: %f" % (i, cost))
    params = {"w": w,
              "b": b}   # 最终参数值
    grads = {"dw": dw,
             "db": db}  # 最终梯度值
    return params, grads, costs

# 预测出结果
def predict(w, b, X):
    m = X.shape[1]                      # 样本个数
    Y_prediction = np.zeros((1,m))      # 初始化预测输出
    w = w.reshape(X.shape[0], 1)        # 转置参数向量w
    Y_hat = sigmoid(np.dot(w.T,X) + b)  # 最终得到的参数代入方程
    for i in range(Y_hat.shape[1]):
        if Y_hat[:,i]>0.5:
            Y_prediction[:,i] = 1
        else:
            Y_prediction[:,i] = 0
    return Y_prediction

# 建立整个预测模型
# num_iterations-梯度下降次数 learning_rate-学习率，即参数ɑ
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])  # 初始化参数w，b
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)  # 梯度下降找到最优参数
    # 得到了训练完成的模型参数w,b
    w = parameters["w"]
    b = parameters["b"]
    # 对模型进行检验，即使用训练集和测试集进行预测
    Y_prediction_train = predict(w, b, X_train)     # 训练集的预测结果
    Y_prediction_test = predict(w, b, X_test)       # 测试集的预测结果
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100  # 训练集识别准确度（1-错误率）
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100     # 测试集识别准确度
    print("训练集识别准确度: {} %".format(train_accuracy))
    print("测试集识别准确度: {} %".format(test_accuracy))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d


if __name__ == '__main__':
    # 初始化数据
    trainDataDir = "F:\\imagefile\\hdf5\\train.h5"
    testDataDir = "F:\\imagefile\\hdf5\\test.h5"
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset(trainDataDir,testDataDir)
    m_train = train_set_x_orig.shape[0]  # 训练集中样本个数
    m_test = test_set_x_orig.shape[0]    # 测试集总样本个数
    num_px = test_set_x_orig.shape[1]    # 图片的像素大小
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  # 原始训练集的设为（49152*1113）
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T     # 原始测试集设为（49152*100）
    train_set_x = train_set_x_flatten / 255.  # 将训练集矩阵标准化
    test_set_x = test_set_x_flatten / 255.    # 将测试集矩阵标准化
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=8000, learning_rate=0.001,
              print_cost=True)


