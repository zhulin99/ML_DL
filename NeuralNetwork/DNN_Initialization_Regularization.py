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
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

# sigmoid 导数
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

# 2、ReLU 函数
def relu(Z):
    A = np.maximum(0, Z)
    return A

# ReLU 导数
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

# 随机初始化参数（Xavier）
# 防止梯度消失/梯度爆炸
def initial_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        # np.sqrt(2.0 / layer_dims[l - 1])  防止梯度消失/梯度爆炸
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2.0 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))  # 初始化为0
    return parameters

# 向前传播线性计算
def linear_activation_forward(A_input, W, b, activation):
    if activation == "sigmoid":
        Z = np.dot(W, A_input) + b
        linear_cache = (A_input, W)        # 缓存A(l-1)、Wl
        A = sigmoid(Z)
        activation_cache = Z               # 缓存Zl
    elif activation == "relu":
        Z = np.dot(W, A_input) + b
        linear_cache = (A_input, W)
        A = relu(Z)
        activation_cache = Z
    cache = (linear_cache, activation_cache)
    return A, cache

# 向前传播线性计算Dropout
def linear_activation_forward_with_dropout(A_input, W, b, keep_prob, activation):
    if activation == "sigmoid":
        Z = np.dot(W, A_input) + b
        linear_cache = (A_input, W)  # 缓存A(l-1)、Wl
        A = sigmoid(Z)
        D = np.random.rand(A.shape[0], A.shape[1])
        D = D < keep_prob         # D(Dropout)为该层的Dropout矩阵
        A = np.multiply(A, D)
        A = A / keep_prob
        activation_cache = Z  # 缓存Zl
        dropout_cache = D     # 缓存Dl
    elif activation == "relu":
        Z = np.dot(W, A_input) + b
        linear_cache = (A_input, W)
        A = relu(Z)
        D = np.random.rand(A.shape[0], A.shape[1])
        D = D < keep_prob
        A = np.multiply(A, D)
        A = A / keep_prob
        activation_cache = Z
        dropout_cache = D  # 缓存Dl
    cache = (linear_cache, activation_cache, dropout_cache)
    return A, cache

# 向前传播
def train_model_forward(X, parameters):
    caches = []
    A = X                     # 初始化第一层的输入X
    L = len(parameters) // 2  # 层数
    # 计算隐含层AL
    for l in range(1, L):
        A_input = A
        A, cache = linear_activation_forward(A_input, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)
    # 计算输出层AL（y_hat）
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    assert (AL.shape == (1, X.shape[1]))
    return AL, caches

# 向前传播（Dropout）
def train_model_forward_with_dropout(X, parameters, keep_prob):
    caches = []
    A = X  # 初始化第一层的输入X
    L = len(parameters) // 2  # 层数
    np.random.seed(1)
    # 计算隐含层AL
    for l in range(1, L):
        A_input = A
        A, cache = linear_activation_forward_with_dropout(A_input, parameters['W' + str(l)], parameters['b' + str(l)],
                                                          keep_prob, activation="relu")
        caches.append(cache)
    # 计算输出层AL（y_hat）
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    assert (AL.shape == (1, X.shape[1]))
    return AL, caches

# 计算损失函数
def compute_cost(Y_hat, Y):
    m = Y.shape[1]
    cost = (1.0/m) * (-np.dot(Y,np.log(Y_hat).T) - np.dot(1-Y, np.log(1-Y_hat).T))
    cost = np.squeeze(cost)
    return cost

# 计算损失函数(L2正则化)
def compute_cost_with_L2Regularization(Y_hat, Y, parameters, lambd):
    L = len(parameters) // 2  # 层数
    L2 = 0
    m = Y.shape[1]
    cross_entropy_cost = compute_cost(Y_hat, Y)
    for l in range(L):
        L2 +=  np.sum(np.square(parameters["W" + str(l + 1)]))
    L2_regularization_cost = lambd / (2 * m) * L2
    cost = cross_entropy_cost + L2_regularization_cost
    return cost

# 反向传播求梯度
def linear_backward(dZ, cache, current_layer):
    A_prev, W = cache            # A_prev = A(l-1)
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    # 当计算到第一层时，前一层为A0=X，不需要计算dA_prev，否则会报错
    if current_layer != 0:
        dA_prev = np.dot(W.T, dZ)
    else:
        dA_prev = None
    return dA_prev, dW, db

# 反向传播求梯度(L2 Regularization)
def linear_backward_with_L2(dZ, cache, current_layer,  lambd):
    A_prev, W = cache            # A_prev = A(l-1)
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T)/m + lambd/m * W
    db = np.sum(dZ, axis=1, keepdims=True)/m
    # 当计算到第一层时，前一层为A0=X，不需要计算dA_prev，否则会报错
    if current_layer != 0:
        dA_prev = np.dot(W.T, dZ)
    else:
        dA_prev = None
    return dA_prev, dW, db

# 反向传播梯度计算
def linear_activation_backward(dA, cache, current_layer, activation):
    # linear_cache 缓存：A(l-1)、W
    # activation_cache 缓存：Zl
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, current_layer)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, current_layer)
    return dA_prev, dW, db

# 反向传播梯度计算（Dropout）
def linear_activation_backward_with_dropout(dA, cache, current_layer, keep_prob, activation):
    # linear_cache 缓存：A(l-1)、W
    # activation_cache 缓存：Zl
    # dropout_cache 缓存Dl
    linear_cache, activation_cache, dropout_cache = cache
    if activation == "relu":
        dA = np.multiply(dA, dropout_cache)
        dA = dA / keep_prob
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, current_layer)
    elif activation == "sigmoid":
        dA = np.multiply(dA, dropout_cache)
        dA = dA / keep_prob
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, current_layer)
    return dA_prev, dW, db

# 反向传播梯度计算（L2Regularization）
def linear_activation_backward_with_L2Regularization(dA, cache, current_layer, lambd, activation):
    # linear_cache 缓存：A(l-1)、W
    # activation_cache 缓存：Zl
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_with_L2(dZ, linear_cache, current_layer, lambd)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_with_L2(dZ, linear_cache, current_layer, lambd)
    return dA_prev, dW, db

# 反向传播
def train_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)                 # 网络层数
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    # 计算输出层的梯度
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                    L, activation="sigmoid")
    # 计算隐层的梯度
    for l in reversed(range(L - 1)):
        current_layer = l
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, current_layer, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

# 反向传播（Dropout）
def train_model_backward_with_dropout(AL, Y, caches, keep_prob):
    grads = {}
    L = len(caches)  # 网络层数
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    # 计算输出层的梯度
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                      L, activation="sigmoid")
    # 计算隐层的梯度
    for l in reversed(range(L - 1)):
        current_layer = l
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward_with_dropout(grads["dA" + str(l + 1)], current_cache,
                                                                    current_layer, keep_prob, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

# 反向传播（L2 Regularization）
def train_model_backward_with_regularization(AL, Y, caches, lambd):
    grads = {}
    L = len(caches)  # 网络层数
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    # 计算输出层的梯度
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads[
        "db" + str(L)] = linear_activation_backward_with_L2Regularization(dAL, current_cache, L, lambd, activation="sigmoid")

    # 计算隐层的梯度
    for l in reversed(range(L - 1)):
        current_layer = l
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward_with_L2Regularization(grads["dA" + str(l + 1)],
                                                                                          current_cache,
                                                                                          current_layer,
                                                                                          lambd,
                                                                                          activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

# 更新参数
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return parameters

# 训练网络参数
def train_model(X, Y, layer_dims, lambd, keep_prob, num_iterations, learning_rate, print_cost=True):
    costs = []
    # 初始化参数
    parameters = initial_parameters(layer_dims)

    for i in range(num_iterations):
        # 向前传播
        if keep_prob==1:
            AL, caches = train_model_forward(X, parameters)
        elif keep_prob<1:
            AL, caches = train_model_forward_with_dropout(X, parameters, keep_prob)

        # 损失函数
        if lambd==0:
            cost = compute_cost(AL, Y)
        else:
            cost = compute_cost_with_L2Regularization(AL, Y, parameters, lambd)

        # 向后传播
        if lambd == 0 and keep_prob == 1:
            grads = train_model_backward(AL, Y, caches)
        elif keep_prob < 1:
            grads = train_model_backward_with_dropout(AL, Y, caches, keep_prob)    # DropOut正则化
        elif lambd != 0:
            grads = train_model_backward_with_regularization(AL, Y, caches, lambd)  # L2正则化

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("循环%i次后的成本值: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # 绘制成本曲线
    plotcost(np.squeeze(costs), learning_rate)
    return parameters

# 绘制成本曲线
def plotcost(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('Iteration(per hundreds)')
    plt.title("Learning rate=" + str(learning_rate))
    plt.show()

# 模型预测
def predict(X, y, parameters):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    Y_hat, caches = train_model_forward(X, parameters)

    for i in range(0, Y_hat.shape[1]):
        if Y_hat[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    print("预测准确度: " + str(np.sum((Y_prediction == y) / m)))
    return Y_prediction

# 输入数据正则化(归一化特征值)
def regularization_input(inputdata):
    m = inputdata.shape[1]
    variance = np.empty([1, m], dtype=float)
    for i in range(m):
        va = np.mean((inputdata[:, i] - inputdata[:, i].mean()) ** 2)
        variance[:, i] = va
    result = inputdata/variance
    return result


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
    # train_set_x = train_set_x_flatten / 255.  # 将训练集矩阵标准化
    # test_set_x = test_set_x_flatten / 255.    # 将测试集矩阵标准化

    # 正则化输入数据
    train_set_x = regularization_input(train_set_x_flatten)
    test_set_x = regularization_input(test_set_x_flatten)

    # 定义网络层数及神经元个数
    n_x = train_set_x_flatten.shape[0]
    layer_dims = [n_x, 20, 7, 5, 1]

    # 训练网络
    parameters = train_model(train_set_x, train_set_y, layer_dims, lambd=0.8, keep_prob=1, num_iterations=7000,
                             learning_rate=0.01, print_cost=True)

    # 测试模型
    train_y_hat = predict(train_set_x, train_set_y, parameters)
    test_y_hat = predict(test_set_x, test_set_y, parameters)
