import random
import numpy as np


# 从txt中导入数据
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readline():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

'''
# 随机选取αj
i 是αi 的下标  m是α参数的总数目
'''
def selectJrand(i,m):
    j = i
    while(j==i):
        j = random.uniform(0,m)
    return j

'''
# 裁剪aj的值，确保更新后的aj在（0<aj<C）内
'''
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0

    while(iter < maxIter):
        alphaPairsChanged = 0      # 用于判断aj 是否优化更新
        for i in range(m):
            # 计算f(xi) = Σai*yi*<x,xi> + b
            fXi = float(np.multiply(alphas, dataMatrix).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # 1、计算预测值与真实值之间误差
            Ei = fXi - float(labelMat[i])
            # 不满足KKT条件，即错分的点的ai需要被更新（yi*fxi<1-e）
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, dataMatrix).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 确保更新后的aj必须满足KKT条件 (0<ai<C)
                if (labelMat[i] == labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H: print('L==H'); continue
                # 2、更新aj
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0: print('eta >= 0'); continue
                alphas[j] -= labelMat[j]*(Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print('变化太小，不做更新')
                    continue
                # 更新ai,修改量相同，但方向相反，不然 ai+ aj != 原来的值
                alphas[i] += labelMat[i] * labelMat[j] * (alphaJold - alphas[j])
                # 3、计算b的值
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                # 关于b的取值，取ai,aj满足KKT条件的值，否则取均值
                if (alphas[i] > 0) and (alphas[i] < C):
                    b = b1
                elif (alphas[j] > 0) and (alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print('iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))

        # 只有当所有样本都遍历了maxIter次，且alpha不再更新了，说明更新完毕，退出循环
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas










