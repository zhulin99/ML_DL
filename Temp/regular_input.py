# -*- coding:utf-8 -*-
import numpy as np


def regular_input():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.empty([1, a.shape[1]], dtype=float)
    for i in range(a.shape[1]):
        va = np.mean((a[:, i] - a[:, i].mean()) ** 2)
        b[:, i] = va
    a = a / b
    print(a)

def testRsndomSeed():
    np.random.seed(1)
    for i in range(5):
        a = np.random.randn(1,5)
        print(a)

def testPermutation():
    np.random.seed(0)
    permutation = list(np.random.permutation(10))
    permutation2 = np.random.permutation(10)
    print (permutation)
    print (permutation2)



if __name__ == '__main__':
    # regular_input()
    # testRsndomSeed()
    testPermutation()




