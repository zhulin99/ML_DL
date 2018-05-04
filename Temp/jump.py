# -*- coding:utf-8 -*-
def jump(end, num):
    if end == 0:
        return 0
    i = 0
    while num[i] < (end - i):
        i += 1
    return 1 + jump(i,num)


if __name__ == '__main__':
    num_list = [2,3,2,1,2,1,5]
    n = len(num_list)
    print(jump(n-1, num_list))




