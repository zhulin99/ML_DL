# -*- coding:utf-8 -*-

def isSushu(num):
    if num > 1:
        for i in range(2, num):
            if (num % i) == 0:
                return 0
        else:
            return 1
    else:
        return 0


if __name__ == '__main__':
    q = 11
    q_list = [1, 2, 3, 10, 15, 30, 50, 65, 80, 95, 100]

    for i in range(q):
        k = q_list[i]
        i = 1
        s = 0
        while True:
            if isSushu(i):
                s += 1
                if s == k:
                    print(str(i) + '\n')
                    break
                else:
                    i += 1
            else:
                i += 1


