# -*- coding:utf-8 -*-

if __name__ == '__main__':

    input = input('请输入一串字符串:')
    str = list(input)
    str.sort()
    sort_str = {}
    for s in str:
        sort_str[s] = str.count(s)

    keys = list(sort_str.keys())
    values = list(sort_str.values())

    result = []
    for k in range(len(keys)):
        for i in range(len(keys)):
            if values[i] != 0:
                result.append(keys[i])
                values[i] -= 1

    result = ''.join(result)
    print(result)