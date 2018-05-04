# -*- coding:utf-8 -*-

if __name__ == '__main__':
    str = input()
    str_list = list(str)
    result_list = []

    for str in str_list:
        if str not in result_list:
            result_list.append(str)

    result = "".join(result_list)
    print(result)