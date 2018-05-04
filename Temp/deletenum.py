# -*- coding:utf-8 -*-

if __name__ == '__main__':
    input_num = input()
    n = int(input_num)
    num_list = []
    for i in range(0,n):
        num_list.append(i)

    i = 0
    while len(num_list) != 1:
        if i != 2:
            pop = num_list[0]
            num_list.pop(0)
            num_list.append(pop)
            i += 1
        else:
            num_list.pop(0)
            i = 0
    result = num_list[0]
    print(result)







