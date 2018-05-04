# -*- coding:utf-8 -*-
import sys

if __name__ == '__main__':

    list = []
    list_new = [] #定义一个空列表
    for line in sys.stdin:
        # py.3中input（）只能输入一行  sys.stdin按下换行键然后ctrl+d程序结束
        list_new = line.split()
        list.extend(list_new)      # 每一行组成的列表合并
    print(list)

    sentinel = 'end'  # 遇到这个就结束
    lines = []
    for line in iter(input, sentinel):
        lines.append(line)

