# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:41:01 2018

@author: zhu_lin
"""

import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf


"""
# 下载文本数据
"""
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print ('Found and verified', filename)
    else:
        print (statinfo.st_size)
        raise Exception('Failed to verify' + filename + '.Can you get to it with a browser?')
        
    return filename

filename = maybe_download('text8.zip', 31344016)


"""
# 解压文本数据，并使用tf.compat.as_str将数据转换成单词列表
"""
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)
print('Data size', len(words))


"""
# 创建字典
"""
vocabulary_size = 50000
def build_dataset(words):
    # 统计词频
    count = [['UNK',-1]]
    # 取词频top50000的词放入字典
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    
    dictionary = dict()   
    for word,_ in count:
        dictionary[word] = len(dictionary)
     
    # 将字典中的词按词频顺序编码
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    
    # 将字典反转
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
        
        







