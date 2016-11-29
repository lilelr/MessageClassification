# coding=utf-8
# Create by Dotomato(ChenJun)

import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def loaddataset(file_name):
    dataMat = []
    labelMat = []
    data = open(file_name).readlines()
    data = data[0:1000]
    for line in data:
        lineArr = line.strip().split('\t')
        labelMat.append(lineArr[0])
        dataMat.append(lineArr[1])
    return labelMat, dataMat


def cutdata(dataMat):
    lines = []
    for line in dataMat:
        kv = ' '.join(jieba.cut(line))
        # kv = jieba.cut(line)
        # print(kv)
        lines.append(kv)
    return lines


# 载入训练集
labelMat, dataMat = loaddataset('train.txt')

# 使用jieba库进行中文分词
dataMat = cutdata(dataMat)

# 计算每个词的出现频率
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(dataMat)

# 计算tf-idf矩阵
tfidf_transformer = TfidfTransformer();
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# TODO 实现SVM分类
