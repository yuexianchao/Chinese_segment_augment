# -*- coding: utf-8 -*-
"""
# @Time    : 2018/05/26 下午5:13
# @Update  : 2018/09/28 上午10:30
# @Author  : zhanzecheng/片刻
# @File    : demo.py.py
# @Software: PyCharm
"""
import os
import jieba
from model import TrieNode
from utils import get_stopwords, load_dictionary, generate_ngram, save_model, load_model
from config import basedir
import time
from pyquery import PyQuery
import jieba.analyse
import jieba.posseg  #词性判断

def load_data(filename, stopwords):
    """

    :param filename:
    :param stopwords:
    :return: 二维数组,[[句子1分词list], [句子2分词list],...,[句子n分词list]]
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            word_list = [x for x in jieba.cut(line.strip(), cut_all=False) if x not in stopwords and x.strip() > '']
            if word_list or len(word_list) > 0:
                data.append(word_list)
    if data.__len__() == 0:
        print('no data')
    return data


# from nltk import ngrams
# sentence = 'this is a foo bar sentences and i want to ngramize it'
# n = 6
# sixgrams = ngrams(sentence.split(), n)
# for grams in sixgrams:
#     print(grams)
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


def handel_data(word_lists):
    _ngrams = []
    for word_list in word_lists:
        _ngrams.append(generate_ngram(word_list, 3))
        # print('_ngrams=====',_ngrams)
    if _ngrams:
        return np.concatenate(_ngrams)
    else:
        print('_ngrams=======null', word_lists)
        return [[]]


def load_data_2_root(data):
    print('------> 插入节点')
    pool_cnt = data.__len__()
    # if pool_cnt > 100:
    #     pool_cnt = 100

    dl = np.array_split(data, pool_cnt)
    pool = Pool(24)
    print('poll start')
    ngrams_items = pool.map(handel_data, tqdm(dl))
    pool.close()
    pool.join()
    print('poll stop')
    for ngrams in tqdm(ngrams_items):
        for d in ngrams:
            root.add(d)

    # for word_list in data:
    #     # tmp 表示每一行自由组合后的结果（n gram）
    #     # tmp: [['它'], ['是'], ['小'], ['狗'], ['它', '是'], ['是', '小'], ['小', '狗'], ['它', '是', '小'], ['是', '小', '狗']]
    #     ngrams = generate_ngram(word_list, 3)
    #     for d in ngrams:
    #         root.add(d)
    print('------> 插入成功')


def tf_idf(words, n):
    print('以下是tf-tdf算法------------TF-IDF关键词抽取，需要使用停用词库-------------------------------------')
    keywords_tf = jieba.analyse.extract_tags(words, topK=n, withWeight=True, allowPOS=(
        'ns', 'n', 'vn', 'v'))  # tf-tdf算法
    for item in keywords_tf:
        print(item[0], item[1], jieba.posseg.lcut(item[0])[0])
    return keywords_tf


if __name__ == "__main__":
    global star_time
    star_time = time.time()
    print('开始 ==', star_time)
    root_name = basedir + "/data/root.pkl"
    stopwords = get_stopwords()
    if os.path.exists(root_name):
        root = load_model(root_name)
    else:
        dict_name = basedir + '/data/dict.txt'
        word_freq = load_dictionary(dict_name)
        root = TrieNode('*', word_freq)
        save_model(root, root_name)

    print('读取stopwords  root.pkl==', time.time() - star_time)

    # 加载新的文章
    filename = 'data/demo_old.txt'
    data = load_data(filename, stopwords)
    print('加载新的文章 ==', time.time() - star_time)

    # 将新的文章插入到Root中
    load_data_2_root(data)
    print('将新的文章插入到Root中 ==', time.time() - star_time)

    # 定义取TOP5个
    topN = 10
    result, add_word = root.find_word(topN)
    # 如果想要调试和选择其他的阈值，可以print result来调整
    # print("\n----\n", result)
    print("\n----\n", '增加了 %d 个新词, 词语和得分分别为: \n' % len(add_word))
    print('#############################')
    for word, score in add_word.items():
        print(word + ' ---->  ', score)
    print('#############################')
    print('topN ==', time.time() - star_time)

    # 前后效果对比
    test_sentence = ''.join(open(filename, encoding='UTF-8').readlines())
    print('添加前：')
    words = "".join(
        [(x + ',') for x in jieba.cut(test_sentence, cut_all=False) if x not in stopwords and x.strip() > ''])
    # print(words)
    tf_idf(words, 20)
    print('添加前分词 ==', time.time() - star_time)

    # 添加新词
    for word in add_word.keys():
        jieba.add_word(word)
    print("添加后：")

    words = "".join(
        [(x + ',') for x in jieba.cut(test_sentence, cut_all=False) if x not in stopwords and x.strip() > ''])
    # print(words)
    tf_idf(words, 20)

    print('添加后分词 ==', time.time() - star_time)
