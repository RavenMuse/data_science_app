# -*- encoding:utf-8 -*-
import os
import csv
import json
import h5py
import itertools
import numpy as np
import pandas as pd
import plotly
import operator
from sklearn.cluster import KMeans
from datetime import *
from plotly.graph_objs import *
from scipy import sparse
from scipy import stats


class FeatureFunction:
    """
    数据分析方法
    包括: freq_analyze,chi_merge,show_null_feature_rank,dummies;z_scores_std,show_confusion_matrix
    """

    @staticmethod
    def chi2_test(array, min_expected_value):
        """
        对观测结果做卡方校验
        :param min_expected_value:期望值
        :param array: np.array 2 consecutive rows from frequeny attribute/class matrix, e.g.,: a = np.matrix('16 0 0; 4 1 1')
        :return chisqr value of distribution of 2 rows
        """

        shape = array.shape
        N = float(array.sum())  # total number of observations

        r = {}
        for i in range(shape[0]):
            r[i] = array[i].sum()

        c = {}
        for j in range(shape[1]):
            c[j] = array[:, j].sum()

        chi2 = 0
        for row in range(shape[0]):
            for col in range(shape[1]):
                e = r[row] * c[col] / N  # expected value
                o = array[row, col]  # observed value
                e = min_expected_value if e < min_expected_value else e
                chi2 += 0. if e == 0. else np.math.pow((o - e), 2) / float(e)

        return chi2

    @staticmethod
    def show_confusion_matrix(y_predict, y_true, col_name, normalize=False):
        """
        绘制并返回混淆矩阵
        :param y_predict:
        :param y_true:
        :param col_name:分类列名
        :param normalize:是否归一化
        :return: confusion matrix
        """
        cnf_matrix = confusion_matrix(y_true, y_predict)
        np.set_printoptions(precision=2)

        if normalize:
            cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(
                axis=1)[:, np.newaxis]
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix"

        x = list(map(lambda i: "predict_%s" % i, col_name))
        y = list(map(lambda i: "true_%s" % i, col_name))
        ChartTools.plot_heat_map((x, y, cnf_matrix), title)
        return cnf_matrix

    @staticmethod
    def number_transform(data_frame, col, coff, strategy='boxcox'):
        """数值变换

        Args:
            data_frame (_type_): _description_
            col (_type_): _description_
            coff (_type_): _description_
            strategy (str, optional): _description_. Defaults to 'boxcox'.
        """
        if strategy == 'move':
            data_frame.loc[:, col + '_move'] = data_frame[col] + coff
        if strategy == 'scale':
            data_frame.loc[:, col + '_scale'] = data_frame[col] * coff
        if strategy == 'boxcox':
            if data_frame[col].min() <= 0:
                pre_process = data_frame[col] + np.abs(
                    data_frame[col].min()) + 0.0001
            else:
                pre_process = data_frame[col]
            data_frame.loc[:, col + '_bxcx'] = stats.boxcox(pre_process, coff)

    @staticmethod
    def binning(data_frame, col, bin_num, strategy='width'):
        """分箱

        Args:
            data_frame (pandas dataframe): 数据
            col (str): 列
            bins_num (int): 分箱数  
            strategy (str, optional): 分箱策略('quantile','width'). 默认是等宽：'width'.
        """

        if strategy == 'width':
            bins = pd.cut(data_frame[col], bins=bin_num)
            data_frame.loc[:, col + '_binno'] = pd.factorize(bins,
                                                             sort=True)[0]
            data_frame.loc[:, col + '_bin'] = bins.astype(str)
        if strategy == 'quantile':
            bins = pd.qcut(data_frame[col], bin_num)
            data_frame.loc[:, col + '_binno'] = pd.factorize(bins,
                                                             sort=True)[0]
            data_frame.loc[:, col + '_bin'] = bins.astype(str)
        if strategy == 'cluster':
            kmeans = KMeans(n_clusters=bin_num, random_state=0)

            cluster = kmeans.fit_predict(data_frame[[col]].values)
            data_frame.loc[:, col + '_binno'] = cluster

    @staticmethod
    def fill_na(data_frame, cols, strategy='mean'):
        """
        空值填充
        :param data_frame:
        :param cols:
        :param type:填充方式
        """
        for col in cols:

            if strategy == 'mean':
                data_frame[col].fillna(data_frame[col].mean(), inplace=True)
            if strategy == 'median':
                data_frame[col].fillna(data_frame[col].median(), inplace=True)
            if strategy == 'most_frequent':
                data_frame[col].fillna(data_frame[col].value_counts().index[0],
                                       inplace=True)

    @staticmethod
    def dummies(data_frame, cols):
        """
        对指定列进行哑编码
        :param data_frame:
        :param cols:
        :param drop_cols: 是否删除原列
        :return: 哑编码后的数据集
        """
        for col in cols:
            dum_df = pd.get_dummies(data_frame[col], prefix=col)
            data_frame[dum_df.columns] = dum_df

    @staticmethod
    def z_scores_std(data_frame, cols):
        """
        Z-Scores标准化
        :param data_frame:
        :param cols:
        """
        for col in cols:
            data_frame.loc[:, col] = data_frame[col].astype(float)
            data_frame.loc[:, col] -= data_frame[col].mean()
            data_frame.loc[:, col] /= data_frame[col].std()

    @staticmethod
    def max_min_std(data_frame, cols):
        """
        Max-Min标准化
        :param data_frame:
        :param cols:
        """
        for col in cols:
            data_frame.loc[:, col] = data_frame[col].astype(float)
            data_frame.loc[:, col] = (data_frame.loc[:, col] - np.min(
                data_frame.loc[:, col])) / np.ptp(data_frame.loc[:, col])

    @staticmethod
    def outlier_denoising(data_frame, cols):
        """
        离群点降噪
        :param data_frame:
        :param cols:
        """
        for col in cols:
            val = data_frame[col].astype(float)
            q1 = np.quantile(val, q=0.25)
            q3 = np.quantile(val, q=0.75)
            up = q3 + 1.5 * (q3 - q1)
            low = q1 - 1.5 * (q3 - q1)
            data_frame.loc[(data_frame[col] > up) | (data_frame[col] < low),
                           col] = None

    @staticmethod
    def similarity(a, b, n=4):
        """
        基于n-grams的jaccard相似度
        :param a:
        :param b:
        :param n:
        :return:
        """

        a = set([a[i:i + n] for i in range(len(a) - n)])
        b = set([b[i:i + n] for i in range(len(b) - n)])
        a_and_b = a & b
        if not a_and_b:
            return 0.
        a_or_b = a | b
        return 1. * len(a_and_b) / len(a_or_b)