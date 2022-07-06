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
    def freq_analyze(data_frame, field_col='', target_col='', barmode='stack'):
        """
        频率分布分析
        :param data_frame:数据源
        :param field_col:需要分析的字段
        :param target_col:分类字段
        :param barmode:统计条显示方式
        :return:
        """
        if field_col is not '':
            data = []
            if target_col is not '':
                class_tag = list(data_frame[target_col].drop_duplicates())
                for cls in class_tag:
                    freq_data_part = eval('data_frame[data_frame.%s == %s]' %
                                          (target_col, cls))
                    freq_data = pd.DataFrame(freq_data_part[field_col].
                                             value_counts()).reset_index()
                    data.append((freq_data['index'], freq_data[field_col]))

                ChartTools.plot_bar(data=data,
                                    traces=class_tag,
                                    title="Frequency_Analyze",
                                    barmode=barmode)
            else:
                freq_data = pd.DataFrame(
                    data_frame[field_col].value_counts()).reset_index()
                data.append((freq_data['index'], freq_data[field_col]))
                ChartTools.plot_bar(data=data,
                                    traces=[field_col],
                                    title="Frequency_Analyze",
                                    barmode=barmode)

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
    def show_feature_importance(feature_list,
                                importance_list,
                                return_importance_df=False):
        """
        统计特征重要性
        :param feature_list:特征名列表
        :param importance_list:特征重要性列表（与特征名列表同序）
        :param return_importance_df:是否返回重要性表
        :return: feature_df
        """

        feature_df = pd.DataFrame()
        feature_df['feature'] = feature_list
        feature_df['importance'] = importance_list
        feature_df.sort_values(by=['importance'], ascending=True, inplace=True)

        ChartTools.plot_bar(
            [(feature_df['importance'], feature_df['feature'])],
            traces=['importance'],
            title='feature_importance',
            orientation='h')

        if return_importance_df:
            return feature_df

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
    def show_null_feature_rank(data_frame):
        """
        显示包含空值特征排行
        :param data_frame:pandas df
        :return:pandas series
        """
        temp = data_frame.isnull().sum()
        return temp[temp > 0].sort_values(ascending=False)

    @staticmethod
    def dummies(data_frame, cols, drop_cols=True):
        """
        对指定列进行哑编码
        :param data_frame:
        :param cols:
        :param drop_cols: 是否删除原列
        :return: 哑编码后的数据集
        """
        for col in cols:
            dum = pd.get_dummies(data_frame[col], prefix=col)
            data_frame = pd.concat([data_frame, dum], axis=1)
            if drop_cols:
                data_frame.drop(col, axis=1, inplace=True)
        return data_frame

    @staticmethod
    def z_scores_std(data_frame, cols):
        """
        Z-Scores标准化
        :param data_frame:
        :param cols:
        :return: 标准化的数据集
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
        离群点去噪
        :param data_frame:
        :param cols:
        """
        for col in cols:
            data_frame.loc[:, col] = data_frame[col].astype(float)
            data_frame.loc[:, col] = (data_frame.loc[:, col] - np.min(
                data_frame.loc[:, col])) / np.ptp(data_frame.loc[:, col])

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