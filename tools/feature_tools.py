import streamlit as st
from scipy import stats
import pandas as pd
import numpy as np
# import plotly.express as px
# import time
from .tools import Tools
import sys

sys.path.append("..")
from func.feature_func import FeatureFunction as ff


class FeatureTools(Tools):

    def __init__(self):
        super().__init__()
        self.tools_name = '特征工具'

    def use_tool(self, data):
        return super().use_tool(data)

    def layout_menu(self):

        ex = st.sidebar.expander('特征工具', True)
        ex.markdown("##### 数值处理")

        ex.checkbox("标准化", key='standardize')
        self.add_tool_func('standardize', self.standardize)
        ex.checkbox("降噪", key='denoising')
        self.add_tool_func('denoising', self.denoising)
        ex.checkbox("空值填充", key='fill_na')
        self.add_tool_func('fill_na', self.fill_na)

        ex.markdown("##### 数据变换")
        ex.checkbox("数值变换", key='number_transform')
        self.add_tool_func('number_transform', self.number_transform)
        ex.checkbox("降维", key='dim_reduction')
        self.add_tool_func('dim_reduction', self.dim_reduction)
        ex.checkbox("离散化", key='discretization')
        self.add_tool_func('discretization', self.discretization)
        ex.checkbox("数值化", key='numeric')
        self.add_tool_func('numeric', self.numeric)

    def __multi_select_with_value(self, label, options, func, help_str=None):
        col1, col2 = st.columns(2)
        cols = col1.multiselect(label, options)
        values = col2.text_input('',
                                 key=label + '_text',
                                 placeholder='2,3,4',
                                 help=help_str).split(',')
        if len(cols) == 0:
            return
        else:
            for i, col in enumerate(cols):
                if i > len(values) - 1:
                    continue
                val = values[i]
                try:
                    val = float(values[i])
                except:
                    st.error('参数必须为数值！')
                    return
                func(col, val)

    def fill_na(self, data):
        with st.expander('空值填充', True):
            numeric_cols = data.select_dtypes(exclude=['object']).columns
            object_cols = data.select_dtypes(include=['object']).columns

            col1, col2, col3 = st.columns([0.3, 0.3, 0.3])
            cols = col1.multiselect('均值填充', numeric_cols)
            ff.fill_na(data, cols, strategy='mean')
            cols = col2.multiselect('中位数填充', numeric_cols)
            ff.fill_na(data, cols, strategy='median')

            if not object_cols.empty:
                cols = col3.multiselect('高频值填充', object_cols)
                ff.fill_na(data, cols, strategy='most_frequent')

            fill_str = st.text_input('自定义填充', help="输入json字符串，例如:{'id':0}")
            if fill_str:
                try:
                    data.fillna(value=eval(fill_str), inplace=True)
                except:
                    st.error('json格式错误')

    def standardize(self, data):
        numeric_cols = data.select_dtypes(exclude=['object']).columns
        with st.expander('标准化', True):
            col1, col2, col3 = st.columns([0.3, 0.3, 0.3])
            cols = col1.multiselect('Z-Scores标准化', numeric_cols)
            ff.z_scores_std(data, cols)

            cols = col2.multiselect('Max-Min标准化', numeric_cols)
            ff.max_min_std(data, cols)

    def denoising(self, data):
        numeric_cols = data.select_dtypes(exclude=['object']).columns
        with st.expander('降噪', True):
            cols = st.multiselect('离群点降噪', numeric_cols)
            ff.outlier_denoising(data, cols)

    def number_transform(self, data):
        with st.expander('数值变换', True):

            numeric_cols = data.select_dtypes(exclude=['object']).columns
            st.write('平移变换')
            st.latex(r'''f(x,\beta) = x + \beta''')
            self.__multi_select_with_value(
                '维度-平移系数',
                numeric_cols,
                lambda col, coff: ff.number_transform(
                    data, col, coff, strategy='move'),
                help_str='平移系数，逗号分隔')

            numeric_cols = data.select_dtypes(exclude=['object']).columns
            st.write('缩放变换')
            st.latex(r'''f(x,\alpha) = x * \alpha''')
            self.__multi_select_with_value(
                '维度-缩放系数',
                numeric_cols,
                lambda col, coff: ff.number_transform(
                    data, col, coff, strategy='scale'),
                help_str='平移系数，逗号分隔')

            numeric_cols = data.select_dtypes(exclude=['object']).columns
            st.write('box-cox变换')
            st.latex(r'''f(x,\lambda) = \begin{cases}
                        \frac{x^\lambda-1}{\lambda}, & \lambda\neq0 \\
                        \ln x,& \lambda=0 \\
                        \end{cases}''')
            self.__multi_select_with_value(
                '维度-变换系数',
                numeric_cols,
                lambda col, coff: ff.number_transform(
                    data, col, coff, strategy='boxcox'),
                help_str=
                "变换系数，逗号分隔；对数变换：lambda=0，倒数变换：lambda=-1，平方根变换：lambda=0.5")

    def numeric(self, data):
        object_cols = data.select_dtypes(include=['object']).columns
        with st.expander('数值化', True):
            col1, col2 = st.columns(2)
            cols = col1.multiselect('枚举编码', object_cols)
            for col in cols:
                dum_df, _ = pd.factorize(data[col])
                data.loc[:, col + '_enum'] = dum_df

            cols = col2.multiselect('哑编码', object_cols)
            ff.dummies(data, cols)

    def discretization(self, data):
        numeric_cols = data.select_dtypes(exclude=['object']).columns
        with st.expander('离散化', True):

            self.__multi_select_with_value(
                '等距分箱',
                numeric_cols,
                lambda col, bin_num: ff.binning(data, col, int(bin_num),
                                                'width'),
                help_str='维度-分箱数，逗号分隔')

            self.__multi_select_with_value(
                '等频分箱',
                numeric_cols,
                lambda col, bin_num: ff.binning(data, col, int(bin_num),
                                                'quantile'),
                help_str='维度-分箱数，逗号分隔')

            self.__multi_select_with_value(
                '聚类分箱',
                numeric_cols,
                lambda col, bin_num: ff.binning(data, col, int(bin_num),
                                                'cluster'),
                help_str='维度-分箱数，逗号分隔')

    def dim_reduction(self, data):
        with st.expander('降维', True):
            st.write('coding')