import streamlit as st
from scipy import stats
# import pandas as pd
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

        ex.markdown("##### 数据加工")
        ex.checkbox("数值变换", key='number_transform')
        self.add_tool_func('number_transform', self.number_transform)
        ex.checkbox("离散化", key='discretization')
        self.add_tool_func('discretization', self.discretization)
        ex.checkbox("数值化", key='numeric')
        self.add_tool_func('numeric', self.numeric)

    def fill_na(self, data):
        tool = st.expander('空值填充', True)
        with tool:
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

    def numeric(self, data):
        tool = st.expander('数值化', True)
        tool.write('coding')

    def standardize(self, data):
        tool = st.expander('标准化', True)
        numeric_cols = data.select_dtypes(exclude=['object']).columns
        with tool:

            col1, col2, col3 = st.columns([0.3, 0.3, 0.3])
            cols = col1.multiselect('Z-Scores标准化', numeric_cols)
            ff.z_scores_std(data, cols)

            cols = col2.multiselect('Max-Min标准化', numeric_cols)
            ff.max_min_std(data, cols)

    def discretization(self, data):
        tool = st.expander('离散化', True)

        tool.write('coding')

    def denoising(self, data):
        tool = st.expander('降噪', True)
        numeric_cols = data.select_dtypes(exclude=['object']).columns
        with tool:
            cols = st.multiselect('离群点降噪', numeric_cols)
            ff.outlier_denoising(data, cols)

    def number_transform(self, data):
        tool = st.expander('数值变换', True)
        with tool:
            numeric_cols = data.select_dtypes(exclude=['object']).columns
            st.write('平移变换')

            st.write('缩放变换')

            st.write('box-cox变换')
            st.latex(r'''y(\lambda) = \begin{cases}
                        \frac{y^\lambda-1}{\lambda}, & \lambda\neq0 \\
                        \ln y,& \lambda=0 \\
                        \end{cases}''')

            trans_str = st.text_input(
                '变换列Json',
                help=
                "{'colname':𝜆},例如：{'sale':-1},对数变换：𝜆=0,倒数变换：𝜆=-1,平方根变换：𝜆=0.5")
            if trans_str:
                for col, lamb in eval(trans_str).items():
                    try:
                        for col, lamb in eval(trans_str).items():
                            if data[col].min() <= 0:
                                pre_process = data[col] + np.abs(
                                    data[col].min()) + 0.0001
                            else:
                                pre_process = data[col]
                            data.loc[:, col + '_bxcx'] = stats.boxcox(
                                pre_process, lamb)
                    except:
                        st.error('数据必须为正值/json格式错误')
