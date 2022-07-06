import streamlit as st
# import pandas as pd
# import numpy as np
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
        ex.checkbox("去噪", key='denoising')
        self.add_tool_func('denoising', self.denoising)
        ex.checkbox("空值填充", key='fill_na')
        self.add_tool_func('fill_na', self.fill_na)
        ex.checkbox("离散化", key='discretization')
        self.add_tool_func('discretization', self.discretization)
        ex.checkbox("数值化", key='numeric')
        self.add_tool_func('numeric', self.numeric)
        ex.checkbox("正则化", key='normalize')
        self.add_tool_func('normalize', self.normalize)

    def fill_na(self, data):
        tool = st.expander('空值填充', True)
        tool.write('coding')

    def numeric(self, data):
        tool = st.expander('数值化', True)
        tool.write('coding')

    def normalize(self, data):
        tool = st.expander('正则化', True)
        numeric_cols = data.select_dtypes(exclude=['object']).columns
        with tool:
            cols = st.multiselect('Z-Scores标准化', numeric_cols)
            ff.z_scores_std(data, cols)

            cols = st.multiselect('Max-Min标准化', numeric_cols)
            ff.max_min_std(data, cols)

    def discretization(self, data):
        tool = st.expander('离散化', True)
        tool.write('coding')

    def denoising(self, data):
        tool = st.expander('去噪', True)
        numeric_cols = data.select_dtypes(exclude=['object']).columns
        with tool:
            cols = st.multiselect('离群点法', numeric_cols)
            ff.outlier_denoising(data, cols)