import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import time
from .tools import Tools


class FeatureTools(Tools):

    def __init__(self):
        super().__init__()
        self.tools_name = '特征工具'

    def use_tool(self, data):
        return super().use_tool(data)

    def layout_menu(self):

        ex = st.sidebar.expander('特征工具', True)
        ex.markdown("##### 数值处理")

        ex.checkbox("空值填充", key='fill_na')
        self.add_tool_func('fill_na', self.fill_na)
        ex.checkbox("正则化", key='normalize')
        self.add_tool_func('normalize', self.normalize)

    def fill_na(self, data):
        tool = st.expander('空值填充', True)
        tool.write('coding')

    def normalize(self, data):
        tool = st.expander('正则化', True)
        tool.write('coding')