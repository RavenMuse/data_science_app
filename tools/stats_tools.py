import streamlit as st
import pandas as pd
import numpy as np
import sys
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

from .tools import Tools

sys.path.append("..")
from func.feature_func import FeatureFunction as ff


class StatsTools(Tools):

    def __init__(self):
        super().__init__()
        self.tools_name = '统计工具'

    def use_tool(self, data):
        return super().use_tool(data)

    def layout_menu(self):
        ex = st.sidebar.expander('统计工具', True)
        ex.markdown("##### 基础分析")
        ex.checkbox("数据概览", key='data_info')
        self.add_tool_func('data_info', self.data_info)
        ex.checkbox("单维分析", key='single_dim_analysis')
        self.add_tool_func('single_dim_analysis', self.single_dim_analysis)
        ex.checkbox("多维分析", key='multi_dim_analysis')
        self.add_tool_func('multi_dim_analysis', self.multi_dim_analysis)
        # ex.markdown("##### 综合分析")
        # ex.checkbox("topsis分析", key='cols_analysis')
        # self.add_tool_func('cols_analysis', self.cols_analysis)
        # ex.checkbox("主成分分析", key='cols_analysis')
        # self.add_tool_func('cols_analysis', self.cols_analysis)
        # ex.checkbox("因子分析", key='cols_analysis')
        # self.add_tool_func('cols_analysis', self.cols_analysis)
        # ex.markdown("##### 预测分析")
        # ex.checkbox("分类预测", key='cols_analysis')
        # self.add_tool_func('cols_analysis', self.cols_analysis)
        # ex.checkbox("数值预测", key='cols_analysis')
        # self.add_tool_func('cols_analysis', self.cols_analysis)

    def data_info(self, data):

        with st.expander('数据概览', True):
            tmp1, over_view_col1, over_view_col2, over_view_col3, tmp2 = st.columns(
                [0.1, 0.25, 0.25, 0.25, 0.1])
            with over_view_col1:
                st.metric(label="数据量", value=len(data))
            with over_view_col2:
                st.metric(label="列数", value=len(data.columns))
            with over_view_col3:
                st.metric(
                    label="内存占用量",
                    value=
                    f"{np.round(data.memory_usage(index=True, deep=True).sum()/1028,2)} Kb"
                )
            st.write("数值分析")
            col1, col2 = st.columns([0.5, 0.6])
            with col1:
                info_table = pd.DataFrame({
                    '列名':
                    data.columns.values,
                    '类型':
                    data.dtypes.apply(lambda x: x.name).values,
                    '非空数据量':
                    data.count().values,
                    '内存占用量': (np.round(
                        data.memory_usage(index=False, deep=True) / 1028,
                        2).astype(str) + 'Kb').values
                })
                st.write(info_table)
                # info_table = go.Figure(data=[
                #     go.Table(
                #         header=dict(values=['列名', '非空数据量', '类型', '内存占用量'],
                #                     align='left'),
                #         cells=dict(values=[
                #             data.columns.values,
                #             data.count().values,
                #             data.dtypes.apply(lambda x: x.name).values,
                #             (np.round(
                #                 data.memory_usage(index=False, deep=True) /
                #                 1028, 2).astype(str) + 'Kb').values
                #         ],
                #                    align='left'))
                # ])
                # info_table.update_layout(height=150,
                #                          margin=dict(t=0, l=10, r=10, b=0))
                # st.plotly_chart(info_table, use_container_width=True)
            with col2:
                st.write(data.describe())

    def single_dim_analysis(self, data):
        with st.expander('单维分析', True):
            col_selected = st.selectbox("请选择单个维度", data.columns)
            col_data = data[col_selected]
            if col_data.dtype.name == 'object':
                unique_count = len(col_data.unique())
                st.write(f"""统计量

                离散数：{unique_count}
                """)
                #todo: 高离散值变量暂不进行图表分析
                if unique_count < 100:
                    col_analysis_col1, col_analysis_col2 = st.columns(2)

                    col_analysis_col2.plotly_chart(
                        px.pie(col_data.value_counts().to_frame(
                            name='count').reset_index(),
                               values='count',
                               names='index'))

                    col_analysis_col1.plotly_chart(px.histogram(
                        col_data, marginal='box'),
                                                   use_container_width=True)
            else:
                col_stats = np.round(col_data.describe(), 2).to_dict()

                varity = 0 if col_stats['mean'] == 0 else np.round(
                    col_stats['std'] / col_stats['mean'], 2)

                kurtosis = np.round(stats.kurtosis(col_data, fisher=False), 2)
                skew = np.round(stats.skew(col_data), 2)

                st.write(f"""统计量

                合计：{np.round(col_data.sum(),2)}  均值：{col_stats['mean']}  标准差：{col_stats['std']}  异变系数：{varity}  峰度：{kurtosis} 偏度：{skew} 最小值：{col_stats['min']} 25%：{col_stats['25%']}  50%：{col_stats['50%']}  75%：{col_stats['75%']}  最大值：{col_stats['max']} """
                         )
                col_analysis_col1, col_analysis_col2 = st.columns(2)
                col_analysis_col1.plotly_chart(px.box(col_data),
                                               use_container_width=True)
                col_analysis_col2.plotly_chart(px.histogram(col_data,
                                                            text_auto=True),
                                               use_container_width=True)

    def multi_dim_analysis(self, data):
        with st.expander('多维分析', True):
            col1, col2 = st.columns([0.3, 0.7])
            chart_type_dict = {
                'scatter': '散点图',
                'line': '折线图',
                'bar': '柱状图',
                'violin': '提琴图'
            }
            chart_type, _ = col1.selectbox("图表",
                                           chart_type_dict.items(),
                                           format_func=lambda x: x[1])

            numeric_cols = data.select_dtypes(exclude=['object']).columns
            object_cols = data.select_dtypes(include=['object']).columns
            if chart_type == 'scatter':
                x = col1.selectbox("X", data.columns)
                y = col1.selectbox("Y", data.columns, index=1)

                text = col1.selectbox(
                    'Text', object_cols) if not object_cols.empty else None
                size = col1.multiselect('Size', numeric_cols)
                color = col1.multiselect('Color', data.columns)
                color = color[0] if len(color) == 1 else None
                size = size[0] if len(size) == 1 else None
                fig = px.scatter(data,
                                 x=x,
                                 y=y,
                                 size=size,
                                 color=color,
                                 hover_name=text)
            if chart_type == 'bar':
                x_col = col1.selectbox("X", data.columns)
                y_col = col1.selectbox("Y", data.columns, index=1)
                fig = px.bar(data, x=x_col, y=y_col)

            if chart_type == 'line':
                x_col = col1.selectbox("X", data.columns)
                y_col = col1.selectbox("Y", data.columns, index=1)
                fig = px.line(data, x=x_col, y=y_col)

            if chart_type == 'violin':
                if not object_cols.empty:
                    x_col = col1.selectbox("X", object_cols)
                    y_col = col1.selectbox("Y", numeric_cols, index=1)
                    fig = px.violin(data, x=x_col, y=y_col)
                else:
                    fig = None

            col2.plotly_chart(fig, use_container_width=True)
