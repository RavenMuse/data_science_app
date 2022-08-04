import streamlit as st
import pandas as pd
import numpy as np
import sys
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as pff
from .tools import Tools

sys.path.append("..")
from func.feature_func import FeatureFunction as ff


class StatisticsTools(Tools):

    def __init__(self):
        super().__init__()
        self.tools_name = '数理统计'

    def use_tool(self, data):
        return super().use_tool(data)

    def layout_menu(self):
        ex = st.sidebar.expander(self.tools_name, True)
        ex.markdown("##### 基础分析")
        ex.checkbox("数据概览", key='data_info')
        self.add_tool_func('data_info', self.data_info)
        ex.checkbox("单维分析", key='single_dim_analysis')
        self.add_tool_func('single_dim_analysis', self.single_dim_analysis)
        ex.checkbox("多维分析", key='multi_dim_analysis')
        self.add_tool_func('multi_dim_analysis', self.multi_dim_analysis)

        ex.markdown("##### 假设检验")
        ex.checkbox("分布检验", key='distribution_test')
        self.add_tool_func('distribution_test', self.distribution_test)
        ex.checkbox("方差检验", key='variance_test')
        self.add_tool_func('variance_test', self.variance_test)
        ex.checkbox("参数检验", key='parameter_test')
        self.add_tool_func('parameter_test', self.parameter_test)
        ex.checkbox("非参检验", key='non_parameter_test')
        self.add_tool_func('non_parameter_test', self.non_parameter_test)

        ex.markdown("##### 高级分析")
        ex.checkbox("综合评估", key='comprehensive_evaluation')
        self.add_tool_func('comprehensive_evaluation',
                           self.comprehensive_evaluation)
        ex.checkbox("因子分析", key='factor_analysis')
        self.add_tool_func('factor_analysis', self.factor_analysis)

    def data_info(self, data):
        with st.expander('数据概览', True):
            st.write("数值分析")

            info_table = pd.DataFrame({
                '列名':
                data.columns.values,
                '类型':
                data.dtypes.apply(lambda x: x.name).values,
                '非空数据量':
                data.count().values,
                '内存占用量':
                (np.round(data.memory_usage(index=False, deep=True) / 1028,
                          2).astype(str) + 'Kb').values
            })
            col1, col2 = st.columns([0.5, 0.5])
            col1.dataframe(info_table, height=300)

            col2.dataframe(data.describe(), height=300)

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
            st.write('相关性分析')
            col1, col2 = st.columns([0.5, 0.5])
            numeric_corr, object_corr = ff.correlation_analysis(data)
            col1.plotly_chart(px.imshow(numeric_corr),
                              use_container_width=True)
            col2.plotly_chart(px.imshow(object_corr), use_container_width=True)

    def single_dim_analysis(self, data):
        with st.expander('单维分析', True):
            col_selected = st.selectbox("请选择单个维度", data.columns)
            col_data = data[col_selected]
            if col_data.dtype.name == 'object':
                unique_count = len(col_data.unique())
                entropy = np.round(ff.entropy(col_data), 2)
                st.write(f"""统计量

                离散数：{unique_count}  熵：{entropy}
                """)
                #todo: 高离散值变量暂不进行图表分析
                if unique_count < 100:
                    col1, col2 = st.columns(2)

                    col1.plotly_chart(
                        px.pie(col_data.value_counts().to_frame(
                            name='count').reset_index(),
                               values='count',
                               names='index'))

                    col2.plotly_chart(px.histogram(col_data, marginal='box'),
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
                col1, col2 = st.columns(2)
                col1.plotly_chart(px.box(col_data), use_container_width=True)
                col2.plotly_chart(px.histogram(col_data, text_auto=True),
                                  use_container_width=True)

    def multi_dim_analysis(self, data):
        with st.expander('多维分析', True):
            col1, col2 = st.columns([0.2, 0.8])
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

    def distribution_test(self, data):
        with st.expander('分布检验', True):
            st.write('conding')

    def variance_test(self, data):
        with st.expander('方差检验', True):
            st.write('conding')

    def parameter_test(self, data):
        with st.expander('参数检验', True):
            st.write('conding')

    def non_parameter_test(self, data):
        with st.expander('非参检验', True):
            st.write('conding')

    def factor_analysis(self, data):
        with st.expander('因子分析', True):
            st.write('conding')

    def comprehensive_evaluation(self, data):
        with st.expander('综合评估', True):
            st.write('conding')