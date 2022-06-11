import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from .tools import Tools


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
        ex.checkbox("列分析", key='col_analysis')
        self.add_tool_func('col_analysis', self.col_analysis)
        ex.checkbox("多列分析", key='cols_analysis')
        self.add_tool_func('cols_analysis', self.cols_analysis)

    def data_info(self, data):

        with st.expander('数据概览', True):
            tmp1, over_view_col1, over_view_col2, over_view_col3, tmp2 = st.columns(
                [0.1, 0.8, 0.6, 2, 0.1])
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
            tmp1, col1, col2, tmp2 = st.columns([0.1, 0.5, 0.6, 0.1])
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
                st.table(info_table)
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
                st.table(data.describe())

    def col_analysis(self, data):
        with st.expander('列分析', True):
            col_selected = st.selectbox("请选择一列", data.columns)
            col_data = data[col_selected]
            if col_data.dtype.name == 'object':
                st.write(f"""统计量

                离散数：{len(col_data.unique())}
                """)
                col_analysis_col1, col_analysis_col2 = st.columns(2)
                col_analysis_col1.plotly_chart(
                    px.pie(col_data.value_counts().to_frame(
                        name='count').reset_index(),
                           values='count',
                           names='index'))
                col_analysis_col2.plotly_chart(px.histogram(col_data,
                                                            marginal='box'),
                                               use_container_width=True)
            else:
                col_stats = np.round(col_data.describe(), 2).to_dict()

                st.write(f"""统计量

                合计：{np.round(col_data.sum(),2)}  非空数据量：{col_stats['count'] }   均值：{col_stats['mean']}   方差：{col_stats['std']}    最小值：{col_stats['min']}   最大值：{col_stats['max']} """
                         )
                col_analysis_col1, col_analysis_col2 = st.columns(2)
                col_analysis_col1.plotly_chart(px.box(col_data),
                                               use_container_width=True)
                col_analysis_col2.plotly_chart(px.histogram(col_data,
                                                            text_auto=True),
                                               use_container_width=True)

    def cols_analysis(self, data):
        with st.expander('多列分析', True):
            col_selected = st.multiselect('请选择两列', data.columns)
            if len(col_selected) != 2:
                return
            col_data = data[col_selected]
            # col_selected = st.selectbox("请选择一列", data.columns)
            col1, col2 = st.columns(2)

            col1.plotly_chart(
                px.scatter(col_data, x=col_selected[0], y=col_selected[1]))
            # px.pie(col_data.value_counts().to_frame(
            #     name='count').reset_index(),
            #     values='count',
            #     names='index'))
            # col_analysis_col2.plotly_chart(px.histogram(col_data,
            #                                             marginal='box'),
            #                             use_container_width=True)