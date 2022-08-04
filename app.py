import os
import warnings
import sqlparse
import pandas as pd
import numpy as np
import streamlit as st
import streamlit_authenticator as stauth
# from streamlit_elements import elements, mui, html, dashboard
from pandasql import sqldf
from streamlit_ace import st_ace
from datetime import datetime, timedelta
from authenticate import Authenticate
from tools.feature_tools import FeatureTools
from tools.statistics_tools import StatisticsTools
from tools.machine_learning_tools import MachineLearingTools

warnings.simplefilter(action="ignore", category=FutureWarning)


class DataToolsApp:

    def __init__(self):
        self.__tools_set = []
        self.__credentials = {
            'admin': {
                'name': 'admin',
                'password': stauth.Hasher(['hmyzch']).generate()[0]
            }
        }
        self.__authentication_status = None
        self.__authenticator = None
        st.set_page_config(
            page_title="数据实验台",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://www.baidu.com',
                'Report a bug': "https://www.extremelycoolapp.com/bug",
                'About': "# This is a header. This is an *extremely* cool app!"
            })

    def __login(self):

        self.__authenticator = Authenticate(self.__credentials,
                                            'data_science_app',
                                            'data_science_app')
        _, self.__authentication_status, _ = self.__authenticator.login()
        col1, col2 = st.columns([0.6, 0.4])
        if self.__authentication_status == False:
            col2.error('用户名/密码不正确')
            return False
        if self.__authentication_status == None:
            col2.info('请输入用户名/密码')
            return False
        return True

    def __get_data(self):

        # with st.expander("数据源", True):
        ## 数据源
        paths = st.file_uploader('选择数据文件',
                                 accept_multiple_files=True,
                                 type='csv')

        # @st.cache(allow_output_mutation=True)
        def __load_data(path=None):
            return pd.read_csv(path)

        if not paths:
            table_name = 'iris'
            locals()[table_name] = __load_data(
                '/data/data_science_app/iris.csv')
            file_name = 'iris.csv'
        else:
            for path in paths:
                table_name = path.name.split('.')[0]
                locals()[table_name] = __load_data(path)
            file_name = paths[0].name

        ## SQL操作
        col1, col2 = st.columns([0.85, 0.15])
        cols = ',\n'.join(locals()[table_name].columns)
        default_sql = sqlparse.format(
            f"SELECT {cols} FROM {table_name} LIMIT 20",
            reindent=True,
            keyword_case='upper')
        with col1:
            st.write("###### SQL")
            sql = st_ace(value=default_sql,
                         language='sql',
                         theme='terminal',
                         show_gutter=False,
                         height=180)

        try:
            data = sqldf(sql, locals())
        except:
            st.error('无效的查询！')
            st.stop()

        ## 样本操作
        if 'data' not in st.session_state:
            st.session_state.data = {'default_sample': data}
            st.session_state.data_count = 1
        col2.write('&nbsp;')
        sample_name = col2.text_input('新样本名称', 'va')

        if col2.button('创建样本'):
            if sample_name not in st.session_state.data:
                st.session_state.data_count += 1
            st.session_state.data[sample_name] = data

        col1, col2 = st.columns([0.85, 0.15])
        col2.write('&nbsp;')

        # options = st.session_state.data.keys()

        def on_remove():
            if st.session_state.data_count != 1:
                st.session_state.data_count -= 1
                del st.session_state.data[sample_name]
                st.session_state.current_sample = list(
                    st.session_state.data.keys())[-1]

        remove_btn = col2.button('删除样本', on_click=on_remove)
        options = st.session_state.data.keys()
        sample_name = col1.selectbox('当前样本',
                                     options,
                                     key='current_sample',
                                     index=len(options) - 1)

        data = st.session_state.data[sample_name]
        # col2.write(
        #     f'`内存占用：{np.round(data.memory_usage(index=True, deep=True).sum()/1028,2)} Kb`'
        # )
        return data, file_name

    def __save_data(self, data, file_name):

        # with st.expander("保存数据", False):
        # 显示结果数据
        st.write(data)

        # 数据下载保存
        @st.cache
        def convert_df(data):
            return data.to_csv(index=False, encoding='utf_8_sig')

        col1, col2 = st.columns([0.8, 0.2])
        file_name_new = col1.text_input('请输入保存文件名：',
                                        file_name.split('.')[0] + '_new.csv')
        col2.write('&nbsp;')
        col2.download_button(
            label="下载结果数据",
            data=convert_df(data),
            file_name=file_name_new,
            mime='text/csv',
        )

    def __load_style(self):
        # st.markdown(
        #     """
        # <style>
        # [data-testid=stSidebar] :first-child {
        #     background-color: rgb(48 47 45);
        #     color: white;
        # }
        # </style>
        # """,
        #     unsafe_allow_html=True,
        # )
        st.markdown(
            """
        <style>
        .main .block-container {
            padding: 3rem 5rem 10rem;
        }
        [data-testid=stSidebar] [data-baseweb=checkbox] :last-child {
            font-size: 0.9rem
        }
        [data-testid=stSidebar] .css-17ziqus {
            width: 18rem
        }
        [data-testid=stExpander] .streamlit-expanderHeader {
            font-size: 1rem;
            font-weight: 500;
        }
        [data-testid=stSidebar] .css-hxt7ib {
            padding-top: 3rem
        }
        footer {visibility: hidden;}
        </style>
        """,
            unsafe_allow_html=True,
        )

    def add_tool(self, tool):
        self.__tools_set.append(tool)

    def run(self):

        if not self.__login():
            return
        self.__load_style()
        # menu picture
        st.sidebar.image('/data/data_science_app/logo/menu_black.png')

        st.sidebar.markdown("""
        """)

        # menu
        for tools in self.__tools_set:
            tools.layout_menu()

            ## menu info
            # side_bar_info = st.sidebar.container()
            # with side_bar_info:
            #     st.markdown("""---""")

            # main
            ## app_info
        st.markdown("""
        ## 数据科学实验台
        - 一个简单实用的数据科学工具集，左侧菜单是数据工具列表，在此之前您需要选择一个数据集，才能开启数据探索之旅！
        """)

        st.write('### 数据原料')
        ## app_tools
        data, file_name = self.__get_data()

        st.write('### 数据实验')
        if data is not None:
            tabs_name = [tools.tools_name for tools in self.__tools_set]
            tabs_name.append('数据保存')
            tabs = st.tabs(tabs_name)
            for i, tools in enumerate(self.__tools_set):
                with tabs[i]:
                    tools.use_tool(data)
            with tabs[-1]:
                self.__save_data(data, file_name)
        self.__authenticator.logout('退出登录', 'sidebar')


app = DataToolsApp()
app.add_tool(FeatureTools())
app.add_tool(StatisticsTools())
app.add_tool(MachineLearingTools())

app.run()