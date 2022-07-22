import os
import warnings
import sqlparse
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth

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

        with st.expander("数据源", True):

            paths = st.file_uploader('选择数据文件',
                                     accept_multiple_files=True,
                                     type='csv')
            col1, col2 = st.columns([0.85, 0.15])

            @st.cache(allow_output_mutation=True)
            def __load_data(path=None):
                return pd.read_csv(path)

            if not paths:
                table_name = 'test_sample'
                locals()[table_name] = __load_data(
                    '/data/data_science_app/test_sample.csv')
                file_name = 'test_sample.csv'
            else:
                for path in paths:
                    table_name = path.name.split('.')[0]
                    locals()[table_name] = __load_data(path)
                file_name = paths[0].name

            # def __sql_format():
            #     st.session_state.sql_input = sqlparse.format(
            #         st.session_state.sql_input,
            #         reindent=True,
            #         keyword_case='upper')

            cols = ',\n'.join(locals()[table_name].columns)
            with col1:
                st.write("###### SQL")
                sql = st_ace(
                    value=f"SELECT {cols}\nFROM {table_name}\nLIMIT 20",
                    language='sql',
                    theme='terminal',
                    show_gutter=False,
                    height=200)

            col2.write("""&nbsp;""")
            try:
                data = sqldf(sql, locals())
                if 'data' not in st.session_state:
                    st.session_state.data = {'样本1': data}
                    st.session_state.data_count = 1

                add_btn = col2.button('创建新样本')
                if add_btn:
                    st.session_state.data_count += 1
                    st.session_state.data[
                        f'样本{st.session_state.data_count}'] = data

                options = list(st.session_state.data.keys())
                sample_name = col2.selectbox('当前样本', options)
                data = st.session_state.data[sample_name]
            except:
                st.error('无效的查询！')
                st.stop()
            return data, file_name

    def __save_data(self, data, file_name):

        with st.expander("保存数据", False):
            # 显示结果数据
            st.write(data)

            # 数据下载保存
            @st.cache
            def convert_df(data):
                return data.to_csv(index=False, encoding='utf_8_sig')

            file_name_new = st.text_input('请输入文件名：',
                                          file_name.split('.')[0] + '_new.csv')
            st.download_button(
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
        [data-testid=stSidebar] .css-1siy2j7 {
            width: 18rem
        }
        [data-testid=stExpander] .streamlit-expanderHeader {
            font-size: 1rem;
            font-weight: 500;
        }
        [data-testid=stSidebar] .css-hxt7ib {
            padding-top: 4rem
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
        main_info = st.container()
        with main_info:
            st.markdown("""
            ### 数据科学实验台
            - 一个简单实用的数据科学工具集，左侧菜单是数据工具列表，在此之前您需要选择一个数据集，才能开启数据探索之旅！
            """)

        ## app_tools
        data, file_name = self.__get_data()
        if data is not None:
            for tools in self.__tools_set:
                tools.use_tool(data)
        self.__save_data(data, file_name)

        self.__authenticator.logout('退出登录', 'sidebar')


app = DataToolsApp()
app.add_tool(FeatureTools())
app.add_tool(StatisticsTools())
app.add_tool(MachineLearingTools())

app.run()