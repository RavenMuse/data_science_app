from faulthandler import disable

import streamlit as st
import streamlit_authenticator as stauth
import os
import pandas as pd
import streamlit as st
from pandasql import sqldf
from tools.feature_tools import FeatureTools
from tools.stats_tools import StatsTools
import plotly.figure_factory as ff
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class DataToolsApp:

    def __init__(self):
        self.__tools_set = []
        self.__credentials = {
            'usernames': {
                'admin': {
                    'name': 'admin',
                    'password': stauth.Hasher(['hmyzch']).generate()[0]
                }
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

    def authenticate(self):
        ### https://github.com/mkhorasani/Streamlit-Authenticator
        self.__authenticator = stauth.Authenticate(self.__credentials,
                                                   'some_cookie_name',
                                                   'some_signature_key')
        name, self.__authentication_status, username = self.__authenticator.login(
            'Login', 'main')

    def add_tool(self, tool):
        self.__tools_set.append(tool)

    def run(self):
        self.authenticate()
        if self.__authentication_status == False:
            st.error('Username/password is incorrect')
            return
        if self.__authentication_status == None:
            st.warning('Please enter your username and password')
            return

        st.markdown(
            """
            <style>
            .css-1adrfps {
                background-color: #6ec0f2;
                padding: 1rem 1rem;
            }
            .css-18e3th9 {
                padding: 4rem 4rem 5rem 5rem;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        # menu picture
        # st.write(os.path.abspath('.'))
        # st.sidebar.image('/data/data_science_app/menu.jpg')

        st.sidebar.markdown("""
        """)

        # menu
        for tools in self.__tools_set:
            tools.layout_menu()

        ## menu info
        side_bar_info = st.sidebar.container()
        with side_bar_info:
            st.markdown("""---""")

        # main
        ## app_info
        main_info = st.container()
        with main_info:
            st.markdown("""
            ### 数据科学实验台
            一个简单实用的数据科学工具集，左侧菜单是数据工具列表，在此之前您需要选择一个数据集，才能开启数据探索之旅！
            """)

        # data_set
        data_panel = st.expander("数据源", True)
        # uploaded_file = data_panel.file_uploader('选择一个数据文件', type='csv')

        orgin_data = pd.read_csv('/data/data_science_app/test_sample.csv')
        # if uploaded_file is not None:
        #     data = pd.read_csv(uploaded_file)
        #     for tools in self.__tools_set:
        #         tools.use_tool(data)
        # data = pd.read_csv('/data/data_science_app/test_sample.csv')
        sql = data_panel.text_area('SQL', 'select * from orgin_data')
        try:
            data = sqldf(sql, locals())
        except:
            st.error('无效的查询')
            return
        data_panel.write(data)

        for tools in self.__tools_set:
            tools.use_tool(data)
        self.__authenticator.logout('Logout', 'main')


app = DataToolsApp()
app.add_tool(FeatureTools())
app.add_tool(StatsTools())

app.run()