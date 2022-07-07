import os
import warnings
import sqlparse
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import streamlit.components.v1 as components
import plotly.figure_factory as ff

from pandasql import sqldf
from datetime import datetime, timedelta
from authenticate import Authenticate
from tools.feature_tools import FeatureTools
from tools.stats_tools import StatsTools

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
        if self.__authentication_status == False:
            st.error('用户名/密码不正确')
            return False
        if self.__authentication_status == None:
            st.info('请输入用户名/密码')
            return False
        return True

    def __get_data(self):
        data_panel = st.expander("数据源", True)
        path = data_panel.file_uploader('选择一个数据文件', type='csv')

        @st.cache(allow_output_mutation=True)
        def __load_data(path=None):
            return pd.read_csv(path)

        if not path:
            orgin_data = __load_data('/data/data_science_app/test_sample.csv')
        else:
            orgin_data = __load_data(path)

        def __sql_format():
            st.session_state.sql_input = sqlparse.format(
                st.session_state.sql_input,
                reindent=True,
                keyword_case='upper')

        sql = data_panel.text_area('SQL',
                                   "SELECT *\nFROM orgin_data\nLIMIT 20",
                                   key='sql_input',
                                   on_change=__sql_format)
        try:
            data = sqldf(sql, locals())
        except:
            st.error('无效的查询!')
            return None
        data_panel.write(data)
        return data

    def add_tool(self, tool):
        self.__tools_set.append(tool)

    def run(self):
        if not self.__login():
            return

        # st.markdown(
        #     """
        # <style>
        # [data-testid=stSidebar] :first-child {
        #     background-color: #6ec0f2;
        #     color: white;
        # }
        # </style>
        # """,
        #     unsafe_allow_html=True,
        # )

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

        # app_tools
        data = self.__get_data()
        if data is not None:
            for tools in self.__tools_set:
                tools.use_tool(data)


app = DataToolsApp()
app.add_tool(FeatureTools())
app.add_tool(StatsTools())

app.run()

# bootstrap 4 collapse example
# components.html(
#     """
#     <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
#     <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
#     <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
#     <div id="accordion">
#       <div class="card">
#         <div class="card-header" id="headingOne">
#           <h5 class="mb-0">
#             <button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
#             Collapsible Group Item #1
#             </button>
#           </h5>
#         </div>
#         <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
#           <div class="card-body">
#             Collapsible Group Item #1 content
#           </div>
#         </div>
#       </div>
#       <div class="card">
#         <div class="card-header" id="headingTwo">
#           <h5 class="mb-0">
#             <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
#             Collapsible Group Item #2
#             </button>
#           </h5>
#         </div>
#         <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
#           <div class="card-body">
#             Collapsible Group Item #2 content
#           </div>
#         </div>
#       </div>
#     </div>
#     """,
#     height=600,
# )