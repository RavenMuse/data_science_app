
from faulthandler import disable
import pandas as pd
import streamlit as st
from tools.feature_tools import FeatureTools 
from tools.stats_tools import StatsTools
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


class DataToolsApp:
    def __init__(self):
        self.__tools_set = []

    def add_tool(self, tool):
        self.__tools_set.append(tool)



    def run(self):
        # config
        st.set_page_config(
            page_title="数据实验台",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://www.baidu.com',
                'Report a bug': "https://www.extremelycoolapp.com/bug",
                'About': "# This is a header. This is an *extremely* cool app!"
            }
        )
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
        st.sidebar.image('menu.jpeg')
    
        # app_info
        main_info=st.container()
        with main_info:
            st.markdown("""
            ### 数据科学实验台
            一个简单实用的数据科学工具集，左侧菜单是数据工具列表，在此之前您需要选择一个数据集，才能开启数据探索之旅！
            """)

        # data_unload
        uploaded_panel=st.expander("数据上传",True)
        uploaded_file = uploaded_panel.file_uploader('选择一个数据文件',type='csv')
        st.sidebar.markdown("""
        """)

        # menu
        for tools in self.__tools_set:
            tools.layout_menu()
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            for tools in self.__tools_set:
                tools.use_tool(data)

        ## mean info
        side_bar_info=st.sidebar.container()
        with side_bar_info:
            st.markdown("""---""")


app = DataToolsApp()
app.add_tool(StatsTools())
app.add_tool(FeatureTools())

app.run()
