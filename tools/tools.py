import streamlit as st


class Tools:

    def __init__(self):
        self.tools_name = ''
        self.__tools_func = {}

    def add_tool_func(self, tool_key, func):
        self.__tools_func[tool_key] = func

    def use_tool(self, data):
        for key, func in self.__tools_func.items():
            if st.session_state[key]:
                func(data)