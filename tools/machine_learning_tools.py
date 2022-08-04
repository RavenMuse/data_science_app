import streamlit as st
import pandas as pd
import numpy as np
import sys
from sklearn.cluster import *
import plotly.express as px
from sklearn import manifold

from .tools import Tools

sys.path.append("..")
from func.feature_func import FeatureFunction as ff


class MachineLearingTools(Tools):

    def __init__(self):
        super().__init__()
        self.tools_name = '机器学习'

    def use_tool(self, data):
        return super().use_tool(data)

    def layout_menu(self):
        ex = st.sidebar.expander(self.tools_name, True)
        ex.markdown("##### 采样")
        ex.checkbox("过采样", key='over_sampling')
        self.add_tool_func('over_sampling', self.over_sampling)
        ex.checkbox("欠采样", key='under_sampling')
        self.add_tool_func('under_sampling', self.under_sampling)

        ex.markdown("##### 聚类")
        ex.checkbox("聚类分析", key='cluster_analysis')
        self.add_tool_func('cluster_analysis', self.cluster_analysis)

        ex.markdown("##### 分类")
        ex.checkbox("线性分类", key='linear_classifier')
        self.add_tool_func('linear_classifier', self.linear_classifier)
        ex.checkbox("树分类", key='tree_classifier')
        self.add_tool_func('tree_classifier', self.tree_classifier)

        ex.markdown("##### 回归")
        ex.checkbox("线性回归", key='linear_regression')
        self.add_tool_func('linear_regression', self.linear_regression)
        ex.checkbox("树回归", key='tree_regression')
        self.add_tool_func('tree_regression', self.tree_regression)

    def over_sampling(self, data):
        with st.expander('过采样', True):
            st.write('coding')

    def under_sampling(self, data):
        with st.expander('欠采样', True):
            st.write('coding')

    def cluster_analysis(self, data):
        with st.expander('聚类分析', True):
            col1, col2 = st.columns([0.2, 0.8])
            cluster_func_dict = {'kmeans': 'KMeans', 'dbscan': 'DBSCAN'}
            cluster_func, _ = col1.selectbox('算法',
                                             cluster_func_dict.items(),
                                             format_func=lambda x: x[1])

            numeric_cols = data.select_dtypes(exclude=['object']).columns
            cols = col1.multiselect('维度',
                                    numeric_cols,
                                    default=numeric_cols[0])
            if not cols:
                st.stop()
            col2.write('聚类评估')
            if cluster_func == 'kmeans':
                n_cluster = col1.number_input('簇数',
                                              value=2,
                                              min_value=2,
                                              max_value=30)
                kmeans = KMeans(n_clusters=n_cluster, random_state=0)
                if n_cluster > len(data):
                    st.error('簇数不能大于样本数！')
                    st.stop()
                cluster = kmeans.fit_predict(data[cols].values)
                ## 聚类评估
                sse = []
                for k in range(2, n_cluster + 1):
                    estimator = KMeans(n_clusters=k, random_state=0)
                    estimator.fit(data[cols].values)
                    sse.append(estimator.inertia_)
                sse_df = pd.DataFrame({
                    '簇数': range(2, n_cluster + 1),
                    '簇内误差平方': sse
                })
                fig = px.line(sse_df, x='簇数', y='簇内误差平方', title='轮廓系数分析')
                col2.plotly_chart(fig, use_container_width=True)

            data.loc[:, 'cluster'] = cluster.astype(str)

            title = '聚类效果'
            if len(cols) == 1:
                fig = px.violin(data, x='cluster', y=cols[0], title=title)

            if len(cols) == 2:
                fig = px.scatter(data,
                                 x=cols[0],
                                 y=cols[1],
                                 color='cluster',
                                 title=title)

            if len(cols) > 2:
                tsne = manifold.TSNE(n_components=2,
                                     init='pca',
                                     random_state=0)
                x = tsne.fit_transform(data[cols].values)
                cluster_df = pd.DataFrame(x,
                                          columns=['tsne_dim1', 'tsne_dim2'])
                ff.max_min_std(cluster_df, ['tsne_dim1', 'tsne_dim2'])
                cluster_df.loc[:, 'cluster'] = cluster.astype(str)
                fig = px.scatter(cluster_df,
                                 x='tsne_dim1',
                                 y='tsne_dim2',
                                 color='cluster',
                                 title=title)

            col2.plotly_chart(fig, use_container_width=True)

    def linear_classifier(self, data):
        with st.expander('线性分类', True):
            st.write('coding')

    def tree_classifier(self, data):
        with st.expander('树分类', True):
            st.write('coding')

    def linear_regression(self, data):
        with st.expander('线性回归', True):
            st.write('coding')

    def tree_regression(self, data):
        with st.expander('树回归', True):
            st.write('coding')