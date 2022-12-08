import streamlit as st
import pandas as pd
import numpy as np
import sys
from scipy import stats
import statsmodels.tsa.stattools as ts
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as pff
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from .tools import Tools

sys.path.append("..")
from func.feature_func import FeatureFunction as ff


class StatisticsTools(Tools):

    def __init__(self):
        super().__init__()
        self.tools_name = '数据分析'

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
        ex.checkbox("相关性分析", key='correlation_analysis')
        self.add_tool_func('correlation_analysis', self.correlation_analysis)

        ex.markdown("##### 假设检验")
        ex.checkbox("参数检验", key='parameter_test')
        self.add_tool_func('parameter_test', self.parameter_test)
        ex.checkbox("非参检验", key='non_parameter_test')
        self.add_tool_func('non_parameter_test', self.non_parameter_test)

        ex.markdown("##### 时序分析")
        ex.checkbox("序列分析", key='sequential_analysis')
        self.add_tool_func('sequential_analysis', self.sequential_analysis)

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

    def single_dim_analysis(self, data):
        with st.expander('单维分析', True):
            col_selected = st.selectbox("请选择单个维度", data.columns)
            col_data = data[col_selected]
            if col_data.dtype.name == 'object':
                unique_count = len(col_data.unique())
                entropy = np.round(ff.entropy(col_data), 2)
                st.write('统计量')
                st.write(f"""<font size=2>离散数：`{unique_count}`&emsp;&emsp;
                    熵：`{entropy}`</font>
                    """,
                         unsafe_allow_html=True)

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

                st.write('统计量')
                st.write(
                    f"""<font size=2>合计：`{np.round(col_data.sum(),2)}`&emsp;&emsp;
                    均值：`{col_stats['mean']}`&emsp;&emsp;
                    标准差：`{col_stats['std']}`&emsp;&emsp;
                    异变系数：`{varity}`&emsp;&emsp;
                    峰度：`{kurtosis}`&emsp;&emsp;
                    偏度：`{skew}`</font>
                    """,
                    unsafe_allow_html=True)

                st.write(f"""<font size=2>最小值：`{col_stats['min']}`&emsp;&emsp;
                    25%：`{col_stats['25%']}`&emsp;&emsp;
                    50%：`{col_stats['50%']}`&emsp;&emsp;
                    75%：`{col_stats['75%']}`&emsp;&emsp;
                    最大值：`{col_stats['max']}` </font>
                    """,
                         unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                # col1.plotly_chart(px.box(col_data), use_container_width=True)
                col2.plotly_chart(px.histogram(col_data,
                                               marginal='box',
                                               text_auto=True),
                                  use_container_width=True)

                col1.plotly_chart(pff.create_distplot(
                    [col_data], ['distplot'],
                    bin_size=1
                    if col_stats['mean'] == 0 else col_stats['mean'] / 20,
                    colors=['#F66095'],
                    show_rug=False),
                                  use_container_width=True)

                def __get_test_text(pvalue):
                    if pvalue < 0.05:
                        return f'<font color=Red>非正态</font>，P=`{pvalue}` 小于置信系数0.05'
                    else:
                        return f'<font color=Lime>正态</font>，P=`{pvalue}` 大于置信系数0.05'

                st.write(
                    f" **Shapiro-Wilk检验**：{__get_test_text(stats.shapiro(col_data).pvalue)}",
                    unsafe_allow_html=True)
                st.write(
                    f" **Normaltest检验**：{__get_test_text(stats.normaltest(col_data).pvalue)}，结合峰度检验和偏度检验得出",
                    unsafe_allow_html=True)
                target_dist = stats.norm(loc=col_stats['mean'],
                                         scale=col_data.std(ddof=1))
                st.write(
                    f"**Kolmogorov-Smirnov检验**：{__get_test_text(stats.kstest(col_data, target_dist.cdf).pvalue)}，使用样本无偏量估计参数构建正态分布",
                    unsafe_allow_html=True)
                # mean_p = col_stats['mean'] - col_stats['min']
                # shape = (mean_p**2) / (col_stats['std']**2)
                # scale = (col_stats['std']**2) / mean_p
                # st.write(
                #     stats.kstest(col_data,
                #                  stats.gamma(shape,
                #                              loc=col_stats['min'],
                #                              scale=scale).cdf,
                #                  N=len(col_data)))
                # st.write(stats.anderson(col_data, "norm").statistic)
                # st.write(stats.anderson(col_data, "norm").critical_values)
                # st.write("- **Anderson-Darling** 检验：非正态（p=0.001>0.05）",
                #          unsafe_allow_html=True)

    def multi_dim_analysis(self, data):
        with st.expander('多维分析', True):
            col1, col2 = st.columns([0.2, 0.8])
            chart_type_dict = {
                'scatter': '散点图',
                'line': '折线图',
                'bar': '柱状图',
                'violin': '提琴图',
                'sunburst': '旭日图'
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
                                 hover_name=text,
                                 marginal_x="box",
                                 marginal_y="violin")
            if chart_type == 'bar':
                x_col = col1.selectbox("X", data.columns)
                y_col = col1.selectbox("Y", data.columns, index=1)
                color = col1.selectbox("Color", object_cols)
                text = col1.selectbox("Text", object_cols)
                fig = px.bar(data, x=x_col, y=y_col, color=color, text=text)

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

            if chart_type == 'sunburst':
                if not object_cols.empty:
                    path = col1.multiselect("Path", object_cols)
                    if col1.checkbox('按值统计'):
                        value = col1.selectbox("Value", numeric_cols, index=1)
                        fig = px.sunburst(data, path=path, values=value)
                    else:
                        fig = px.sunburst(data, path=path)
                else:
                    fig = None

            col2.plotly_chart(fig, use_container_width=True)

    def correlation_analysis(self, _):
        with st.expander('相关性分析', True):
            options = st.session_state.data.keys()
            sample_names = st.multiselect(
                '选择一个或两个样本',
                options,
                key='correlation_analysis_multiselect',
                default=list(options)[0])
            if len(sample_names) == 0:
                return
            sample1 = st.session_state.data[sample_names[0]]
            if len(sample_names) > 1:
                sample2 = st.session_state.data[sample_names[1]]

                if len(sample1) != len(sample2):
                    st.warning('样本大小需要相同！')
                    return
                numeric_corr, object_corr = ff.correlation_analysis(
                    sample1, sample2)
            else:
                numeric_corr, object_corr = ff.correlation_analysis(sample1)
            col1, col2 = st.columns([0.5, 0.5])
            col1.plotly_chart(px.imshow(numeric_corr),
                              use_container_width=True)
            col2.plotly_chart(px.imshow(object_corr), use_container_width=True)

            st.info(
                "连续变量Pearson相关系数p，p<0：负相关，p>0：正相关，p=0：不相关 ； 离散变量信息增益系数p，p越大相关性越强"
            )

    def parameter_test(self, _):
        with st.expander('参数检验', True):
            col1, col2 = st.columns([0.2, 0.8])
            options = st.session_state.data.keys()
            sample_names = col1.multiselect('选择一个或两个样本',
                                            options,
                                            key='parameter_test_multiselect',
                                            default=list(options)[0])
            if len(sample_names) == 0:
                return

            sample1 = sample2 = st.session_state.data[sample_names[0]]
            if len(sample_names) > 1:
                sample2 = st.session_state.data[sample_names[1]]

            sample1_numeric_cols = sample1.select_dtypes(
                exclude=['object']).columns
            sample2_numeric_cols = sample2.select_dtypes(
                exclude=['object']).columns
            first_dim = col1.selectbox('第一个维度',
                                       sample1_numeric_cols,
                                       key='parameter_dim1')
            second_dim = col1.selectbox('第二个维度',
                                        sample2_numeric_cols,
                                        key='parameter_dim2')

            col_data1 = sample1[first_dim]
            col_data2 = sample2[second_dim]

            if len(col_data1) != len(col_data2):
                col1.warning('样本大小不同，无法检验！')
                return

            def __get_test_text(pvalue):
                if pvalue < 0.05:
                    return f'<font color=Lime>有显著性差异</font>，P=`{pvalue}` 小于置信系数0.05'
                else:
                    return f'<font color=Red>无显著性差异</font>，P=`{pvalue}` 大于置信系数0.05'

            col2.write('方差检验')
            levene_p = stats.levene(col_data1, col_data2).pvalue
            col2.write(f"- Levene检验： {__get_test_text(levene_p)}",
                       unsafe_allow_html=True)
            col2.write(
                f"- Bartlett检验： {__get_test_text(stats.bartlett(col_data1, col_data2).pvalue)}",
                unsafe_allow_html=True)

            col2.info("注：Bartlett检验要求样本近似正态")
            col2.write('均值检验')
            col2.write(
                f"- T检验（独立样本）： {__get_test_text(stats.ttest_ind(col_data1, col_data2,equal_var=levene_p>0.05).pvalue)}",
                unsafe_allow_html=True)
            col2.write(
                f"- T检验（配对样本）： {__get_test_text(stats.ttest_rel(col_data1, col_data2).pvalue)}",
                unsafe_allow_html=True)
            col2.write(
                f"- F检验（ANOVA）： {__get_test_text(stats.f_oneway(col_data1, col_data2).pvalue)}",
                unsafe_allow_html=True)
            col2.info("注：T检验、F检验要求样本近似正态，F检验（方差分析）常用于验证外部因素对前后数据的影响")
            col2.write('同分布检验')
            col2.write(
                f"- Kolmogorov-Smirnov检验： {__get_test_text(stats.ks_2samp(col_data1, col_data2).pvalue)}",
                unsafe_allow_html=True)
            col2.write(
                f"- Epps-Singleton检验： {__get_test_text(stats.epps_singleton_2samp(col_data1, col_data2).pvalue)}",
                unsafe_allow_html=True)
            col2.write(
                f"- Cramér-von Mises检验： {__get_test_text(stats.cramervonmises_2samp(col_data1, col_data2).pvalue)}",
                unsafe_allow_html=True)

    def non_parameter_test(self, _):
        with st.expander('非参检验', True):
            col1, col2 = st.columns([0.2, 0.8])
            options = st.session_state.data.keys()
            sample_names = col1.multiselect(
                '选择一个或两个样本',
                options,
                key='nonparameter_test_multiselect',
                default=list(options)[0])
            if len(sample_names) == 0:
                return

            if len(sample_names) == 1:
                is_contingency_test = col1.checkbox('列联表检验',
                                                    key='is_contingency_test')
            else:
                is_contingency_test = False

            if is_contingency_test:
                sample1 = st.session_state.data[sample_names[0]]
                object_cols = sample1.select_dtypes(include=['object']).columns
                if len(object_cols) == 0:
                    st.warning('无离散维度！')
                    return
                first_dim = col1.selectbox('离散维度', object_cols)
                second_dim = col1.selectbox('状态维度', object_cols)

                crosstab = pd.crosstab(sample1[first_dim], sample1[second_dim])

                def __get_test_text(pvalue):
                    if pvalue < 0.05:
                        return f'<font color=Lime>离散变量与观察状态有关</font>，P=`{pvalue}` 小于置信系数0.05'
                    else:
                        return f'<font color=Red>离散变量与观察状态无关</font>，P=`{pvalue}` 大于置信系数0.05'

                col2.write('列联表')
                col2.write(crosstab, height=200)
                col2.write('列联表检验')
                chi2, p, dof, expected = stats.chi2_contingency(crosstab)
                col2.write(f"- 卡方检验： {__get_test_text(p)}",
                           unsafe_allow_html=True)
                if crosstab.shape == (2, 2):
                    col2.write(
                        f"- Fisher检验： {__get_test_text(stats.fisher_exact(crosstab)[1])}",
                        unsafe_allow_html=True)
                    col2.write(
                        f"- Barnard检验： {__get_test_text(stats.barnard_exact(crosstab).pvalue)}",
                        unsafe_allow_html=True)
                    col2.write(
                        f"- Boschloo检验： {__get_test_text(stats.boschloo_exact(crosstab).pvalue)}",
                        unsafe_allow_html=True)
                    col2.info(
                        "注：Fisher精准检验在状态频数小于5时可靠性强，Barnard精准检验场景广，但要求状态相互独立不干涉"
                    )

            else:
                sample1 = sample2 = st.session_state.data[sample_names[0]]
                if len(sample_names) > 1:
                    sample2 = st.session_state.data[sample_names[1]]

                sample1_numeric_cols = sample1.select_dtypes(
                    exclude=['object']).columns
                sample2_numeric_cols = sample2.select_dtypes(
                    exclude=['object']).columns
                first_dim = col1.selectbox('第一个维度',
                                           sample1_numeric_cols,
                                           key='nonparameter_dim1')
                second_dim = col1.selectbox('第二个维度',
                                            sample2_numeric_cols,
                                            key='nonparameter_dim2')

                col_data1 = sample1[first_dim]
                col_data2 = sample2[second_dim]

                if len(col_data1) != len(col_data2):
                    col1.warning('样本大小不同，无法检验！')
                    return
                if first_dim == second_dim:
                    col1.warning('维度相同，无法检验！')
                    return

                def __get_test_text(pvalue):
                    if pvalue < 0.05:
                        return f'<font color=Lime>有显著性差异</font>，P=`{pvalue}` 小于置信系数0.05'
                    else:
                        return f'<font color=Red>无显著性差异</font>，P=`{pvalue}` 大于置信系数0.05'

                col2.write('秩检验')
                col2.write(
                    f"- Wilcoxon检验： {__get_test_text(stats.wilcoxon(col_data1, col_data2).pvalue)}",
                    unsafe_allow_html=True)
                col2.write(
                    f"- Mann-Whitney-U检验：{__get_test_text(stats.mannwhitneyu(col_data1, col_data2).pvalue)}",
                    unsafe_allow_html=True)
                col2.info("注：秩检验要求样本足够大，对数据分布无要求，常用于非正态大样本数据")
                col2.write('其他检验')
                col2.write(
                    f"- Fligner-Killeen检验（方差）： {__get_test_text(stats.fligner(col_data1, col_data2).pvalue)}",
                    unsafe_allow_html=True)
                col2.write(
                    f"- Ansari-Bradley检验（尺度）： {__get_test_text(stats.ansari(col_data1, col_data2).pvalue)}",
                    unsafe_allow_html=True)
                col2.write(
                    f"- Mood检验（尺度）： {__get_test_text(stats.mood(col_data1, col_data2)[1])}",
                    unsafe_allow_html=True)

    def sequential_analysis(self, data):
        with st.expander('序列分析', True):

            numeric_cols = data.select_dtypes(exclude=['object']).columns
            col1, col2 = st.columns([0.2, 0.8])
            col = col1.selectbox('时序维度', numeric_cols)
            diff = col1.slider('差分阶数', min_value=0, max_value=3, value=0)
            col_data = data[col] if diff == 0 else data[col].diff(
                diff).dropna()

            fig = px.line(col_data, title='时序图')
            col2.plotly_chart(fig, use_container_width=True)

            # 平稳性检验
            stats_val, p_val, _, _, stats_dic, _ = ts.adfuller(col_data)

            if p_val < 0.05 and stats_val < stats_dic['1%']:
                text = f"<font color=Lime>十分平稳</font>，P=`{p_val}` 小于置信系数0.05，统计值`{stats_val}`小于1%阈值`{stats_dic['1%']}`"
            elif p_val < 0.05 and stats_dic['1%'] < stats_val < stats_dic['5%']:
                text = f"<font color=Lime>相对平稳</font>，P=`{p_val}` 小于置信系数0.05，统计值`{stats_val}`介于1%、5%阈值[`{stats_dic['1%']}`,`{stats_dic['5%']}`]"
            else:
                text = f'<font color=Red>不平稳</font>，P=`{p_val}` 大于置信系数0.05；建议进行差分处理'

            col2.write(f"Augmented Dickey-Fuller（ADF）检验： {text}",
                       unsafe_allow_html=True)

            pacf = ts.pacf(col_data, method='ols')
            acf = ts.acf(col_data)

            # fig_df = pd.DataFrame(zip(np.append(
            #     acf, pacf), ['acf'] * len(acf) + ['pacf'] * len(pacf)),
            #                       columns=['value', 'type'])
            acf_df = pd.DataFrame(zip(acf, ['acf'] * len(acf)),
                                  columns=['value', 'type']).reset_index()
            pacf_df = pd.DataFrame(zip(pacf, ['pacf'] * len(acf)),
                                   columns=['value', 'type']).reset_index()
            fig_df = acf_df.append(pacf_df).rename(columns={'index': 'lags'})
            fig = px.bar(fig_df,
                         x='lags',
                         y='value',
                         color='type',
                         barmode='group',
                         title='ACF/PACF')

            col2.plotly_chart(fig, use_container_width=True)

            col2.info(
                """ACF/PACF 解析 \n - AR(p)模型：ACF-拖尾、PACF-p阶截尾 \n - MA(q)模型：ACF-q阶截尾、PACF-拖尾 \n - ARMA(p,q)：ACF-拖尾、PACF--拖尾 \n **拖尾**：为无论如何都不会为0，而是在某阶之后在0附近随机变化 ； **截尾**：在大于某阶后快速趋于0"""
            )

    def factor_analysis(self, data):
        with st.expander('因子分析', True):

            st.write('conding')

    def comprehensive_evaluation(self, data):
        with st.expander('综合评估', True):
            st.write('conding')