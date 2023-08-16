import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import streamlit as st


train_pre_metric, test_pre_metric, pre_train, pre_test = [], [], 0, 0


def model_performance(metrics, site, mode):
    """
    可视化模型表现
    """
    # 转换为DataFrame
    df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Values'])
    index = df.index.tolist()
    # 显示表格
    st.markdown("### {}. 模型表现".format(site))
    if df.values[0] == -1:
        st.write('因为所给的数据不包含\'label\'列，所以无法查看模型的表现，不过可以下载预测数据')
    else:
        col1, col2, col3, col4 = st.columns(4)
        if mode == 'train':
            global train_pre_metric, pre_train
            if pre_train == 0:
                col1.metric(index[0], round(float(df.values[0]), 4))
                col2.metric(index[1], round(float(df.values[1]), 4))
                col3.metric(index[2], round(float(df.values[2]), 4))
                col4.metric(index[3], round(float(df.values[3]), 4))
                pre_train = 1
                train_pre_metric = df
            else:
                col1.metric(index[0], round(float(df.values[0]), 4),
                            round(float(df.values[0]) - float(train_pre_metric.values[0]), 4))
                col2.metric(index[1], round(float(df.values[1]), 4),
                            round(float(df.values[1]) - float(train_pre_metric.values[1]), 4))
                col3.metric(index[2], round(float(df.values[2]), 4),
                            round(float(df.values[2]) - float(train_pre_metric.values[2]), 4))
                col4.metric(index[3], round(float(df.values[3]), 4),
                            round(float(df.values[3]) - float(train_pre_metric.values[3]), 4))
                train_pre_metric = df
        elif mode == 'test':
            global test_pre_metric, pre_test
            if pre_test == 0:
                col1.metric(index[0], round(float(df.values[0]), 4))
                col2.metric(index[1], round(float(df.values[1]), 4))
                col3.metric(index[2], round(float(df.values[2]), 4))
                col4.metric(index[3], round(float(df.values[3]), 4))
                test_pre_metric = df
                pre_test = 1
            else:
                col1.metric(index[0], round(float(df.values[0]), 4),
                            round(float(df.values[0]) - float(test_pre_metric.values[0]), 4))
                col2.metric(index[1], round(float(df.values[1]), 4),
                            round(float(df.values[1]) - float(test_pre_metric.values[1]), 4))
                col3.metric(index[2], round(float(df.values[2]), 4),
                            round(float(df.values[2]) - float(test_pre_metric.values[2]), 4))
                col4.metric(index[3], round(float(df.values[3]), 4),
                            round(float(df.values[3]) - float(test_pre_metric.values[3]), 4))
                test_pre_metric = df

    return df


def data_value_count(data, site, mode='training'):
    """
    可视化每一个类别的个数
    """
    if mode == 'training':
        st.markdown("### {}. 上传数据各类别数量".format(site))
        trace = [go.Bar(x=[0, 1, 2, 3, 4, 5], y=data['label'].value_counts(), width=0.6)]
    else:
        st.markdown("### {}. 预测数据各类别数量".format(site))
        trace = [go.Bar(x=[0, 1, 2, 3, 4, 5], y=data['pred_label'].value_counts(), width=0.6)]
    fig = go.Figure(data=trace)
    st.plotly_chart(fig, use_container_width=True)


def data_percentage(data, site, mode='training'):
    """
    可视化每个类所占的比例
    """
    labels = [0, 1, 2, 3, 4, 5]
    if mode == 'training':
        st.markdown("### {}. 上传数据各类别所占百分比".format(site))
        values = data['label'].value_counts()
    else:
        st.markdown("### {}. 预测数据各类别所占百分比".format(site))
        values = data['pred_label'].value_counts()
    trace = [go.Pie(labels=labels, values=values, hole=0.4)]
    # layout = go.Layout(title='各类别所占百分比')
    fig = go.Figure(data=trace)
    st.plotly_chart(fig, use_container_width=True)


def data_nan_distribution(data, site):
    """
    可视化空缺值（NaN值）在数据集中所占位置
    """
    st.markdown("### {}. 上传数据的缺失值分布".format(site))
    data = data.drop('sample_id', axis=1)
    try:
        data = data.drop('label', axis=1)
    except KeyError:
        pass

    def find_nan(data_s):
        indexes = data_s.columns.tolist()  # 将列名变为列表
        tmp = pd.DataFrame(index=data_s.index, columns=data_s.columns)
        for i in range(data_s.shape[1]):
            tmp[indexes[i]] = data_s[indexes[i]].apply(lambda x: 1 if np.isnan(x) else 2)
        return tmp
    df = find_nan(data)
    fig = px.imshow(df, color_continuous_scale='PuBu')
    st.plotly_chart(fig, use_container_width=True)


def data_distribution(data, site):
    """
    可视化数据集总体数据分布（在标准化之后的，因为要是同一种颜色）
    """
    st.markdown("### {}. 上传数据的总体数据分布".format(site))
    df = data.drop('sample_id', axis=1)
    try:
        df = df.drop('label', axis=1)
    except KeyError:
        pass

    def has_single_value(series):
        begin = series[0]
        for value in series:
            if begin != value:
                return False
        return True

    def min_max_scaler(x):
        if has_single_value(x):
            return x.apply(lambda y: 0.5)
        else:
            return (x - np.min(x)) / (np.max(x) - np.min(x))

    df = df.apply(min_max_scaler)
    # print(df)
    fig = px.imshow(df, color_continuous_scale='PuBu')
    st.plotly_chart(fig, use_container_width=True)


def data_classification(data, label, site):
    """
    可视化数据集分类结果
    不要用在训练集上，会很慢
    """
    st.markdown("### {}. 数据数据预测结果可视化".format(site))
    # label = data['label']
    data = data[['feature5', 'feature10', 'feature15', 'feature22', 'feature45', 'feature71']].copy()
    data.fillna(data.mean(), inplace=True, axis=0)
    if len(data) < 3 or isinstance(data, pd.Series):
        data = pd.DataFrame(data[['feature10', 'feature15', 'feature71']])
        data.fillna(0, inplace=True)
        print(data)
        data.columns = ['x', 'y', 'z']
        data_after = pd.concat([data, label], axis=1)
    elif len(data) < 30:
        pca = PCA(n_components=3).fit_transform(data)
        pca = pd.DataFrame(pca, columns=['x', 'y', 'z'])
        data_after = pd.concat([pca, label], axis=1)
    else:
        tsne = TSNE(n_components=3, perplexity=25.0, learning_rate=30, random_state=21).fit_transform(data)
        tsne = pd.DataFrame(tsne, columns=['x', 'y', 'z'])
        data_after = pd.concat([tsne, label], axis=1)

    fig = px.scatter_3d(data_after, x='x', y='y', z='z', color='pred_label', color_continuous_scale='Spectral')
    st.plotly_chart(fig, use_container_width=True)
