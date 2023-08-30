import numpy as np
import os
import streamlit as st
from streamlit_login_auth_ui.widgets import __login__

os.environ["STREAMLIT_SERVER_SHOW_BROWSER"] = "false"
__login__obj = __login__(auth_token="dk_prod_D8CE9EJ7MGM24AM5SHPT2CKNRT7F",
                         company_name="Shims",
                         width=200, height=250,
                         logout_button_name='Logout', hide_menu_bool=False,
                         hide_footer_bool=False,
                         lottie_url='https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

LOGGED_IN = __login__obj.build_login_ui()  # 首先进行登录界面的展示
username = __login__obj.get_username()  # 然后获取username
print(username)

if LOGGED_IN == True:  # 如果登陆成功，才进行下面的过程

    st.title('欢迎回来，{}'.format(username))

    import streamlit as st
    import os
    import pandas as pd
    from train import train_and_save
    from train import test_and_pred
    from utils import model_performance, data_value_count, data_percentage, data_nan_distribution, data_distribution, data_classification


    # -------------------------  可视化   -------------------------

    def update_head(metrics, mode='train', label=0):
        if visualization_option == "Training Results" and save_path_train is not None and mode == 'train':
            # 显示训练结果的可视化内容
            data = pd.read_csv(save_path_train)
            if metrics:
                # 显示模型训练结果可视化内容
                tab1, tab2 = st.tabs(['模型表现', '下载模型'])
                with tab1:
                    model_performance(metrics, 5, 'train')
                with tab2:
                    down_load_model(username, model_name=st.session_state.get('model_selection'))

            else:
                # 显示上传数据集的分布等可视化内容
                tab1, tab2, tab3, tab4 = st.tabs(['类别分布', '类别比例', '缺失值分布', '数据总体分布'])
                with tab1:
                    data_value_count(data, 1)
                with tab2:
                    data_percentage(data, 2)
                with tab3:
                    data_nan_distribution(data, 3)
                with tab4:
                    data_distribution(data, 4)

        if visualization_option == "Testing Results" and save_path_test is not None and mode == 'test':
            # 显示测试结果的可视化内容
            data = pd.read_csv(save_path_test)
            if metrics:
                tab1, tab2, tab3 = st.tabs(['模型表现', '下载预测结果', '预测结果可视化'])
                with tab1:
                    model_performance(metrics, 3, 'test')
                with tab2:
                    down_load_test_csv(username, model_name=st.session_state.get('model_selection'))
                with tab3:
                    if isinstance(label, np.ndarray):
                        label = pd.DataFrame(label, columns=['pred_label'])
                        data = pd.concat([data, label], axis=1)
                        data_value_count(data, 4, mode='testing')
                        data_percentage(data, 5, mode='testing')
                        data_classification(data, label, 6)

            else:
                tab1, tab2 = st.tabs(['缺失值分布', '数据总体分布'])
                with tab1:
                    data_nan_distribution(data, 1)
                with tab2:
                    data_distribution(data, 2)

    #  -------------------------  训练模型  -------------------------
    def train_model(save_path_train, username):
        # 提取模型
        model_name = st.session_state.get('model_selection')
        # 提取参数
        model_params = st.session_state.get('model_parameters')
        # 提取方法
        robust_method = st.session_state.get('robust_method')
        # 提取参数
        robust_params = st.session_state.get('robust_params')
        # 模型训练
        st.session_state['train_metrics'] = train_and_save(save_path_train, username, model_name, model_params, robust_method, robust_params)
        # 可视化模型在训练集上的表现
        train_metrics = st.session_state.get('train_metrics')
        update_head(train_metrics, mode='train')

    #   -------------------------  下载模型  -------------------------
    def down_load_model(username, model_name, div=st):

        model_path = './users/{}/models/{}_model.pkl'.format(username, model_name)

        if os.path.exists(model_path):
            if div == st:
                div.markdown("### 下载当前训练模型")
            div.write(f"Model Name: {model_path.split('/')[-1]}")
            div.write(f"Model Size: {os.path.getsize(model_path)} bytes")
            div.download_button(
                label="Download Model",
                data=open(model_path, 'rb'),
                file_name=model_path.split('/')[-1],
                key=str(div),
            )
        else:
            div.warning("Model file not found.")

    #   -------------------------  测试模型  -------------------------
    def test_model(save_path_test, username):
        # 测试的就是在最上面选好的模型
        model_name = st.session_state.get('model_selection')
        # 测试模型
        st.session_state['test_metrics'], pred = test_and_pred(save_path_test, username, model_name)
        # 可视化模型在测试集上的表现
        test_metrics= st.session_state.get('test_metrics')
        update_head(test_metrics, mode='test', label=pred)

    #   ------------------  下载训练后识别的类别  ------------------
    def down_load_test_csv(username, model_name, div=st):
        test_file_path = './users/{}/data/{}_prediction_result.csv'.format(username, model_name)

        if os.path.exists(test_file_path):
            if div == st:
                div.markdown("### 下载当前预测结果")
            div.write(f"File Name: {test_file_path.split('/')[-1]}")
            div.write(f"File Size: {os.path.getsize(test_file_path)} bytes")
            div.download_button(
                label="Download test csv",
                data=open(test_file_path, 'rb'),
                file_name=test_file_path.split('/')[-1],
                key=str(div),
            )
        else:
            div.warning("File not found.")

    #  -------------------------  显示与调整模型参数  -------------------------
    def choose_params(model_name):
        parameters = dict()
        with st.expander(f'{model_name}的参数选择'):
            if model_name == 'LightGBM':
                default = st.checkbox(label='使用调整好的默认参数')
                parameters['default'] = default
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    # learning_rate
                    parameters['learning_rate'] = st.number_input(
                        label='learning_rate', help='shrinkage rate (>=0.001)', disabled=default,
                        format='%.3f', min_value=0.001, value=0.100, step=0.001,
                    )
                    # num_iterations
                    parameters['num_iterations'] = st.number_input(
                        label='num_iterations', help='number of boosting iterations (>=0)',
                        disabled=default, format='%d', min_value=1, value=100
                    )
                with col2:
                    # min_child_sample
                    parameters['min_data_in_leaf'] = st.number_input(
                        label='min_data_in_leaf',
                        help='minimal number of data in one leaf. Can be used to deal with over-fitting (>=0)',
                        disabled=default, min_value=0, value=20, format='%d'
                    )
                    # max_depth
                    parameters['max_depth'] = st.number_input(
                        label='max_depth', help='limit the max depth for tree model (>=-1, -1 means no limit)',
                        disabled=default, min_value=-1, value=-1
                    )
                with col3:
                    # num_leaves
                    parameters['num_leaves'] = st.number_input(
                        label='num_leaves', help='max number of leaves in one tree (1< num_leaves <=131072)',
                        disabled=default, min_value=2, max_value=131072, value=31, format='%d'
                    )
                    # bagging_fraction
                    parameters['bagging_fraction'] = st.number_input(
                        label='bagging_fraction',
                        help='this will randomly select part of data without resampling (0.0 < bagging_fraction <=1.0)',
                        disabled=default, format='%.3f', min_value=0.001, max_value=1.000, value=1.000, step=0.001,
                    )
                with col4:
                    # lambda_l1
                    parameters['lambda_l1'] = st.number_input(
                        label='lambda_l1', help='L1 regularization (>=0.0)', disabled=default,
                        format='%.3f', min_value=0.000, value=0.000, step=0.001
                    )
                    # lambda_l2
                    parameters['lambda_l2'] = st.number_input(
                        label='lambda_l2', help='L2 regularization (>=0.0)', disabled=default,
                        format='%.3f', min_value=0.000, value=0.000, step=0.001
                    )
            elif model_name == 'GBDT':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    # learning_rate
                    parameters['learning_rate'] = st.number_input(
                        label='learning_rate', help='Learning rate shrinks the contribution of each tree (>=0.001)',
                        format='%.3f', min_value=0.001, value=0.100, step=0.001,
                    )
                    # n_estimators
                    parameters['n_estimators'] = st.number_input(
                        label='n_estimators', help='The number of boosting stages to perform (>=1)',
                        min_value=1, value=100,
                    )
                with col2:
                    # subsample
                    parameters['subsample'] = st.number_input(
                        label='subsample',
                        help='The fraction of samples to be used for fitting the individual base learners '
                             '(0.001 < subsample <=1.000)',
                        format='%.3f', min_value=0.001, max_value=1.000, value=1.000, step=0.001,
                    )
                    # max_features
                    parameters['max_features'] = st.number_input(
                        label='max_features', help='The number of features to consider when looking for the best split '
                                                   '(>=1)',
                        min_value=1, value=107
                    )
                with col3:
                    # max_depth
                    parameters['max_depth'] = st.number_input(
                        label='max_depth', help='Maximum depth of the individual regression estimators (>=1)',
                        min_value=1, value=3
                    )
                    # min_samples_split
                    parameters['min_samples_split'] = st.number_input(
                        label='min_samples_split', help='The minimum number of samples required to split an internal '
                                                        'node (>=2)',
                        min_value=2, value=2
                    )
                with col4:
                    # min_samples_leaf
                    parameters['min_samples_leaf'] = st.number_input(
                        label='min_samples_leaf', help='The minimum number of samples required to be at a leaf node '
                                                       '(>=1)',
                        min_value=1, value=1
                    )
                    # max_leaf_nodes
                    parameters['max_leaf_nodes'] = st.number_input(
                        label='max_leaf_nodes', help='Grow trees with max_leaf_nodes in best-first fashion (>=2)',
                        min_value=2, value=100
                    )
            elif model_name == 'XGBoost':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    parameters['learning_rate'] = st.number_input(
                        label='learning_rate', help='Step size shrinkage used in update to prevents overfitting '
                                                    '(0.0<= learning_rate <=1.0)',
                        format='%.1f', min_value=0.0, max_value=1.0, value=0.3, step=0.1,
                    )
                    parameters['max_depth'] = st.number_input(
                        label='max_depth', help='Maximum depth of a tree (>=0)',
                        min_value=0, value=6
                    )
                with col2:
                    parameters['min_child_weight'] = st.number_input(
                        label='min_chilg_weight', help='Minimum sum of instance weight (hessian) needed in a child '
                                                       '(>=0)',
                        min_value=0, value=1
                    )
                    parameters['gamma'] = st.number_input(
                        label='gamma', help='Minimum loss reduction required to make a further partition on '
                                            'a leaf node of the tree (0<= gamma <=1)',
                        format='%.2f', min_value=0.00, max_value=1.00, value=0.00, step=0.01,
                    )
                with col3:
                    parameters['subsample'] = st.number_input(
                        label='subsample', help='Subsample ratio of the training instances '
                                                '(0< subsample <=1)',
                        format='%.1f', min_value=0.1, max_value=1.0, value=1.0, step=0.1,
                    )
                    parameters['colsample_bytree'] = st.number_input(
                        label='colsample_bytree', help='is the subsample ratio of columns when constructing each tree '
                                                       '(0< colsample_bytree <=1)',
                        format='%.1f', min_value=0.1, max_value=1.0, value=1.0, step=0.1,
                    )
                with col4:
                    parameters['alpha'] = st.number_input(
                        label='alpha', help='L1 regularization term on weights (>=0)',
                        format='%.3f', min_value=0.000, value=0.000, step=0.001
                    )
                    parameters['lambda'] = st.number_input(
                        label='lambda', help='L2 regularization term on weights (>=0)',
                        format='%.3f', min_value=0.000, value=1.000, step=0.001
                    )
            elif model_name == 'SVM':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    parameters['C'] = st.number_input(
                        label='C', help='Regularization parameter (>0)',
                        format='%.3f', min_value=0.001, value=1.000, step=0.001
                    )
                with col2:
                    parameters['kernel'] = st.selectbox(
                        label='kernel', help='Specifies the kernel type to be used in the algorithm',
                        options=('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
                    )
                with col3:
                    parameters['shrinking'] = st.selectbox(
                        label='shrinking', help='Whether to use the shrinking heuristic',
                        options=(True, False)
                    )
                with col4:
                    parameters['decision_function_shape'] = st.selectbox(
                        label='decision_function_shape', help='"ovo"表示one vs one，"ovr"表示one vs rest',
                        options=('ovo', 'ovr')
                    )
            elif model_name == 'RandomForest':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    parameters['n_estimators'] = st.number_input(
                        label='n_estimators', help='The number of trees in the forest (>0)',
                        min_value=1, value=100
                    )
                    parameters['criterion'] = st.selectbox(
                        label='criterion', help='The function to measure the quality of a split',
                        options=('gini', 'entropy', 'log_loss')
                    )
                with col2:
                    parameters['max_depth'] = st.number_input(
                        label='max_depth', help='The maximum depth of the tree (>0)',
                        min_value=1, value=100
                    )
                with col3:
                    parameters['min_samples_split'] = st.number_input(
                        label='min_samples_split', help='The minimum number of samples required to split an internal '
                                                        'node (>=2)',
                        min_value=2, value=2
                    )
                with col4:
                    parameters['min_samples_leaf'] = st.number_input(
                        label='min_samples_leaf', help='The minimum number of samples required to be at a leaf node '
                                                       '(>=2)',
                        min_value=1, value=1
                    )
            elif model_name == 'GaussianNaiveBayes':
                st.markdown('没有较为合适的参数可供调整')
            elif model_name == 'KNN':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    parameters['n_neighbors'] = st.number_input(
                        label='n_neighbors', help='Number of neighbors to use by default for kneighbors queries (>=1)',
                        min_value=1, value=5
                    )
                with col2:
                    parameters['weights'] = st.selectbox(
                        label='weights', help='Weight function used in prediction (about how to voting)',
                        options=('uniform', 'distance')
                    )
                with col3:
                    parameters['algorithm'] = st.selectbox(
                        label='algorithm', help='Weight function used in prediction',
                        options=('auto', 'ball_tree', 'kd_tree', 'brute')
                    )
                with col4:
                    parameters['leaf_size'] = st.number_input(
                        label='leaf_size', help='Leaf size passed to BallTree or KDTree (>0)',
                        min_value=1, value=30
                    )
            elif model_name == 'LogisticRegression':
                st.markdown('没有较为合适的参数可供调整')
        return parameters


    def choose_robust_parameters(method_name):
        parameters = dict()
        with st.expander(f'扰动方法的参数选择'):
            if method_name == 'None':
                st.markdown('您没有选择扰动方法')
            elif method_name == 'AddRandomNoise':
                parameters['noise_scale'] = st.number_input(
                    label='noise scale', help='The scale of the input random noise(0<nosie scale<=1)',
                    format='%.2f', min_value=0.01, value=1.00, step=0.05
                )
            elif method_name == 'Clip':
                col1, col2 = st.columns(2)
                with col1:
                    parameters['lower bound'] = st.number_input(
                        label='clip range', help='Remove the extreme value beyond the range [-100, 10000]',
                        format='%d', min_value=-100, value=10000, step=1000
                    )
                with col2:
                    parameters['upper bound'] = st.number_input(
                        label='clip range', help='Remove the extreme value beyond the range [10000, 10000000]',
                        format='%d', min_value=10000, value=10000000, step=1000
                    )
            elif method_name == 'Quantization':
                parameters['num_bins'] = st.number_input(
                    label='the number of bins', help='Larger values result in finer grained discretization, while smaller values result in coarser grained discretization [1, 10000]',
                    min_value=1, value=10000, step=50
                )
            elif method_name == 'Smoothing':
                # learning_rate
                parameters['window_size'] = st.number_input(
                    label='the size of the smoothing window', help='A smaller window size will be more sensitive to detecting details and noise in the data, while a larger window size will smooth the data and reduce the impact of noise.[1, 9], should be odd.',
                    format='%d', min_value=1, value=9, step=2,
                )
        return parameters



    #  -------------------------  获取当前脚本的目录  -------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))

    save_path = None
    save_path_test = None
    save_path_train = None

    # -------------------------  创建头部导航容器  -------------------------
    header_container = st.container()
    with header_container:
        # st.markdown("## 分布式系统诊断系统")
        st.markdown("## 分布式医疗诊断系统")
        # st.write("欢迎来到分布式医疗诊断系统在线训练软件！")
        # 在这里添加其他导航元素，如链接、按钮等

        # 添加visualization_option
        st.markdown("### 模型选项")
        st.session_state['model_selection'] = st.selectbox(
            label="请选择你想要使用的机器学习模型",
            options=('LightGBM',  # LightGBM_model
                     'GBDT',  # GBDT_model
                     'XGBoost',  # XGBoost_model
                     'SVM',  # SVM_model
                     'Random Forest',  # RandomForest_model
                     'Gaussian Naive Bayes',  # GaussianNaiveBayes_model
                     'KNN',  # KNN_model
                     'LogisticRegression'),
            help="推荐使用LightGBM模型",
            format_func=lambda x: x+'分类器',  # 在每个选项后面加上”分类器“
        ).replace(' ', '')  # 去掉空格

        st.markdown("### 训练模型的参数选项")
        st.session_state['model_parameters'] = choose_params(st.session_state.get('model_selection'))

        st.markdown("### 增强鲁棒性的扰动方法")
        st.session_state['robust_method'] = st.selectbox(
            label="请选择你想要使用的扰动方法",
            options=("None",
                     "Add Random Noise",
                     "Clip",
                     "Quantization",
                     "Smoothing")
        ).replace(' ', '')

        st.markdown("### 扰动方法的参数选项")
        st.session_state['robust_params'] = choose_robust_parameters(st.session_state.get('robust_method'))

        st.markdown("### Train or Test?")
        visualization_option = st.radio(
            "请先在对应的位置上传对应的数据，然后选择对应的可视化选项",
            ("Training Results", "Testing Results"),
            horizontal=True,
        )


    # -------------------------  创建侧边栏  -------------------------
    sidebar = st.sidebar
    # 添加图片到侧边栏最上方
    _, col, _ = sidebar.columns([1] + [4] + [1])
    image_path = './imgs/healyou.png'
    col.image(image_path, use_column_width=True)

    # -------------------------  训练  -------------------------

    # 有没有方法可以不用每次都重新加载全部文件，那些图片的加载还挺消耗时间的

    # 上传训练数据
    train_expander = sidebar.expander("1. Training", expanded=True)
    train_expander.title("1. Upload Training Data")
    uploaded_training_data = train_expander.file_uploader(label='Choose a CSV file', key="train")

    if uploaded_training_data is not None and uploaded_training_data.name.endswith('.csv'):  # 如果是csv结尾的
        st.sidebar.success("Training data uploaded successfully!")
        # 保存上传的训练数据到当前目录
        save_path_train = os.path.join(script_dir, 'users\\{}\\data'.format(username), 'train.csv')
        if not os.path.exists(os.path.join(script_dir, 'users\\{}\\data'.format(username))):
            os.makedirs(os.path.join(script_dir, 'users\\{}\\data'.format(username)))
        with open(save_path_train, 'wb') as f:
            f.write(uploaded_training_data.getbuffer())

        #

        # 显示"Start Training"按钮
        start_training_button = sidebar.empty()
        # 显示训练集的标签分布情况
        if visualization_option == "Training Results":
            update_head(None, mode='train')
            train_metrics = st.session_state.get('train_metrics', None)

            # 可以开始训练模型
            if start_training_button.button("Start Training"):
                train_model(save_path_train, username)
            elif train_metrics != None:
                update_head(train_metrics, mode='train')

    elif uploaded_training_data is not None and not uploaded_training_data.name.endswith('.csv'):  # 如果不是csv结尾的
        # 上传文件类型必须是csv
        st.sidebar.error("Invalid file format. Only CSV files are allowed.")

    # -------------------------  下载训练模型  -------------------------

    model_expander = sidebar.expander("2. Download Training Model")
    model_expander.title("2. Select and Download Training Model")

    model_dir = './users/{}/models'.format(username)  # 返回所有模型的列表，如果没有模型返回空列表
    trained_models = os.listdir(model_dir)
    if trained_models == []:
        model_expander.warning('You need to train a model first.')
    else:
        model = model_expander.selectbox(
            label='choose the model you want to download',
            options=trained_models,
            format_func=lambda x: x.split('.')[0]
        )
        down_load_model(username, model.split('.')[0].split('_')[0], model_expander)

    # -------------------------  测试  -------------------------
    # 上传测试样本
    test_expander = sidebar.expander("3. Testing")
    test_expander.title("3. Upload Test Samples")
    uploaded_test_samples = test_expander.file_uploader('Choose a CSV file', key="test")

    if uploaded_test_samples is not None and uploaded_test_samples.name.endswith('.csv'):
        st.sidebar.success("Test samples uploaded successfully!")
        # 保存上传的测试样本到当前目录
        save_path_test = os.path.join(script_dir, 'users\\{}\\data'.format(username), 'test.csv')
        with open(save_path_test, 'wb') as f:
            f.write(uploaded_test_samples.getbuffer())

        # 显示"Start Testing"按钮
        start_testing_button = sidebar.empty()
        # 可视化
        if visualization_option == "Testing Results":
            update_head(None, mode='test')
            test_metrics = st.session_state.get('test_metrics', None)

            if start_testing_button.button("Start Testing"):
                # 测试模型
                try:
                    test_model(save_path_test, username)
                except FileNotFoundError:
                    st.warning('必须先进行所选模型的训练，才能进行模型预测')
            elif test_metrics != None:
                update_head(test_metrics, mode='test')

    elif uploaded_test_samples is not None and not uploaded_test_samples.name.endswith('.csv'):
        # 上传文件类型必须是csv
        st.sidebar.error("Invalid file format. Only CSV files are allowed.")

    # -------------------------  下载预测结果  -------------------------

    pred_expander = sidebar.expander("4. Download Testing Result")
    pred_expander.title("4. Select and Download Testing Result")

    pred_dir = './users/{}/data'.format(username)  # 返回所有模型的列表，如果没有模型返回空列表
    pred_results = os.listdir(pred_dir)
    if 'train.csv' in pred_results:
        pred_results.remove('train.csv')
    if 'test.csv' in pred_results:
        pred_results.remove('test.csv')

    if pred_results == []:
        pred_expander.warning('You need to test a model first.')
    else:
        test_file = pred_expander.selectbox(
            label='choose the test file you want to download',
            options=pred_results,
            format_func=lambda x: x.split('.')[0]
        )
        down_load_test_csv(username, test_file.split('.')[0].split('_')[0], pred_expander)
