import argparse
import os.path

# ------------------  可选模型  ------------------
import lightgbm as lgbm
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    confusion_matrix, accuracy_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def add_random_noise(data, noise_scale):
    noise = np.random.normal(loc=0, scale=noise_scale, size=data.shape)
    noisy_data = data + noise
    return noisy_data


def clip_data(data, lower_bound, upper_bound):
    clipped_data = np.clip(data, lower_bound, upper_bound)
    return clipped_data


def flatten_nested_object(obj):
    if isinstance(obj, (list, tuple)):
        flattened = np.concatenate([flatten_nested_object(item) for item in obj])
    else:
        flattened = np.array(obj)
    return flattened


def quantize_data(data, num_bins):
    data = flatten_nested_object(data)
    quantize_data = np.digitize(data, np.linspace(np.min(data),np.max(data), num_bins))
    return quantize_data


def smooth_data(data, window_size):

    from scipy.signal import medfilt

    smoothed_data = medfilt(data, kernel_size=window_size)
    return smoothed_data


def evaluate_multiclass_model(model, X, y):

    if not isinstance(X, np.ndarray):
        X = X.values
    else:
        X = X
    y = y.values

    accuracy = []

    label = []
    pred = []

    cv = StratifiedKFold(n_splits=10, shuffle=True)

    for train, test in cv.split(X, y):
        model.fit(X[train], y[train])
        label.extend(y[test])
        pred.extend(model.predict_proba(X[test]))
        y_pred = model.predict(X[test])
        accuracy.append(accuracy_score(y[test], y_pred))
        print("本次训练准确度为：", accuracy[-1])

    pred = np.array(pred)  # Convert pred to numpy array

    conf_mat = confusion_matrix(label, np.argmax(pred, axis=1))
    print(conf_mat)

    Accuracy = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
    Precision = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
    Recall = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    metrics = {'Accuracy': Accuracy.mean(), 'Precision': Precision.mean(),
               'Recall': Recall.mean(), 'F1_score': F1_score.mean()}

    return model, metrics


def train_and_save(train_path, username, model_name, model_params, robust_method, robust_params):
    data = pd.read_csv(train_path)

    data.drop('sample_id', axis=1, inplace=True)
    data.drop(['feature1', 'feature20', 'feature32', 'feature54', 'feature57', 'feature60', 'feature64', 'feature65',
               'feature77', 'feature78', 'feature80', 'feature88', 'feature92', 'feature100'], axis=1, inplace=True)

    mean_values = data.mean()
    data_copy = data.fillna(mean_values)
    train_copy = data_copy.drop('label', axis=1)

    X = train_copy
    Y = data_copy['label']

    if model_name == 'LightGBM':
        if model_params['default']:
            clf = lgbm.LGBMClassifier(random_state=66, metric='None', n_jobs=10, objective='multiclass', learning_rate=0.2,
                                      reg_lambda=0.25, reg_alpha=0.3, is_unbalance=True, num_class=6, max_depth=-1,
                                      num_leaves=25, min_child_weight=0.0015,
                                      min_child_samples=14, feature_fraction=0.65, bagging_fraction=0.1,
                                      n_estimators=int(100))
        else:
            del model_params['default']
            clf = lgbm.LGBMClassifier(**model_params)

    elif model_name == 'GBDT':
        clf = GradientBoostingClassifier(**model_params)

    elif model_name == 'XGBoost':
        clf = XGBClassifier(**model_params)

    elif model_name == 'SVM':
        clf = SVC(**model_params)

    elif model_name == 'RandomForest':
        clf = RandomForestClassifier(**model_params)

    elif model_name == 'GaussianNaiveBayes':
        clf = GaussianNB()

    elif model_name == 'KNN':
        clf = KNeighborsClassifier(**model_params)

    elif model_name == 'LogisticRegression':
        clf = LogisticRegression()

    else:
        clf = lgbm.LGBMClassifier(random_state=66, metric='None', n_jobs=10, objective='multiclass', learning_rate=0.2,
                                      reg_lambda=0.25, reg_alpha=0.3, is_unbalance=True, num_class=6, max_depth=-1,
                                      num_leaves=25, min_child_weight=0.0015,
                                      min_child_samples=14, feature_fraction=0.65, bagging_fraction=0.1,
                                      n_estimators=int(100))


    if robust_method == 'AddRandomNoise':
        X = add_random_noise(X, **robust_params)

    elif robust_method == 'Clip':
        lower_bound, upper_bound = robust_params['lower bound'], robust_params['upper bound']
        X = clip_data(X, lower_bound, upper_bound)

    elif robust_method == "Quantization":
        X = quantize_data(data, **robust_params)

    elif robust_method == 'Smoothing':
        X = smooth_data(data, **robust_params)

    elif robust_method == 'None':
        X = X

    clf, metrics = evaluate_multiclass_model(clf, X, Y)

    if not os.path.exists('./users/{}/models'.format(username)):  # 如果没有models文件夹，则创建文件夹
        os.makedirs('./users/{}/models'.format(username))

    model_path = './users/{}/models/{}_model.pkl'.format(username, model_name)
    joblib.dump(clf, model_path)

    return metrics


def test_and_pred(test_path, username, model_name):
    # import lightgbm as lgbm
    # lgbm_clf = lgbm.LGBMClassifier(random_state=66, metric='None', n_jobs=10, objective='multiclass', learning_rate=0.2,
    #                                reg_lambda=0.25, reg_alpha=0.3, is_unbalance=True, num_class=6, max_depth=-1,
    #                                num_leaves=25, min_child_weight=0.0015,
    #                                min_child_samples=14, feature_fraction=0.65, bagging_fraction=0.1,
    #                                n_estimators=100,
    #                                )

    # 加载
    model_path = './users/{}/models/{}_model.pkl'.format(username, model_name)
    lgbm_clf = joblib.load(model_path)

    valid = pd.read_csv(test_path)

    output_raw = valid

    valid.drop('sample_id', axis=1, inplace=True)
    valid.drop(['feature1', 'feature20', 'feature32', 'feature54', 'feature57', 'feature60', 'feature64', 'feature65',
                'feature77', 'feature78', 'feature80', 'feature88', 'feature92', 'feature100'], axis=1, inplace=True)

    mean_values = valid.mean()
    valid_copy = valid.fillna(mean_values)

    try:
        val_copy = valid_copy.drop('label', axis=1)

        X_val = val_copy
        Y_val = valid_copy['label']

        X_val = X_val.values
        Y_val = Y_val.values

        val_proba = []
        val_proba.extend(lgbm_clf.predict_proba(X_val))
        val_pred = lgbm_clf.predict(X_val)

        val_proba = np.array(val_proba)

        val_conf_mat = confusion_matrix(Y_val, np.argmax(val_proba, axis=1))

        Accuracy_val = np.sum(val_conf_mat.diagonal()) / np.sum(val_conf_mat)
        Precision_val = np.diag(val_conf_mat) / np.sum(val_conf_mat, axis=0)
        Recall_val = np.diag(val_conf_mat) / np.sum(val_conf_mat, axis=1)
        F1_score_val = 2 * (Precision_val * Recall_val) / (Precision_val + Recall_val)

        metrics = {'Accuracy_val': Accuracy_val.mean(), 'Precision_val': Precision_val.mean(),
                   'Recall_val': Recall_val.mean(), 'F1_score_val': F1_score_val.mean()}
    except KeyError:  # 没有[‘label’]这一列，就不统计结果了，只做预测

        val_pred = lgbm_clf.predict(valid_copy)
        metrics = {'Accuracy_val': -1, 'Precision_val': -1,
                   'Recall_val': -1, 'F1_score_val': -1}

    output_pred = pd.DataFrame(val_pred, columns=['prediction_label'])
    output_pred.to_csv('./users/{}/data/{}_prediction_result.csv'.format(username, model_name))

    return metrics, val_pred





# if __name__ == "__main__":
#     main()
