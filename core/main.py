import numpy as np
import pandas as pd
import sklearn.metrics
import xgboost as xgb
import pickle, statistics
import sys
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ColumnSelector

exclude_participant = [22]
channels = {
'Fp1': 0, 'AF3': 1, 'F3': 2, 'F7': 3, 'FC5': 4, 'FC1': 5, 'C3': 6, 'T7': 7,
    'CP5': 8, 'CP1': 9, 'P3': 10, 'P7': 11, 'PO3': 12, 'O1': 13, 'Oz': 14,
    'Pz': 15, 'Fp2': 16, 'AF4': 17, 'Fz': 18, 'F4': 19, 'F8': 20, 'FC6': 21,
    'FC2': 22, 'Cz': 23, 'C4': 24, 'T8': 25, 'CP6': 26, 'CP2': 27, 'P4': 28,
    'P8': 29, 'PO4': 30, 'O2': 31, "hEOG": 32, "vEOG": 33, "Zygomaticus Major EMG": 34,
    "tEMG": 35,"GSR": 36, "Temperature": 39, "Respiration belt": 37, "Plethysmograph": 38
}
# print(sklearn.metrics.get_scorer_names())
# ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy',
#  'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
#  'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples',
#  'jaccard_weighted', 'matthews_corrcoef', 'max_error', 'mutual_info_score', 'neg_brier_score', 'neg_log_loss',
#  'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_mean_gamma_deviance',
#  'neg_mean_poisson_deviance', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error',
#  'neg_root_mean_squared_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro',
#  'precision_samples', 'precision_weighted', 'r2', 'rand_score', 'recall', 'recall_macro', 'recall_micro',
#  'recall_samples', 'recall_weighted', 'roc_auc', 'roc_auc_ovo','' 'roc_auc_ovo_weighted', 'roc_auc_ovr',
#  'roc_auc_ovr_weighted', 'top_k_accuracy', 'v_measure_score']

def make_metrics(y_true, y_pred):
    return {
        'mse': metrics.mean_squared_error(y_true, y_pred),
        'rmse': metrics.mean_squared_error(y_true, y_pred, squared=False),
        'mae': metrics.mean_absolute_error(y_true, y_pred),
        'coeff': metrics.r2_score(y_true, y_pred),
    }
def make_metrics_crossval(y_true_arr, y_pred_arr):
    mse, rmse, mae, coeff = [[] for i in range(4)]
    for idx, y_true in enumerate(y_true_arr):
        y_pred = y_pred_arr[idx]
        mse.append(metrics.mean_squared_error(y_true, y_pred))
        rmse.append(metrics.mean_squared_error(y_true, y_pred, squared=False))
        mae.append(metrics.mean_absolute_error(y_true, y_pred))
        coeff.append(metrics.r2_score(y_true, y_pred))
    return {
        'mse': np.mean(mse),
        'rmse': np.mean(rmse),
        'mae': np.mean(mae),
        'coeff': np.mean(coeff),
    }

def reduce_features_fold(model, X_train, X_test, y):
    sfs = SequentialFeatureSelector(model)
    sfs.fit(X_train, y)
    sfs.get_support()
    return sfs.transform(X_train), sfs.transform(X_test)
def reduce_features(model, X, y):
    sfs = SequentialFeatureSelector(model)
    sfs.fit(X, y)
    sfs.get_support()
    return sfs.transform(X)

def reduce_features_worker(model, input, output, best_features=True):
    features = (1, input.shape[1]) if best_features is True else input.shape[1]
    # print(features)
    sfs = SFS(model,
              k_features=features,
              forward=True,
              floating=False,
              scoring='neg_mean_absolute_error',
              cv=5)
    sfs.fit(input, output)
    selector_metrics = sfs.get_metric_dict()
    best_score_key = sorted(selector_metrics.keys(),
                            key=lambda x: (selector_metrics[x]['avg_score']), reverse=True)[0]
    return selector_metrics[best_score_key]['avg_score'], selector_metrics[best_score_key]['feature_idx']


# # # # # # # # # # # CLASSIFIERS # # # # # # # # # # # # #
def run_classify_regression(input, output):
    # print(np.array(input))
    # print(np.array(input).shape)
    # print(np.array(output))
    # print(np.array(output).shape)
    model = LogisticRegression()
    return reduce_features_worker(model, np.array(input), np.array(output), best_features=False)

def run_classify_forest(input, output):
    model = RandomForestClassifier(max_depth=2, random_state=0)
    return reduce_features_worker(model, np.array(input), np.array(output), best_features=False)

def run_classify_xgboost(input, output):
    model = xgb.XGBClassifier(objective="reg:squarederror", eval_metric='neg_mean_absolute_error')
    return reduce_features_worker(model, np.array(input), np.array(output), best_features=False)

def run_classify_regression_best_features(input, output):
    model = LogisticRegression()
    return reduce_features_worker(model, np.array(input), np.array(output))

def run_classify_forest_best_features(input, output):
    model = RandomForestClassifier(max_depth=2, random_state=0)
    return reduce_features_worker(model, np.array(input), np.array(output))

def run_classify_xgboost_best_features(input, output):
    model = xgb.XGBClassifier(objective="reg:squarederror", eval_metric='neg_mean_absolute_error')
    return reduce_features_worker(model, np.array(input), np.array(output))

# # # # # # # # # # # # REGRESSORS # # # # # # # # # # # # #
def run_test_regression_feature_selector(input, output):
    # group it with featureSelector because it has cross-validation included
    model = LinearRegression()
    return reduce_features_worker(model, np.array(input), np.array(output))

def run_test_forest_feature_selector(input, output):
    model = RandomForestRegressor(max_depth=2, random_state=0)
    return reduce_features_worker(model, np.array(input), np.array(output))

def run_test_xgboost_feature_selector(input, output):
    model = xgb.XGBRegressor(objective="reg:squarederror")
    return reduce_features_worker(model, np.array(input), np.array(output))

def run_test_regression(input, output):
    input = np.array(input)
    output = np.array(output)
    model = LinearRegression()

    # input = reduce_features(model, input, output)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_arr, y_test_arr, feats_imp = [[] for i in range(3)]
    for train_index, test_index in kfold.split(input):
        X_train, X_test = input[train_index], input[test_index]
        y_train, y_test = output[train_index], output[test_index]

        # X_train, X_test = reduce_features(model, X_train, X_test, y_train)

        reg = model.fit(X_train, y_train)
        feats_imp.append(model.coef_.tolist())
        y_test_arr.append(y_test.tolist())
        y_pred_arr.append(model.predict(X_test).tolist())

    metrics_dict = make_metrics_crossval(y_test_arr, y_pred_arr)
    feature_importance = np.array(feats_imp).mean(axis=0).tolist()
    # np.mean(scores), np.std(scores)
    return metrics_dict, feature_importance

def run_test_forest(input, output):
    input = np.array(input)
    output = np.array(output)
    model = RandomForestRegressor(max_depth=2, random_state=0)

    # input = reduce_features(model, input, output)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_arr, y_test_arr, feats_imp = [[] for i in range(3)]
    for train_index, test_index in kfold.split(input):
        print('trial')
        X_train, X_test = input[train_index], input[test_index]
        y_train, y_test = output[train_index], output[test_index]

        # X_train, X_test = reduce_features(model, X_train, X_test, y_train)

        reg = model.fit(X_train, y_train)
        feats_imp.append(model.feature_importances_.tolist())
        y_test_arr.append(y_test.tolist())
        y_pred_arr.append(model.predict(X_test).tolist())

    metrics_dict = make_metrics_crossval(y_test_arr, y_pred_arr)
    feature_importance = np.array(feats_imp).mean(axis=0).tolist()
    return metrics_dict, feature_importance

def run_test_xgboost(input, output):
    input = np.array(input)
    output = np.array(output)
    model = xgb.XGBRegressor(objective="reg:squarederror")

    # input = reduce_features(model, input, output)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_arr, y_test_arr, feats_imp = [[] for i in range(3)]
    for train_index, test_index in kfold.split(input):
        print('trial')
        X_train, X_test = input[train_index], input[test_index]
        y_train, y_test = output[train_index], output[test_index]


        model.fit(X_train, y_train)
        feats_imp.append(model.feature_importances_.tolist())
        y_test_arr.append(y_test.tolist())
        y_pred_arr.append(model.predict(X_test).tolist())

    metrics_dict = make_metrics_crossval(y_test_arr, y_pred_arr)
    feature_importance = np.array(feats_imp).mean(axis=0).tolist()
    return metrics_dict, feature_importance

def read_eeg_signal_from_file(filename):
    x = pickle._Unpickler(open(filename, 'rb'))
    x.encoding = 'latin1'
    p = x.load()
    return p

def extract_data_by_video():
    path = "./deap_entire_frame_32.pkl"
    print(os.path.isfile(path))
    if os.path.isfile(path):
        return pd.read_pickle(path)
    file_list = ['data_preprocessed_python/s{:02d}.dat'.format(i) for i in range(1, 33)]
    data_by_video = {}
    target = {}
    frame_rows = []
    for participant_id, file in enumerate(file_list):
        # print(participant_id)
        # print(file)
        file_contents = read_eeg_signal_from_file(file)
        # print(file_contents)
        labels = file_contents['labels']
        data = file_contents['data'].tolist()

        for video_id, video_data in enumerate(data):
            for chn in range(0, 40):
                frame_rows.append([participant_id, video_id, chn, video_data[chn],
                                 labels[video_id, 0], labels[video_id, 1], labels[video_id, 2], labels[video_id, 3]])

    frame = pd.DataFrame(frame_rows,
                         columns=['participant', 'video', 'channel_id', 'channel_data', 'Valence', 'Arousal', 'Dominance', 'Liking'])
    frame.to_pickle(path)
    return frame

def read_video_info(video_id):
    video_info = pd.read_csv("./metadata_csv/video_list.csv")
    row = video_info[video_info['Experiment_id'] == video_id].iloc[0]
    return row['Artist'] + " - " + row['Title']



