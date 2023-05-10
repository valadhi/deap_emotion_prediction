# recheck feature accuracy from eda & hrv pickles

# investigate class imbalance and ways to better split to reduce lack of accuracy

# run hyperparam search for linear prediction

# linear prediction for all data
from core.main import get_ratings_second, collate_feature_pickles
from sklearn import svm
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn import model_selection
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid
import xgboost
import hyperopt as hp
import os.path
import json
from csv import writer
import numpy as np
import pandas as pd
# run feature selection with a minimal score increment for selecting other combination
########################################################################################################
            ############################### PARAMETERS #################################
########################################################################################################
randomForestParams = {
    # 'bootstrap': [True, False],
    # 'max_depth': list(range(10, 100, 40)) + [None],
    # 'max_features': ['sqrt'],
    # 'min_samples_split': [2, 5, 10],
    # 'n_estimators': list(range(200, 1800, 700)),
    'min_samples_leaf': [1, 4, 8, 16],
    'max_depth': [None, 10, 50, 60, 90],
    'max_features': ['sqrt'],
    'min_samples_split': [2, 5,7,10],
    'n_estimators': list(range(200, 1800, 400)),
}
### xgboost regressor hyper search
xgboostRegressorParameters = {
    # 'max_depth': list(range(3, 18, 2)),
    # 'gamma': list(range(1, 9, 3)),
    # 'reg_alpha': list(range(40, 180, 50)),
    # 'reg_lambda': list(range(0, 1)),
    # 'colsample_bytree': list(range(0.5, 1)),
    # 'min_child_weight': list(range(0, 12, 2)),
    # 'n_estimators': [180],
    # 'seed': [0]
    'max_depth': [3, 5, 7, 9],
    'gamma': [7],
    'reg_alpha': list(range(40, 180, 40)),
    'reg_lambda': list(range(0, 1)),
    'colsample_bytree': [0.5, 0.7, 1],
    'min_child_weight': list(range(0, 12, 2)),
}
GradientBoostingParameters = {
    # 'n_estimators': list(range(5, 25, 5)),
    # 'max_depth': list(range(5, 16, 6)),
    # 'min_samples_split': list(range(200, 2100, 450)),
    # 'min_samples_leaf': list(range(30, 71, 35)),
    # 'max_features': list(range(7, 20, 7)),
    # 'subsample': [0.6, 0.7, 0.8, 0.85, 0.9]
    'n_estimators': [10],
    'max_depth': [5],
    'min_samples_split': list(range(200, 2100, 300)),
    'min_samples_leaf': list(range(30, 71, 10)),
    'max_features': [7, 9, 19],
    'subsample': [0.6, 0.7, 0.8, 0.85, 0.9],
}
SVRParameters = {
    # 'kernel': ['rbf'],
    # 'gamma': [1e-4, 1e-3, 0.01, 0.2, 0.6],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': [1e-4],
    'C': [1]
}

model_param_map_regression = [
    (ensemble.RandomForestRegressor(), randomForestParams, 'random forest regress'),
    (xgboost.XGBRegressor(), xgboostRegressorParameters, 'xgbooost regress'),
    (ensemble.GradientBoostingRegressor(), GradientBoostingParameters, 'gradien boost regress'),
    (svm.SVR(), SVRParameters, 'svm regress')
]

def write_to_csv(filename, to_add):
    csv_header = ['regressor_type', 'emotion_dimension', 'feature_group_criteria', 'group_id', 'hyperparams_json', 'score']
    csv_filename = filename
    if not os.path.isfile(csv_filename):
        with open(csv_filename, 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(csv_header)
            f_object.close()
    with open(csv_filename, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(to_add)
        f_object.close()

def get_hypersearch_size(params, cv_size=5):
    return str(cv_size * len(ParameterGrid(params)))

input = collate_feature_pickles(scale=True)
y_pred_arr, y_test_arr, feats_imp = [[] for i in range(3)]
kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)

def run_hyper_search(model, params, model_descr, group_name='none', group_id='none', regression=True):
    print("Doing search for " + str(model_descr) + " and group type " + str(group_name) + " with size " + get_hypersearch_size(params))
    for emotion in ['Valence', 'Arousal', 'Dominance', 'Liking']:
        if regression:
            target = get_ratings_second(emotion, linear=True)
            grid = model_selection.GridSearchCV(model, params, verbose=1, scoring="neg_mean_absolute_error")
            csv_filename = 'hyper_search_results.csv'
        else:
            target = get_ratings_second(emotion, linear=False)
            grid = model_selection.GridSearchCV(model, params, verbose=1, scoring="accuracy")
            csv_filename = 'hyper_search_classifier_results.csv'

        output = np.array(target)
        grid.fit(input, output)

        best_params = grid.best_params_
        model = grid.best_estimator_
        score = grid.best_score_
        # 'regressor_type', 'emotion_dimension', 'feature_group_criteria', 'group_id', 'hyperparams_json', 'score'
        results_row = [model_descr, emotion, group_name, group_id, json.dumps(best_params), str(score)]
        print(results_row)
        write_to_csv(csv_filename, results_row)

def get_ratings_group_highlow(emotion, target_col, target_val, input, classifier):
    exclude = True if target_col == 'video' else False
    groupby = 'participant' if target_col == 'video' else 'video'

    input = np.array(input)
    # print(input)
    # print(input.shape)

    target_val = int(target_val) + 1
    ratings = pd.read_csv('metadata_csv/participant_ratings.csv')
    target_emotion = ratings[(ratings[target_col] == target_val)].sort_values(
        by=[groupby])[emotion].to_list()

    if exclude:
        target_emotion = [t for i, t in enumerate(target_emotion) if i not in [22]]

    if classifier:
        target_emotion = np.array(target_emotion)
        class_low_idxs = np.argwhere(target_emotion < 4.5).flatten()
        class_high_idxs = np.argwhere(target_emotion > 5.5).flatten()
        lenL = len(class_low_idxs)
        lenH = len(class_high_idxs)
        # print(lenL)
        # print(lenH)
        maxLen = max(lenL, lenH)
        if maxLen / (maxLen + min(lenL, lenH)) > 0.75:
            return [], []

        kept_idxs = np.concatenate((class_low_idxs, class_high_idxs))
        kept_idxs = np.sort(kept_idxs)
        output = np.arange(len(kept_idxs))
        # # print(output)
        # print(class_low_idxs)
        # print(class_high_idxs)
        # print(target_emotion)
        np.put(target_emotion, class_low_idxs, [0])
        np.put(target_emotion, class_high_idxs, [1])
        # print(target_emotion)
        # print(kept_idxs)
        output = np.take(target_emotion, kept_idxs)
        # print(output)
        # input = np.array(input)
        #
        # print(input)
        # print(input.shape)
        input = np.take(input, kept_idxs, axis=0)
        # print(input.shape)
        # print(input)
        # print(input.shape)
        # print(output.shape)
        # print(output)
        # sys.exit(0)
    else:
        output = target_emotion
    return input, output

def get_ratings_group(emotion, target_col, target_val, input, classifier):
    exclude = True if target_col == 'video' else False
    groupby = 'participant' if target_col == 'video' else 'video'
    target_val = int(target_val) + 1
    ratings = pd.read_csv('metadata_csv/participant_ratings.csv')

    target_emotion = ratings[(ratings[target_col] == target_val)].sort_values(
        by=[groupby])[emotion].to_list()
    if exclude:
        target_emotion = [t for i, t in enumerate(target_emotion) if i not in [22]]
    if classifier:
        # target_emotion = [0 if (float(e) < 5.0) else 1 for e in target_emotion]
        mean = np.array(target_emotion).mean()
        median = np.median(target_emotion)
        target_emotion = [0 if (float(e) < mean) else 1 for e in target_emotion]

    return input, target_emotion

def run_group_hyper_search(model, input, params, model_descr, group_name, group_id, regression):
    print("Doing search for " + str(model_descr) + " and group type " + str(group_name) + " with size " + get_hypersearch_size(params))
    facet_group_map = {'participant': 'video', 'video': 'participant'}
    # is_classifier = False
    is_classifier = not regression
    for emotion in ['Valence', 'Arousal', 'Dominance', 'Liking']:
        exclude = True if group_name == 'video' else False
        input, target = get_ratings_group(emotion, group_name, group_id, input, classifier=is_classifier)
        # this_input, target = get_ratings_group_highlow(emotion, group_name, group_id, input, classifier=is_classifier)

        if len(this_input) == 0:
            continue

        output = np.array(target)
        if regression:
            grid = model_selection.GridSearchCV(model, params, verbose=1, scoring="neg_mean_absolute_error")
            csv_group_filename = 'hyper_search_results_group.csv'
        else:
            scoring = ['accuracy', 'f1']
            grid = model_selection.GridSearchCV(
                model, params, verbose=1, scoring=scoring, refit='accuracy')
            csv_group_filename = 'hyper_search_classifier_results_group_mean.csv'

# RECOVER ROC AUC SCORE FROM GRID SEARCH AND INCLUDE IN CSV


        grid.fit(this_input, output)
        best_params = grid.best_params_
        print(best_params)
        print(type(best_params))
        print([type(x) for k, x in best_params.items()])
        best_params = json.dumps(best_params)

        model = grid.best_estimator_
        score = grid.best_score_
        # 'regressor_type', 'emotion_dimension', 'feature_group_criteria', 'group_id', 'hyperparams_json', 'score'
        results_row = [model_descr, emotion, group_name, group_id, json.dumps(best_params), str(score)]
        write_to_csv(csv_group_filename, results_row)

########################################################################################################
            ############################### REGRESSORS #################################
########################################################################################################
# model = ensemble.RandomForestRegressor()
# run_hyper_search(model=model, params=randomForestParams, model_descr='random_forest', regression=True)
# model = xgboost.XGBRegressor()
# run_hyper_search(model=model, params=xgboostRegressorParameters, model_descr='xgboost_regressor', regression=True)
# model = ensemble.GradientBoostingRegressor()
# run_hyper_search(model=model, params=GradientBoostingParameters, model_descr='gradientboost_regressor', regression=True)
# model = svm.SVR()
# run_hyper_search(model=model, params=SVRParameters, model_descr='svr_regressor', regression=True)

def yield_group_data(model, params, descr, regression):
    for group_name in ['participant', 'video']:
        hr_path = "hr_features.pkl"
        eda_path = "eda_features.pkl"
        if os.path.isfile(hr_path):
            hr_features_frame = pd.read_pickle(hr_path)
        if os.path.isfile(eda_path):
            eda_features_frame = pd.read_pickle(eda_path)

        eda_features_frame = eda_features_frame.drop(columns=['participant', 'video'])
        full_feature_data = pd.concat([hr_features_frame, eda_features_frame], axis=1)
        full_feature_data = full_feature_data.fillna(0)
        clean_frame = full_feature_data.sort_values(['participant', 'video'])

        for video_id, group_data in clean_frame.groupby([group_name]):
            video_id = video_id[0]

            print("{0} {1}".format(group_name, video_id))
            feats = group_data.drop(columns=['participant', 'video'])
            scaler = preprocessing.StandardScaler()
            feats[feats.columns] = scaler.fit_transform(feats[feats.columns])
            input_data = feats.values.tolist()
            input_data = np.array(input_data)

            print("Running hyper search for groups")
            print(input_data.shape)


            run_group_hyper_search(model=model, input=input_data,
                                   params=params, model_descr=descr,
                                   group_name=group_name, group_id=video_id, regression=regression)


if __name__ == "__main__":
    # df = pd.read_csv("hyper_search_classifier_results_group_mean.csv")
    # df = pd.read_csv("hyper_search_classifier_feature_reduction_group.csv")
    # df = pd.read_csv("hyper_search_feature_reduction_group.csv")
    #
    # maxframe = {'emotion_dimension': [], 'feature_group_criteria': [], 'group_id': [], 'score': []}
    # for x, i in df.groupby(['emotion_dimension', 'feature_group_criteria', 'group_id']):
    #     maxframe['score'].append(i.score.max())
    #     maxframe['feature_group_criteria'].append(i.feature_group_criteria.iloc[0])
    #     maxframe['group_id'].append(i.group_id.iloc[0])
    #     maxframe['emotion_dimension'].append(i.emotion_dimension.iloc[0])
    #
    # pd.set_option('display.max_rows', 500)
    # pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 1000)
    #
    # df2 = pd.DataFrame.from_dict(maxframe).reset_index()
    # print(df2.groupby(['emotion_dimension', 'feature_group_criteria'])['score'].mean())
    # sys.exit(0)

    for model, params, descr in model_param_map_regression:
        run_hyper_search(model=model, params=params, model_descr=descr, regression=True)

    for conf in model_param_map_regression:
        model = conf[0]
        params = conf[1]
        descr = conf[2]
        yield_group_data(model, params, descr, regression=True)
# for group_name in ['participant', 'video']:
#     hr_path = "hr_features.pkl"
#     eda_path = "eda_features.pkl"
#     if os.path.isfile(hr_path):
#         hr_features_frame = pd.read_pickle(hr_path)
#     if os.path.isfile(eda_path):
#         eda_features_frame = pd.read_pickle(eda_path)
#
#     eda_features_frame = eda_features_frame.drop(columns=['participant', 'video'])
#     full_feature_data = pd.concat([hr_features_frame, eda_features_frame], axis=1)
#     full_feature_data = full_feature_data.fillna(0)
#     clean_frame = full_feature_data.sort_values(['participant', 'video'])
#
#     for video_id, group_data in clean_frame.groupby([group_name]):
#         video_id = video_id[0]
#
#         print("{0} {1}".format(group_name, video_id))
#         feats = group_data.drop(columns=['participant', 'video'])
#         scaler = preprocessing.StandardScaler()
#         feats[feats.columns] = scaler.fit_transform(feats[feats.columns])
#         input_data = feats.values.tolist()
#
#         for conf in model_param_map_regression:
#             model = conf[0]
#             params = conf[1]
#             descr = conf[2]
#             run_group_hyper_search(model=model, input=input_data,
#                              params=params, model_descr=descr,
#                              group_name=group_name, group_id=video_id, regression=False)


        # model = ensemble.RandomForestRegressor()
        # model = ensemble.RandomForestClassifier()
        # run_group_hyper_search(model=model, input=input_data, params=randomForestClassifierParams, model_descr='random_forest', group_name=group_name, group_id=video_id)
        # model = xgboost.XGBRegressor()
        # model = xgboost.XGBClassifier()
        # run_group_hyper_search(model=model, input=input_data, params=xgboostClassifierParameters, model_descr='xgboost_regressor', group_name=group_name, group_id=video_id)
        # model = ensemble.GradientBoostingRegressor()
        # model = ensemble.GradientBoostingClassifier()
        # run_group_hyper_search(model=model, input=input_data, params=GradientBoostingClassifierParameters, model_descr='gradientboost_regressor', group_name=group_name, group_id=video_id)
        # model = svm.SVR()
        # model = svm.SVC()
        # run_group_hyper_search(model=model, input=input_data, params=SVCParameters, model_descr='svr_regressor', group_name=group_name, group_id=video_id)
