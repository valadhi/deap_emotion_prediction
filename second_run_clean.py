# recheck feature accuracy from eda & hrv pickles

# investigate class imbalance and ways to better split to reduce lack of accuracy

# run hyperparam search for linear prediction

# linear prediction for all data

# run feature selection with a minimal score increment for selecting other combination
def MSE(y_true,y_pred):
    mse = mean_squared_error(y_true, y_pred)
    print ('MSE: %2.3f' % mse)
    return mse

def R2(y_true,y_pred):
     r2 = r2_score(y_true, y_pred)
     print ('R2: %2.3f' % r2)
     return r2

def two_score(y_true,y_pred):
    MSE(y_true,y_pred) #set score here and not below if using MSE in GridCV
    score = R2(y_true,y_pred)
    return score

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

def two_scorer():
    return make_scorer(two_score, greater_is_better=True)
from core.main import *
from sklearn import svm
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn import model_selection
from sklearn import ensemble
from sklearn.model_selection import ParameterGrid
import xgboost
import hyperopt as hp
import os.path
import json
from csv import writer

input = collate_feature_pickles(scale=True)
y_pred_arr, y_test_arr, feats_imp = [[] for i in range(3)]
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

randomForestParams = {
    'bootstrap': [True, False],
    'max_depth': list(np.arange(10, 100, 40)) + [None],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': list(np.arange(200, 2000, 500)),
}
### xgboost regressor hyper search
xgboostRegressorParameters = {
    'max_depth': list(np.arange(3, 18, 6)),
    'gamma': list(np.arange(1, 9, 3)),
    'reg_alpha': list(np.arange(40, 180, 50)),
    'reg_lambda': list(np.arange(0, 1)),
    'colsample_bytree': list(np.arange(0.5, 1)),
    'min_child_weight': list(np.arange(0, 10, 3)),
    'n_estimators': [180],
    'seed': [0]
}
GradientBoostingParameters = {
    'n_estimators': range(20, 81, 30),
    'max_depth': range(5, 16, 6),
    'min_samples_split': range(200, 2100, 450),
    'min_samples_leaf': range(30, 71, 35),
    'max_features': range(7, 20, 7),
    'subsample': [0.6, 0.7, 0.8, 0.85, 0.9]
}
SVRParameters = {
    'kernel': ['rbf'],
    'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],
    'C': [1, 10, 100, 1000, 10000]
}
GaussianProcessParameters = {}
BayesianRidgeParameters = {}
KNeighborsParameters = {}

print("xgboostRegressorParameters")
print(5 * len(ParameterGrid(xgboostRegressorParameters)))
print("randomForestParams")
print(5 * len(ParameterGrid(randomForestParams)))
print("GradientBoostingParameters")
print(5 * len(ParameterGrid(GradientBoostingParameters)))
print("SVRParameters")
print(5 * len(ParameterGrid(SVRParameters)))
# print("GaussianProcessParameters")
# print(5 * len(ParameterGrid(GaussianProcessParameters)))
# print("BayesianRidgeParameters")
# print(5 * len(ParameterGrid(BayesianRidgeParameters)))
# print("KNeighborsParameters")
# print(5 * len(ParameterGrid(KNeighborsParameters)))

# def get_ratings_group(emotion, target_col, target_val, groupby, exclude, binary=False):
#     target_val = int(target_val) + 1
#     ratings = pd.read_csv('metadata_csv/participant_ratings.csv')
#     target_emotion = ratings[(ratings[target_col] == target_val)].sort_values(
#         by=[groupby])[emotion].to_list()
#     if exclude:
#         target_emotion = [t for i, t in enumerate(target_emotion) if i not in main.exclude_participant]
#     if binary:
#         target_emotion = [0 if (float(e) < 5.0) else 1 for e in target_emotion]
#     return target_emotion
# csv_group_filename = 'hyper_search_results_group.csv'
# def run_group_hyper_search(model, params, model_descr, group_name, group_id):
#     facet_group_map = {'participant': 'video', 'video': 'participant'}
#     is_classifier = False
#     for emotion in ['Valence', 'Arousal', 'Dominance', 'Liking']:
#         exclude = True if group_name == 'video' else False
#         target = get_ratings_group(emotion, group_name, video_id, facet_group_map[group_name], exclude, binary=is_classifier)
#         output = np.array(target)
#         grid = model_selection.GridSearchCV(model, params, verbose=1, scoring="neg_mean_squared_error")
#         grid.fit(input, output)
#         best_params = grid.best_params_
#         model = grid.best_estimator_
#         score = grid.best_score_
#         # 'regressor_type', 'emotion_dimension', 'feature_group_criteria', 'group_id', 'hyperparams_json', 'score'
#         results_row = [model_descr, emotion, group_name, group_id, json.dumps(best_params), str(score)]
#         write_to_csv(csv_group_filename, results_row)
#
# for group_name in ['participant', 'video']:
#     hr_path = "hr_features.pkl"
#     eda_path = "eda_features.pkl"
#     if os.path.isfile(hr_path):
#         hr_features_frame = pd.read_pickle(hr_path)
#     if os.path.isfile(eda_path):
#         eda_features_frame = pd.read_pickle(eda_path)
#     eda_features_frame = eda_features_frame.drop(columns=['participant', 'video'])
#     full_feature_data = pd.concat([hr_features_frame, eda_features_frame], axis=1)
#     full_feature_data = full_feature_data.fillna(0)
#     clean_frame = full_feature_data.sort_values(['participant', 'video'])
#     for video_id, group_data in clean_frame.groupby([group_name]):
#         print("{0} {1}".format(group_name, video_id))
#         model = ensemble.RandomForestRegressor()
#         run_group_hyper_search(model=model, params={}, model_descr='random_forest', group_name=group_name, group_id=video_id)
#         model = xgboost.XGBRegressor()
#         run_group_hyper_search(model=model, params={}, model_descr='xgboost_regressor', group_name=group_name, group_id=video_id)
#         model = sklearn.ensemble.GradientBoostingRegressor()
#         run_group_hyper_search(model=model, params={}, model_descr='xgboost_regressor', group_name=group_name, group_id=video_id)
#         model = sklearn.svm.SVR()
#         run_group_hyper_search(model=model, params={}, model_descr='xgboost_regressor', group_name=group_name, group_id=video_id)
#
# import sys
# sys.exit(0)

csv_filename = 'hyper_search_results.csv'
def run_hyper_search(model, params, model_descr, group_name='none', group_id='none'):
    print("Doing search for " + str(model_descr) + " and group type " + str(group_name))
    for emotion in ['Valence', 'Arousal', 'Dominance', 'Liking']:
        target = get_ratings_second(emotion, linear=True)
        output = np.array(target)

        grid = model_selection.GridSearchCV(model, params, verbose=1, scoring="neg_mean_squared_error")
        grid.fit(input, output)

        best_params = grid.best_params_
        model = grid.best_estimator_
        score = grid.best_score_
        # 'regressor_type', 'emotion_dimension', 'feature_group_criteria', 'group_id', 'hyperparams_json', 'score'
        results_row = [model_descr, emotion, group_name, group_id, json.dumps(best_params), str(score)]
        write_to_csv(csv_filename, results_row)
########################################################################################################
############################### sklearn.ensemble.RandomForestRegressor #################################
########################################################################################################
model = ensemble.RandomForestRegressor()
run_hyper_search(model=model, params=randomForestParams, model_descr='random_forest')
# run_hyper_search(model=model, params={}, model_descr='random_forest')
model = xgboost.XGBRegressor()
run_hyper_search(model=model, params=xgboostRegressorParameters, model_descr='xgboost_regressor')
# run_hyper_search(model=model, params={}, model_descr='xgboost_regressor')
model = sklearn.ensemble.GradientBoostingRegressor()
run_hyper_search(model=model, params=GradientBoostingParameters, model_descr='xgboost_regressor')
# run_hyper_search(model=model, params={}, model_descr='gradient_boosting')
model = sklearn.svm.SVR()
run_hyper_search(model=model, params=SVRParameters, model_descr='xgboost_regressor')
# run_hyper_search(model=model, params={}, model_descr='SVM_regressor')

# sklearn.ensemble.GradientBoostingRegressor
# sklearn.gaussian_process.GaussianProcessRegressor
# sklearn.linear_model.BayesianRidge
# sklearn.neighbors.KNeighborsRegressor
# sklearn.svm.SVR