import os

from core.main import *
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

hr_path = "hr_features.pkl"
eda_path = "eda_features.pkl"
if os.path.isfile(hr_path):
    hr_features_frame = pd.read_pickle(hr_path)
if os.path.isfile(eda_path):
    eda_features_frame = pd.read_pickle(eda_path)

eda_features_frame = eda_features_frame.drop(columns=['participant', 'video'])
full_feature_data = pd.concat([hr_features_frame, eda_features_frame], axis=1)
df = full_feature_data
# what do about missing/failing feature values
clean_frame = df.fillna(0)

clean_frame = clean_frame.sort_values(['participant', 'video'])
feats = clean_frame.drop(columns=['participant', 'video'])
scaler = preprocessing.StandardScaler()
print(feats.shape)
feats[feats.columns] = scaler.fit_transform(feats[feats.columns])
input_data = feats.values.tolist()
# print(input_data)
# import sys
# sys.exit(0)



## gridsearch random forest
from xgboost import XGBClassifier
frame_columns = ['emotion', 'mae', 'best_features']

class_params = {
'Valence':  {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 200, 'random_state': None},
# Valence 0.5056451612903226
'Arousal': {'bootstrap': True, 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100, 'random_state': 0},
# Arousal 0.5653225806451613
'Dominance': {'bootstrap': True, 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100, 'random_state': 0},
# Dominance 0.6129032258064516
'Liking': {'bootstrap': True, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100, 'random_state': None}
# Liking 0.5967741935483871
}
feature_names = feats.columns.values
for emotion in ['Valence', 'Arousal', 'Dominance', 'Liking']:
    ratings = pd.read_csv('metadata_csv/participant_ratings.csv')
    target_emotion = ratings[~ratings.participant.isin([22])].sort_values(['participant', 'video'])[
        emotion].to_list()
    is_classifier = True
    if is_classifier:
        target_emotion = [0 if (float(e) < 5.0) else 1 for e in target_emotion]

    input = np.array(input_data)
    output = np.array(target_emotion)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred_arr, y_test_arr, feats_imp = [[] for i in range(3)]
    for train_index, test_index in kfold.split(input):
        X_train, X_test = input[train_index], input[test_index]
        y_train, y_test = output[train_index], output[test_index]

        model = XGBClassifier()
        model.set_params(**class_params[emotion])
        model.fit(X_train, y_train)

        feats_imp.append(model.feature_importances_.tolist())
        y_test_arr.append(y_test.tolist())
        y_pred_arr.append(model.predict(X_test).tolist())

    feature_importance = np.array(feats_imp).mean(axis=0).tolist()
    acc_score = make_metrics_accuracy_crossval(y_test_arr, y_pred_arr)
    print(emotion)
    print(acc_score)
    # continue
    # model = XGBClassifier()
    # parameters = {
    #     'n_estimators': [100, 1000],
    #     # 'booster': ['gbtree', 'gblinear'],
    #     'eta': [0.009, 0.01, 0.1],
    #     # 'eta': [0.01, 0.3],
    #     'gamma': [0, 1, 5, 10],
    #     # 'max_depth': [3, 6, 10, 20],
    #     'max_depth': [2, 3, 5],
    #     # 'min_child_weight': [0, 1],
    #     'max_delta_step': [0, 1, 2],
    #     'subsample': [0.4, 0.5, 0.6, 1],
    #     # 'colsample_bytree': [0.5, 0.7, 1]
    # }
    # clf = GridSearchCV(model, parameters)
    # clf.fit(input, output)
    #
    # # print(clf.cv_results_)
    # print(clf.best_estimator_)
    # print(clf.best_params_)
    # print(clf.best_score_)

# {'eta': 0.01, 'gamma': 10, 'max_delta_step': 1, 'max_depth': 2, 'n_estimators': 100, 'subsample': 0.5}
# 0.6072580645161291
