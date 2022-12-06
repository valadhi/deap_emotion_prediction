import os
from join_frames_evaluate import get_ratings
from join_frames_evaluate import facet_group_map

from core.main import get_exclude_participant
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

## gridsearch random forest
from sklearn.ensemble import RandomForestClassifier
frame_columns = ['emotion', 'mae', 'best_features']
scaler = preprocessing.StandardScaler()

clean_frame = clean_frame.sort_values(['participant', 'video'])
feats = clean_frame.drop(columns=['participant', 'video'])
print(feats.shape)
feats[feats.columns] = scaler.fit_transform(feats[feats.columns])
input_data = feats.values.tolist()
for emotion in ['Valence', 'Arousal', 'Dominance', 'Liking']:
    ratings = pd.read_csv('metadata_csv/participant_ratings.csv')
    target_emotion = ratings[~ratings.participant.isin(get_exclude_participant())].sort_values(['participant', 'video'])[
        emotion].to_list()
    is_classifier = True
    if is_classifier:
        target_emotion = [0 if (float(e) < 5.0) else 1 for e in target_emotion]

    input = np.array(input_data)
    output = np.array(target_emotion)

    # input = input[:10]
    # output = output[:10]

    model = RandomForestClassifier()

    parameters = {'bootstrap': [True, False],
     # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
     'max_depth': [10, 70, 80, 100],
     # 'max_depth': [70, None],
     # 'max_features': [None, 'sqrt'],
     'max_features': ['sqrt'],
     'min_samples_leaf': [1, 2, 4],
     # 'min_samples_leaf': [2],
     'min_samples_split': [2, 5, 10],
     # 'min_samples_split': [2, 5],
     # 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
     'n_estimators': [100, 400, 600, 1200]}
    clf = GridSearchCV(model, parameters)
    clf.fit(input, output)

    print(clf.cv_results_)
    print(clf.best_estimator_)
    print(clf.best_params_)

