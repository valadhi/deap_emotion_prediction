import pandas as pd
import numpy as np
import os, sys
import core.main as main
from sklearn import preprocessing

gname_map = {'participant': 'Participant_id', 'video': 'Experiment_id'}
gname_sortmap = {'participant': 'Experiment_id', 'video': 'Participant_id'}

CLASSIFIER = 'classifier'
LINEAR = 'linear'
BEST_FEATURES = 'bestscore'
ALL_FEATURES = 'allfeatures'
REGRESSION = 'regression'
FOREST = 'forest'
XGBOOST = 'xgboost'
facet_group_map = {'participant': 'video', 'video': 'participant'}

def run_tests_for_emotions(clean_frame, group_name, estimator):
    feature_importance_data = []
    for video_id, group_data in clean_frame.groupby([group_name]):
        feats = group_data.drop(columns=['participant', 'video'])

        # scale values
        scaler = preprocessing.StandardScaler()
        feats[feats.columns] = scaler.fit_transform(feats[feats.columns])
        input_data = feats.values.tolist()
        feature_names = feats.columns.values
        print(feature_names)
        for emotion in ['Valence', 'Arousal', 'Dominance', 'Liking']:
            ratings = pd.read_csv('metadata_csv/participant_ratings.csv')
            target_emotion = ratings[(ratings[gname_map[group_name]] == 1)].sort_values(
                by=[gname_sortmap[group_name]])[emotion].to_list()

            test_score, importance = getattr(main, estimator)(input_data, target_emotion)
            metric_names = list(test_score.keys())
            metric_values = list(test_score.values())

            feature_importance_data.append([video_id, emotion] + metric_values
                                           # + importance
                                           )
    results_frame = pd.DataFrame(feature_importance_data,
                                 columns=['video_id', 'emotion'] + metric_names
                                         # + feature_names.tolist()
                                 )
    return results_frame

def run_tests_for_emotions_feat_selector(clean_frame, group_name, estimator, is_classifier, best_features=True):
    feature_importance_data = []
    frame_columns = ['video_id', 'emotion', 'mae', 'best_features']
    for video_id, group_data in clean_frame.groupby([group_name]):
        print("{0} {1}".format(group_name, video_id))
        # continue
        feats = group_data.drop(columns=['participant', 'video'])

        # scale values
        scaler = preprocessing.StandardScaler()
        feats[feats.columns] = scaler.fit_transform(feats[feats.columns])
        input_data = feats.values.tolist()
        feature_names = feats.columns.values
        # print(feature_names)
        for emotion in ['Valence', 'Arousal', 'Dominance', 'Liking']:
            exclude = True if group_name == 'video' else False
            target_emotion = get_ratings(emotion, group_name, video_id, facet_group_map[group_name],
                                         exclude, binary=is_classifier)

            if best_features:
                test_score, feat_idx = getattr(main, estimator)(input_data, target_emotion)
                best_features = [', '.join([feature_names[i] for i in feat_idx])]
                feature_importance_data.append([video_id, emotion, test_score, best_features])
            else:
                test_score, importance = getattr(main, estimator)(input_data, target_emotion)
                metric_names = list(test_score.keys())
                metric_values = list(test_score.values())
                feature_importance_data.append([video_id, emotion] + metric_values)
                frame_columns = ['video_id', 'emotion'] + metric_names

    results_frame = pd.DataFrame(feature_importance_data, columns=frame_columns)
    return results_frame

def get_ratings(emotion, target_col, target_val, groupby, exclude, binary=False):
    target_val = int(target_val) + 1
    ratings = pd.read_csv('metadata_csv/participant_ratings.csv')
    target_emotion = ratings[(ratings[target_col] == target_val)].sort_values(
        by=[groupby])[emotion].to_list()
    if exclude:
        target_emotion = [t for i, t in enumerate(target_emotion) if i not in main.exclude_participant]
    if binary:
        target_emotion = [0 if (float(e) < 5.0) else 1 for e in target_emotion]
    return target_emotion


config = [
    # # # # # # # # # # # # LINEARS # # # # # # # # # # # # #
    # # # # all features # # # #
    # {'path': 'participant_all_feats_forest.csv',
    #  'grouping': 'participant', 'estimator': 'run_test_forest'},
    # {'path': 'video_all_feats_forest.csv',
    #  'grouping': 'video', 'estimator': 'run_test_forest'},
    # {'path': 'participant_all_feats_xgboost.csv',
    #  'grouping': 'participant', 'estimator': 'run_test_xgboost'},
    # {'path': 'video__all_feats_xgboost.csv',
    #  'grouping': 'video', 'estimator': 'run_test_xgboost'},
    # {'path': 'participant_all_feats_regression.csv',
    #  'grouping': 'participant', 'estimator': 'run_test_regression'},
    # {'path': 'video__all_feats_regression.csv',
    #  'grouping': 'video', 'estimator': 'run_test_regression'},
    # # # # best features # # # #
    # {'path': 'participant_best_score_regression.csv',
    #  'grouping': 'participant', 'estimator': 'run_test_regression_feature_selector'},
    # {'path': 'video_best_score_regression.csv',
    #  'grouping': 'video', 'estimator': 'run_test_regression_feature_selector'},
    # {'path': 'participant_best_score_forest.csv',
    #  'grouping': 'participant', 'estimator': 'run_test_forest_feature_selector'},
    # {'path': 'video_best_score_forest.csv',
    #  'grouping': 'video', 'estimator': 'run_test_forest_feature_selector'},
    {'path': 'participant_best_score_xgboost.csv',
     'grouping': 'participant', 'estimator': 'run_test_xgboost_feature_selector'},
    {'path': 'video_best_score_xgboost.csv',
     'grouping': 'video', 'estimator': 'run_test_xgboost_feature_selector'},
    # # # # # # # # # # # CLASSIFIERS # # # # # # # # # # # # #
    # # # # all features # # # #
    # # # # best features # # # #
]

estimator_maps = {
    CLASSIFIER: {
        ALL_FEATURES: {REGRESSION: 'run_classify_regression', FOREST: 'run_classify_forest', XGBOOST: 'run_classify_xgboost'},
        BEST_FEATURES: {REGRESSION: 'run_classify_regression_best_features', FOREST: 'run_classify_forest_best_features', XGBOOST: 'run_classify_xgboost_best_features'}
    },
    LINEAR: {
        ALL_FEATURES: {REGRESSION: 'run_test_regression', FOREST: 'run_test_forest', XGBOOST: 'run_test_xgboost'},
        BEST_FEATURES: {REGRESSION: 'run_test_regression_feature_selector', FOREST: 'run_test_forest_feature_selector', XGBOOST: 'run_test_xgboost_feature_selector'},
    }}


# if os.path.isfile(c['path']):
#     feats_by_participant = pd.read_csv(c['path'])
#     print(feats_by_participant)
# else:
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

for e in [LINEAR]:
    for f in [ALL_FEATURES, BEST_FEATURES]:
        for m in [REGRESSION, FOREST, XGBOOST]:
            for g in ['participant', 'video']:

                path = os.path.join('results', '_'.join([g, f, e, m]) + '.csv')
                if os.path.isfile(path): continue

                best_features = True if f == BEST_FEATURES else False
                is_classifier = True if e == CLASSIFIER else False
                results_frame = run_tests_for_emotions_feat_selector(clean_frame, g, estimator_maps[e][f][m],
                                                                     is_classifier=is_classifier, best_features=best_features)

                results_frame.to_csv(path)



