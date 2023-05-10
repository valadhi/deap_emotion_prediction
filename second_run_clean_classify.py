from core.main import get_ratings_second, collate_feature_pickles
from second_run_clean_regression import yield_group_data, run_hyper_search
from sklearn import svm
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn import model_selection
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid
import xgboost

randomForestClassifierParams = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9],
}
xgboostClassifierParameters = {
    'max_depth': [3, 5, 9],
    'gamma': [7],
    'reg_alpha': list(range(40, 180, 50)),
    'reg_lambda': list(range(0, 1)),
    'colsample_bytree': [0.5, 1],
    'min_child_weight': list(range(0, 12, 2)),}
GradientBoostingClassifierParameters = {
    "n_estimators":[5,50,250,500],
    "max_depth":[1,3,5,7,9],
    "learning_rate":[0.01,0.1,1,10,100]}
SVCParameters = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': [1e-4, 1, 10],
    'C': [0.1, 1, 100]}
model_param_map_classifier = [
    (ensemble.RandomForestClassifier(), randomForestClassifierParams, 'random forest class'),
    (xgboost.XGBClassifier(), xgboostClassifierParameters, 'xgboost class'),
    (ensemble.GradientBoostingClassifier(), GradientBoostingClassifierParameters, 'gradient boost class'),
    (svm.SVC(), SVCParameters, 'svm class')
]


if __name__ == "__main__":

    # from collections import Counter
    # for emotion in ['Valence', 'Arousal', 'Dominance', 'Liking']:
    #     print(emotion)
    #     exclude = False
    #     for group_id in range(32):
    #         input, target = get_ratings_group_highlow(
    #             emotion, 'participant', group_id, None, classifier=True)
    #         print(Counter(target))
    #
    #     exclude = True
    #     for group_id in range(40):
    #         input, target = get_ratings_group_highlow(
    #             emotion, 'video', group_id, None, classifier=True)
    #         print(Counter(target))

    for model, params, descr in model_param_map_classifier:
        yield_group_data(model, params, descr, regression=False)

    for model, params, descr in model_param_map_classifier:
        run_hyper_search(model=model, params=params, model_descr=descr,
                         regression=False)


