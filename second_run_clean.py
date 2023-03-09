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

def two_scorer():
    return make_scorer(two_score, greater_is_better=True)
from core.main import *
from sklearn import svm
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn import model_selection
from sklearn import ensemble
import xgboost
import hyperopt as hp
#####################################################################################
        ############################### SVM ##################################
#####################################################################################
input = collate_feature_pickles(scale=True)


y_pred_arr, y_test_arr, feats_imp = [[] for i in range(3)]
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

for emotion in ['Valence', 'Arousal', 'Dominance', 'Liking']:
    target = get_ratings(emotion, linear=True)
    output = np.array(target)

    parameters = {
        # 'bootstrap': [True, False],
         # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
         # 'max_depth': [10,  100],
         # 'max_features': ['sqrt'],
         # 'min_samples_leaf': [1, 2, 4],
         # 'min_samples_split': [2, 5, 10],
         # 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
         # 'n_estimators': [200]
    }
    model = ensemble.RandomForestRegressor()

    parameters = {'max_depth': list(np.arange(3, 18, 4)),
        'gamma': list(np.arange(1, 9, 2)),
        'reg_alpha' : list(np.arange(40, 180, 10)),
        'reg_lambda' : list(np.arange(0, 1)),
        'colsample_bytree' : list(np.arange(0.5, 1)),
        'min_child_weight' : list(np.arange(0, 10, 2)),
        'n_estimators': [180],
        'seed': [0]
    }
    model = xgboost.XGBRegressor()
    grid = model_selection.GridSearchCV(model, parameters, verbose=5,  scoring="neg_mean_squared_error")
    grid.fit(input, output)

    print(emotion)
    best_params = grid.best_params_
    model = grid.best_estimator_
    score = grid.best_score_
    print(score)
    print(grid.cv_results_)


