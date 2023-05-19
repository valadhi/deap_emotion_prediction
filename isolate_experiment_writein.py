from sklearn import ensemble
from sklearn import model_selection
from sklearn import svm

import matplotlib.pyplot as plt
from core.main import get_ratings_second, collate_feature_pickles
import numpy as np
import pandas as pd
import xgboost
import json

def crossval(model, input, emotion, feature_names):

	cv_val = 5
	target = get_ratings_second(emotion, linear=True)
	output = np.array(target)
	crossval_output = model_selection.cross_validate(
		model, input, output, cv=cv_val,
		scoring='neg_mean_absolute_error', return_estimator=True)

	print(crossval_output['test_score'].mean())

	aggregator = []
	for idx, estimator in enumerate(crossval_output['estimator']):
		# print("Features sorted by their score for estimator {}:".format(idx))
		featimp = estimator.feature_importances_
		aggregator.append(featimp)

	featsum = np.array(aggregator).sum(axis=0)
	feature_average = np.divide(featsum, cv_val)

	feature_importances = pd.DataFrame(feature_average,
	                                   index = feature_names,
	                                   columns=['importance']).sort_values('importance', ascending=False)
	return feature_importances

input, feature_names = collate_feature_pickles(scale=True, return_names=True)

model_params = pd.read_csv("hyper_search_results.csv")
model_params.sort_values('emotion_dimension', inplace=True)

# model_params = model_params[model_params.emotion_dimension == 'Arousal']
model_map = {"random forest regress": ensemble.RandomForestRegressor(),
             "xgbooost regress": xgboost.XGBRegressor(),
             # "gradien boost regress": ensemble.GradientBoostingRegressor()
             }
model_official_name = {"xgbooost regress": "XGBoost", "random forest regress": "Random Forest"}
def generate_image(idx, em ,mod):
	row = model_params[(model_params['emotion_dimension'] == em) & (model_params['regressor_type'] == mod)]
	row = row.iloc[0]
	model = model_map[row.regressor_type]
	params = json.loads(row.hyperparams_json)
	model.set_params(**params)
	print(row.emotion_dimension + ' -  ' + row.regressor_type)
	feature_importance = crossval(model, input, row.emotion_dimension, feature_names)
	names = feature_importance.index
	values = feature_importance.importance

	this_axis = ax[idx]
	this_axis.bar(names, values)
	this_axis.set_title(model_official_name[mod])
	# this_axis.set_xticklabels(this_axis.get_xticklabels(), rotation = 45)
	plt.setp(this_axis.get_xticklabels(), rotation=45, horizontalalignment='right')

# em = 'Valence'
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# fig.canvas.manager.set_window_title(em)
# fig.suptitle(em)
# generate_image(0, em=em, mod='random forest regress')
# generate_image(1, em=em, mod='xgbooost regress')
# plt.tight_layout()
# plt.show()
#
# em = 'Arousal'
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# fig.canvas.manager.set_window_title(em)
# fig.suptitle(em)
# generate_image(0, em=em, mod='random forest regress')
# generate_image(1, em=em, mod='xgbooost regress')
# plt.tight_layout()
# plt.show()

em = 'Dominance'
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fig.canvas.manager.set_window_title(em)
fig.suptitle(em)
generate_image(0, em=em, mod='random forest regress')
generate_image(1, em=em, mod='xgbooost regress')
plt.tight_layout()
plt.show()

# for idx, row in model_params.iterrows():
# 	model_type = row.regressor_type
# 	if model_type not in model_map:
# 		continue
# 	model = model_map[row.regressor_type]
# 	params = json.loads(row.hyperparams_json)
# 	model.set_params(**params)
# 	print(row.emotion_dimension + ' -  ' + row.regressor_type)
# 	feature_importance = crossval(model, input, row.emotion_dimension, feature_names)
#
# 	names = feature_importance.index
# 	values = feature_importance.importance
#
# 	fig = plt.figure(figsize=(10, 5))
# 	plt.bar(names, values, color='maroon',width=0.4)
# 	plt.xlabel("Feature name")
# 	plt.ylabel("Feature importance")
# 	plt.title(model_type)
# 	plt.show()


# for emotion in ["Valence", "Arousal", "Dominance"]:
# 	print(emotion)
# 	model = xgboost.XGBRegressor()
# 	params = {'colsample_bytree': 0.7, 'gamma': 7, 'max_depth': 5, 'min_child_weight': 2,
# 	          'reg_alpha': 160, 'reg_lambda': 0}
# 	model.set_params(**params)
# 	crossval(model, input, emotion, feature_names)
