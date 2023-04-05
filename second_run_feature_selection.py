import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import svm
from sklearn import preprocessing
from core.main import reduce_features_worker, get_ratings_group, get_ratings_second
import xgboost
import json, os, sys
from csv import writer

hyperparams_grouped_filename = "hyper_search_results_group.csv"
hyperparams_all_filename = "hyper_search_results.csv"

def write_to_csv(filename, to_add):
    csv_header = ['regressor_type', 'emotion_dimension', 'feature_group_criteria', 'group_id', 'best_feature_names', 'score']
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

if __name__ == "__main__":
	csv_filename_group = "hyper_search_feature_reduction_group.csv"
	csv_filename_overall = "hyper_search_feature_reduction.csv"

	ratings = pd.read_csv("metadata_csv/participant_ratings.csv")
	# valence_statistics = ratings.groupby("participant")["Valence"].agg(
	# 	[np.min, np.max, np.mean, np.std, np.median, lambda x: ((x > x.median())*1).sum() / 40, lambda x: ((x > x.mean())*1).sum() / 40])

	hyperparams_groups = pd.read_csv("hyper_search_results_group.csv")
	hyperparams_overall = pd.read_csv("hyper_search_results.csv")
	# print(hyperparams_groups.groupby(['emotion_dimension'])['score'].agg([np.mean]))
	# feature_data = pd.read_csv(csv_filename)
	# print(feature_data.groupby(['emotion_dimension'])['score'].agg([np.mean]))
	# import sys
	# sys.exit(0)

	### OVERALL ###
	for emotion, group_data in hyperparams_overall.groupby(['emotion_dimension']):
		print(emotion)

		hr_features_frame = pd.read_pickle("hr_features.pkl")
		eda_features_frame = pd.read_pickle("eda_features.pkl")
		eda_features_frame = eda_features_frame.drop(columns=['participant', 'video'])
		full_feature_data = pd.concat([hr_features_frame, eda_features_frame], axis=1)
		full_feature_data = full_feature_data.fillna(0)
		clean_frame = full_feature_data.sort_values(['participant', 'video'])

		feats = clean_frame
		feats = feats.drop(columns=['participant', 'video'])
		scaled_features = preprocessing.StandardScaler().fit_transform(feats.values)
		scaled_features = pd.DataFrame(scaled_features, index=feats.index, columns=feats.columns)
		print(scaled_features)
		feature_names = scaled_features.columns

		input_data = scaled_features.values.tolist()
		input_data = np.array(input_data)

		best_score_row = group_data[group_data.score == group_data.score.max()]
		model_id = best_score_row.regressor_type.iloc[0]
		# print(model_id)
		if model_id == "random_forest":
			model = ensemble.RandomForestRegressor()
		if model_id == "gradientboost_regressor":
			model = ensemble.GradientBoostingRegressor()
		if model_id == "xgboost_regressor":
			model = xgboost.XGBRegressor()
		if model_id == "svr_regressor":
			model = svm.SVR()

		hyperperams = best_score_row.hyperparams_json.iloc[0]
		hyperperams = json.loads(hyperperams)

		target = get_ratings_second(emotion)
		output = np.array(target)

		best_score, features = reduce_features_worker(model, np.array(input_data), np.array(output), is_linear=True)
		best_feature_names = [feature_names[i] for i in features]

		group_name = "none"
		group_id = "none"
		csv_row = [model_id, emotion, group_name, group_id, json.dumps(best_feature_names), best_score]
		write_to_csv(csv_filename_overall, csv_row)

	sys.exit(0)
	for emotion, emotion_data in hyperparams_groups.groupby(['emotion_dimension']):
		print(emotion)
		facet_group_map = {'participant': 'video', 'video': 'participant'}

		## GROUPBY ##
		for group_name in ['video', 'participant']:
			criteria_trim_data = emotion_data.loc[emotion_data.feature_group_criteria == group_name]
			for group_id, group_data in criteria_trim_data.groupby(["group_id"]):
				print(group_id)
				hr_features_frame = pd.read_pickle("hr_features.pkl")
				eda_features_frame = pd.read_pickle("eda_features.pkl")
				eda_features_frame = eda_features_frame.drop(columns=['participant', 'video'])
				full_feature_data = pd.concat([hr_features_frame, eda_features_frame], axis=1)
				full_feature_data = full_feature_data.fillna(0)
				clean_frame = full_feature_data.sort_values(['participant', 'video'])

				if group_name == 'participant':
					feats = clean_frame[clean_frame.participant == int(group_id)]
				else:# group_name == 'video':
					feats = clean_frame[clean_frame.video == int(group_id)]
				feats = feats.drop(columns=['participant', 'video'])
				scaled_features = preprocessing.StandardScaler().fit_transform(feats.values)
				scaled_features = pd.DataFrame(scaled_features, index=feats.index, columns=feats.columns)
				feature_names = scaled_features.columns

				input_data = scaled_features.values.tolist()
				input_data = np.array(input_data)

				best_score_row = group_data[group_data.score == group_data.score.max()]
				model_id = best_score_row.regressor_type.iloc[0]
				# print(model_id)
				if model_id == "random_forest":
					model = ensemble.RandomForestRegressor()
				if model_id == "gradientboost_regressor":
					model = ensemble.GradientBoostingRegressor()
				if model_id == "xgboost_regressor":
					model = xgboost.XGBRegressor()
				if model_id == "svr_regressor":
					model = svm.SVR()

				hyperperams = best_score_row.hyperparams_json.iloc[0]
				hyperperams = json.loads(hyperperams)

				exclude = True if group_name == 'video' else False
				target = get_ratings_group(
					emotion, group_name, int(group_id), facet_group_map[group_name], exclude, binary=False)
				output = np.array(target)

				# print(input_data.shape)
				# print(output.shape)

				best_score, features = reduce_features_worker(model, np.array(input_data), np.array(output), is_linear=True)
				best_feature_names = [feature_names[i] for i in features]
				csv_row = [model_id, emotion, group_name, group_id, json.dumps(best_feature_names), best_score]
				write_to_csv(csv_filename_group, csv_row)
