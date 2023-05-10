import pandas as pd

df = pd.read_csv("hyper_search_classifier_results_group_mean.csv")
df = pd.read_csv("hyper_search_classifier_feature_reduction_group.csv")
df = pd.read_csv("hyper_search_feature_reduction_group.csv")

maxframe = {'emotion_dimension': [], 'feature_group_criteria': [], 'group_id': [], 'score': []}
for x, i in df.groupby(['emotion_dimension', 'feature_group_criteria', 'group_id']):
	maxframe['score'].append(i.score.max())
	maxframe['feature_group_criteria'].append(i.feature_group_criteria.iloc[0])
	maxframe['group_id'].append(i.group_id.iloc[0])
	maxframe['emotion_dimension'].append(i.emotion_dimension.iloc[0])

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df2 = pd.DataFrame.from_dict(maxframe).reset_index()
print(df2.groupby(['emotion_dimension', 'feature_group_criteria'])['score'].mean())
