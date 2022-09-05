
import pandas as pd
import matplotlib.pyplot as plt

import heartpy as hp
from hrvanalysis import *
from core.main import *

# problems with hr signal on
# participant 22 - video 0, 10, 12, 18, 22, 26, 31
# participant 25 - video 5
# participant 26 - video 9, 22

path = "hr_features.pkl"
print("HR pickle not found starting generation: ") if os.path.isfile(path) else None
if os.path.isfile(path):
    data_with_features_frame = pd.read_pickle(path)
    print(data_with_features_frame)
else:
    hr_channel = 38
    sample_rate = 128
    data = extract_data_by_video()
    video_info = pd.read_csv("metadata_csv/video_list.csv")
    feature_names = []
    results_frame = {}
    hr_data = []
    for video_id in range(40):
        # process HEART RATE data
        section = data.loc[(data['video'] == video_id) & (data['channel_id'] == hr_channel)]
        X_input = section['channel_data'].tolist()
        # heart_rate_input = [list(get_time_domain_features(x).values()) for x in X_input]

        for participant, x in enumerate(X_input):
            if participant in exclude_participant:
                continue

            filtered = hp.filter_signal(x, [0.81, 3.5], sample_rate=sample_rate,
                                        order=3, filtertype='bandpass')
            wd, m = hp.process(filtered, sample_rate=sample_rate)

            filtered_input = list(m.values())
            part_data = np.array([participant, video_id] + filtered_input)
            hr_data.append(part_data)

        if not feature_names:
            _, feats = hp.process(X_input[0], sample_rate=sample_rate)
            feature_names = list(feats.keys())
        #     results_frame = {x: [] for x in ['video_name', 'emotion', 'test_score'] + feature_names}


        # row = video_info[video_info['Experiment_id'] == (video_id + 1)].iloc[0]
        # video_name = row['Artist'] + " - " + row['Title']
    # results_frame = pd.DataFrame(results_frame)
    # print(results_frame)

    data_with_features_frame = pd.DataFrame(hr_data, columns=['participant', 'video'] + feature_names)
    # write to pickle
    data_with_features_frame.to_pickle(path)
