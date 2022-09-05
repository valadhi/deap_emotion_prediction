import sys
import pandas as pd
import matplotlib.pyplot as plt

import heartpy as hp
from hrvanalysis import *
import ledapy

from core.main import *

path = "eda_features.pkl"
print(os.path.isfile(path))
if os.path.isfile(path):
    data_with_features_frame = pd.read_pickle(path)
    print(data_with_features_frame)
else:
    eda_channel = 36
    sample_rate = 128
    data = extract_data_by_video()
    video_info = pd.read_csv("metadata_csv/video_list.csv")

    # EDA convex optimization novel feature extraction method
    feature_names = ["conductance_error","conductance_min","conductance_max"]
    peak_feature_names = ["onset","peaktime"]
    results_frame = {c: [] for c in ['video_name', 'emotion', 'test_score'] + feature_names + peak_feature_names + ["mean_gsr"]}

    eda_data = []
    for video_id in range(40):

        # process EDA data
        section = data.loc[(data['video'] == video_id) & (data['channel_id'] == eda_channel)]
        X_input = np.array(section['channel_data'].tolist())

        for participant, x in enumerate(X_input):
            m, wd, eda_clean = process_statistical(x, use_scipy=False, sample_rate=128,
                                               new_sample_rate=20, segment_width=10,segment_overlap=0)
            # eda_data.append(m['mean_gsr'])
            try:
                phasicdata = ledapy.runner.getResult(x, 'phasicdata', sample_rate, downsample=4, optimisation=2)
            except Exception as e:
                print("participant "+str(participant)+" for video "+str(video_id)+" returns error")
                eda_data.append(np.array([0 for f in range(len(feature_names) + len(peak_feature_names) + 3)]))
                continue
            leda_data = ledapy.leda2.data
            leda_data2 = ledapy.leda2.trough2peakAnalysis
            trough_peak_data = [np.mean(getattr(leda_data2, tfeat)) for tfeat in peak_feature_names]
            core_data = [getattr(leda_data, feat_name) for feat_name in feature_names]

            new_row = np.array([participant, video_id] + core_data + trough_peak_data + [np.mean(m['mean_gsr'])])
            eda_data.append(new_row)

        sys.exit(0)
        row = video_info[video_info['Experiment_id'] == (video_id+1)].iloc[0]
        video_name = row['Artist'] + " - " + row['Title']

    data_with_features_frame = pd.DataFrame(eda_data, columns=['participant', 'video']+
                                                              feature_names + peak_feature_names + ["mean_gsr"])
    data_with_features_frame.to_pickle(path)