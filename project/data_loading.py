import os
import scipy.io
import pandas as pd

def load_mat_file():
    root_folder = 'GazeMatV22'
    test_subs = [f"sub{num}" for num in range(11, 302, 10)]
    data = {}
    
    for sub_dir in os.listdir(root_folder):
        if sub_dir in test_subs:
            mat_file_path = os.path.join(root_folder, sub_dir, 'finalSampled.mat')
            if os.path.exists(mat_file_path):
                mat_data = scipy.io.loadmat(mat_file_path)
                data[sub_dir] = mat_data['itemsSampled']
            else:
                print(f"{mat_file_path} does not exist.")
    print("Data loaded successfully.")
    return data

def process_trials_videos(file_path):
    df = pd.read_csv(file_path)
    parsed_rows = []
    
    for index, row in df.iterrows():
        video_name = row['videoName']
        parsed_data = parse_video_name(video_name)
        parsed_data['firstTargets'] = row['firstTargets']
        parsed_data['secondTargets'] = row['secondTargets']
        parsed_data['fullRotation'] = row['rotAngle']
        parsed_rows.append(parsed_data)
    
    return pd.DataFrame(parsed_rows)

def get_subject_trial_info():
    root_folder = 'GazeMatV21'
    test_subs = ['sub11']

    data = {}

    for sub_dir in os.listdir(root_folder):
        if sub_dir in test_subs:
            csv_file_path = f'./{sub_dir}.csv' 
            
            video_data = pd.DataFrame()
            
            if os.path.exists(csv_file_path):
                video_data = process_trials_videos(csv_file_path)
            else:
                print(f"{csv_file_path} does not exist.")
            
            data[sub_dir] = video_data
    
    print("Data loaded and processed successfully.")
    return data
