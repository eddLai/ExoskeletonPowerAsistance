import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from EMG_class import EMG_DATA

base_path = 'C:/Users/sean9/Desktop/ExoskeletonPowerAsistance/simulation/mocap_EMG_EEG_data_unsync/data_1009'

folders = [folder for folder in os.listdir(base_path) if folder.startswith('path1_') and os.path.isdir(os.path.join(base_path, folder))]

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    print(f"Processing folder: {folder_path}")

    trc_file_path = glob.glob(os.path.join(folder_path, 'opensim', '*.trc'))[0]
    trc_output_file_path = f'{folder_path}/preprocessing/output/filtered_trc_file.trc'

    try:
        trc_data = pd.read_csv(trc_file_path, sep='\t', skiprows=5, header=None)
    except pd.errors.ParserError:
        print("pandas_error")
        with open(trc_file_path, 'r') as file:
            trc_data = file.read()

    # # 讀取頭部信息的前4行
    with open(trc_file_path, 'r') as file:
        trc_header = [next(file) for _ in range(4)]

    # # 解析第二行和第三行的相機信息
    # camera_info_header = trc_header[1].strip().split('\t')
    # camera_info_values = trc_header[2].strip().split('\t')
    # camera_info = dict(zip(camera_info_header, camera_info_values))

    # # 解析第四行的標頭信息
    header_line = trc_header[3].strip().split('\t')
    cleaned_header = [col for col in header_line if col != '']

    # # 根據標頭信息生成列名，假設每個標記都有X, Y, Z三個軸的資料
    column_names = ['Frame#', 'Time'] + [f'{marker}_{axis}' for marker in cleaned_header[2:] for axis in ['X', 'Y', 'Z']]

    # # 讀取數據部分，跳過前5行（包含文件頭部信息）
    trc_data = pd.read_csv(trc_file_path, sep='\t', skiprows=5, header=None)
    trc_data.columns = column_names[:len(trc_data.columns)]  # 依據解析的標頭信息來設定列名
    # trc_data_cut = trc_data[(trc_data['Frame#'] >= flicker_start_frame_2) & (trc_data['Frame#'] <= flicker_end_frame_2)]
    # print(trc_data_cut)
    emg_data= EMG_DATA(folder_path)
    emg_data.read_emg_file(0)
    first_event_index = emg_data.get_event_id_indices()[0]
    last_event_index = emg_data.get_event_id_indices()[-1]
    emg_data.our_data = emg_data.our_data.iloc[first_event_index:last_event_index]
    emg_data_cut = emg_data.process_data(normalize=True)

    trc_start_time = trc_data['Time'].min()
    trc_end_time = trc_data['Time'].max()
    emg_start_time = emg_data_cut.iloc[0]["Timestamp"]
    emg_end_time = emg_data_cut.iloc[-1]["Timestamp"]
    print(f"TRC Data Time Range: {trc_start_time} to {trc_end_time} = {trc_end_time - trc_start_time}")
    print(f"EMG Data Time Range: {emg_start_time} to {emg_end_time} = {emg_end_time - emg_start_time}")

    emg_data.save_processed_data("cutted_EMG_data")
    trc_data.to_csv(trc_output_file_path, sep='\t', index=False)

    # 提取肌肉名稱（從第二列開始）
    muscle_names = emg_data_cut.columns[2:]
    trc_data.set_index('Time', inplace=True)
    # 準備 TRC 額外數據
    additional_data = {
        'data': trc_data['RKnee_Y'], 
        'label': 'RHeelY',
        'color': 'saddlebrown'
    }

    # 迭代每個肌肉名稱並繪製圖表
    for muscle_name in muscle_names:
        print(f"Processing muscle: {muscle_name}")
        
        emg_data.plot_emg(
            muscle_name=muscle_name, 
            start_flag=emg_start_time, 
            end_flag=emg_end_time, 
            show_raw=False,
            show_processed=False, 
            show_envelope_raw=False, 
            show_normalized=True,
            additional_start_flag=trc_start_time,
            additional_end_flag=trc_end_time,
            additional_data=additional_data,
            save_path=f'{folder_path}/preprocessing/sync_output{muscle_name}'
        )
