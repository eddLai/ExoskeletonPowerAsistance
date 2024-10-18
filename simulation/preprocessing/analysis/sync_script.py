import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

base_path = 'C:/Users/sean9/Desktop/ExoskeletonPowerAsistance/simulation/mocap_EMG_EEG_data_unsync/data_1009'

# 列出 base_path 中所有 path1_ 開頭的資料夾
folders = [folder for folder in os.listdir(base_path) if folder.startswith('path1_') and os.path.isdir(os.path.join(base_path, folder))]

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    print(f"Processing folder: {folder_path}")
    # 檢查視頻文件路徑
    video_dir = os.path.join(folder_path, 'videos')
    print(f"Looking for video files in: {video_dir}")
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    if len(video_files) == 0:
        raise FileNotFoundError(f"No video files found in {video_dir}")

    trc_file_path = glob.glob(os.path.join(folder_path, 'opensim', '*.trc'))[0]
    trc_output_file_path = f'{folder_path}/preprocessing/output/filtered_trc_file.trc'

    video_path = video_files[0]
    print(f"Found video file: {video_path}")

    # 打開視頻檔案並獲取總幀數
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: ", total_frames)

    # 初始化參數
    brightness_values = []
    red_values = []
    green_values = []
    frame_number = 0
    max_offset = 30
    enable_movement = False

    # 變數
    lamp_x, lamp_y = None, None
    half_size = 15

    # 滑鼠事件處理函數，選擇燈的位置
    def mouse_callback(event, x, y, flags, param):
        global lamp_x, lamp_y, frame_copy
        if event == cv2.EVENT_MOUSEMOVE:  # 當滑鼠移動時，更新動態矩形框
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (x - half_size, y - half_size), (x + half_size, y + half_size), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONDOWN:  # 當滑鼠左鍵按下時
            lamp_x, lamp_y = x, y
            print(f'Selected point: ({lamp_x}, {lamp_y})')

    # 選擇幀進行手動燈標記
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    frame_copy = frame.copy()  # 用於動態顯示滑鼠移動的矩形框

    cv2.namedWindow('Select Point', cv2.WINDOW_NORMAL)  # 設置窗口可調整大小
    cv2.resizeWindow('Select Point', 640*2, 480*2)  # 將窗口大小設置為 1280x960

    cv2.setMouseCallback('Select Point', mouse_callback)

    # 顯示圖像並等待滑鼠選擇
    while True:
        cv2.imshow('Select Point', frame_copy)
        if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC鍵退出
            break
        if lamp_x is not None and lamp_y is not None:  # 如果有選擇點，則退出循環
            break

    cv2.destroyAllWindows()

    # 若未選擇點，則退出程式
    if lamp_x is None or lamp_y is None:
        print("No point selected. Exiting.")
        cap.release()
        exit()

    # 開始處理視頻，追蹤燈的位置和計算亮度、顏色
    initial_lamp_x = lamp_x
    initial_lamp_y = lamp_y
    THRESHOLD = np.mean(cv2.cvtColor(frame[max(0, lamp_y - half_size):min(frame.shape[0], lamp_y + half_size),
                                            max(0, lamp_x - half_size):min(frame.shape[1], lamp_x + half_size)], 
                                    cv2.COLOR_BGR2GRAY))

    # 重置視頻捕捉到開始位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 設置窗口來顯示處理過程中的幀
    cv2.namedWindow("Original Frame with Cropped Area", cv2.WINDOW_NORMAL)  # 設置窗口大小可以調整
    cv2.resizeWindow("Original Frame with Cropped Area", int(1280*3/4), int(960*3/4))  # 調整窗口大小為1280x960

    # 循環處理每一幀
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        y_start = max(0, lamp_y - half_size)
        y_end = min(frame.shape[0], lamp_y + half_size)
        x_start = max(0, lamp_x - half_size)
        x_end = min(frame.shape[1], lamp_x + half_size)
        brightness_frame = gray_frame[y_start:y_end, x_start:x_end]

        # 確定燈位置和亮度
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(brightness_frame)
        if max_val > THRESHOLD and enable_movement:
            new_lamp_x = x_start + max_loc[0]
            new_lamp_y = y_start + max_loc[1]
            if abs(new_lamp_x - initial_lamp_x) > max_offset or abs(new_lamp_y - initial_lamp_y) > max_offset:
                print(f"Error: Lamp moved too far at frame {frame_number}")
                break
            else:
                lamp_x = new_lamp_x
                lamp_y = new_lamp_y
            
        avg_brightness = np.mean(brightness_frame)
        brightness_values.append(avg_brightness)

        red_channel = frame[y_start:y_end, x_start:x_end, 2]
        green_channel = frame[y_start:y_end, x_start:x_end, 1]
        avg_red = np.mean(red_channel)
        avg_green = np.mean(green_channel)
        red_values.append(avg_red)
        green_values.append(avg_green)

        # 在原始影像上繪製燈的裁剪框
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        
        # 顯示處理過程中的原始圖像和裁剪的區域
        cv2.imshow("Original Frame with Cropped Area", frame)

        # 檢查是否按下 'q' 鍵以退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 每隔 100 幀打印進度
        if frame_number % 100 == 0:
            print(f"Processed frame {frame_number}/{total_frames}")

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

    import numpy as np
    import matplotlib.pyplot as plt

    # 設定參數
    time_interval = 1 / 30  # 假設每幀時間間隔為 1/30 秒 (30 fps)
    threshold_multiplier = 1.3  # 閾值設定為平均亮度變化的倍數

    def smooth_data(data, window_size=10):
        """
        使用移動平均對數據進行平滑處理
        :param data: 原始數據
        :param window_size: 移動平均的窗口大小
        :return: 平滑後的數據
        """
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def detect_flicker_start_and_end(brightness_values, time_interval, threshold_multiplier):
        """
        基於整體亮度的平均值自動計算閾值，檢測第一個和最後一個亮度變化超過閾值的幀號
        :param brightness_values: 紅燈亮度的序列
        :param time_interval: 每一幀的時間間隔
        :param threshold_multiplier: 閾值倍率，用來乘以亮度變化的標準差
        :return: 方波開始和結束的時間點, 對應的幀號
        """
        # 計算亮度信號的差分來檢測變化
        brightness_diff = np.diff(brightness_values)

        # 計算亮度差分的平均值和標準差，動態設定閾值
        mean_brightness_diff = np.mean(np.abs(brightness_diff))
        std_brightness_diff = np.std(np.abs(brightness_diff))
        threshold = mean_brightness_diff + threshold_multiplier * std_brightness_diff
        print(f"自動計算的閾值為: {threshold:.2f}")

        # 找到第一個亮度變化大於設定閾值的點（開始）
        start_index = np.argmax(np.abs(brightness_diff) > threshold)

        # 找到最後一個亮度變化大於設定閾值的點（結束）
        end_index = len(brightness_diff) - np.argmax(np.abs(brightness_diff[::-1]) > threshold) - 1

        if start_index >= 0 and end_index >= 0 and start_index < end_index:
            # 計算開始和結束的幀號和時間
            flicker_start_frame = start_index
            flicker_end_frame = end_index

            flicker_start_time = flicker_start_frame * time_interval
            flicker_end_time = flicker_end_frame * time_interval

            print(f"紅燈開始的時間是: {flicker_start_time:.2f} 秒, 幀號: {flicker_start_frame}")
            print(f"紅燈結束的時間是: {flicker_end_time:.2f} 秒, 幀號: {flicker_end_frame}")

            return flicker_start_time, flicker_start_frame, flicker_end_time, flicker_end_frame
        else:
            print("未檢測到紅燈的開始或結束")
            return None, None, None, None
        

    def remove_extreme_from_smoothed(smoothed_brightness_values, start_frame, end_frame):
        """
        從平滑後的亮度數據中去除極端值區間
        :param smoothed_brightness_values: 平滑後的亮度數據
        :param start_frame: 開始幀號
        :param end_frame: 結束幀號
        :return: 去除極端值後的亮度數據
        """
        # 將極端區間內的亮度數據設置為均值，或者將其刪除
        smoothed_brightness_values[start_frame:end_frame] = np.mean(smoothed_brightness_values)
        return smoothed_brightness_values

    # 使用亮度值偵測紅燈開始和結束
    smoothed_brightness_values = smooth_data(brightness_values)
    flicker_start_time, flicker_start_frame, flicker_end_time, flicker_end_frame = detect_flicker_start_and_end(smoothed_brightness_values, time_interval, threshold_multiplier)
    smoothed_brightness_values_no_extreme = remove_extreme_from_smoothed(smoothed_brightness_values.copy(), flicker_start_frame, flicker_end_frame)
    flicker_start_time_2, flicker_start_frame_2, flicker_end_time_2, flicker_end_frame_2 = detect_flicker_start_and_end(smoothed_brightness_values_no_extreme, time_interval, threshold_multiplier)

    # 畫圖顯示結果
    time_values = np.arange(0, len(brightness_values)) * time_interval
    plt.figure(figsize=(10, 6))
    plt.plot(time_values[:len(smoothed_brightness_values)], smoothed_brightness_values, label='Original Smoothed Brightness Values', linestyle='--')
    plt.plot(time_values[:len(smoothed_brightness_values_no_extreme)], smoothed_brightness_values_no_extreme, label='Smoothed Brightness Values Without Extreme', linestyle='--')

    if flicker_start_time_2 is not None and flicker_end_time_2 is not None:
        plt.axvline(x=flicker_start_time_2, color='r', linestyle='--', label=f'Second Start at {flicker_start_time_2:.2f}s (Frame {flicker_start_frame_2})')
        plt.axvline(x=flicker_end_time_2, color='b', linestyle='--', label=f'Second End at {flicker_end_time_2:.2f}s (Frame {flicker_end_frame_2})')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Brightness')
    plt.title('Smoothed Brightness Values with Flicker Start and End Detection (Extreme Removed)')
    plt.legend()
    plt.show()

    try:
        trc_data = pd.read_csv(trc_file_path, sep='\t', skiprows=5, header=None)
    except pd.errors.ParserError:
        print("pandas_error")
        with open(trc_file_path, 'r') as file:
            trc_data = file.read()

    # 讀取頭部信息的前4行
    with open(trc_file_path, 'r') as file:
        trc_header = [next(file) for _ in range(4)]

    # 解析第二行和第三行的相機信息
    camera_info_header = trc_header[1].strip().split('\t')
    camera_info_values = trc_header[2].strip().split('\t')
    camera_info = dict(zip(camera_info_header, camera_info_values))

    # 解析第四行的標頭信息
    header_line = trc_header[3].strip().split('\t')
    cleaned_header = [col for col in header_line if col != '']

    # 根據標頭信息生成列名，假設每個標記都有X, Y, Z三個軸的資料
    column_names = ['Frame#', 'Time'] + [f'{marker}_{axis}' for marker in cleaned_header[2:] for axis in ['X', 'Y', 'Z']]

    # 讀取數據部分，跳過前5行（包含文件頭部信息）
    trc_data = pd.read_csv(trc_file_path, sep='\t', skiprows=5, header=None)
    trc_data.columns = column_names[:len(trc_data.columns)]  # 依據解析的標頭信息來設定列名
    trc_data_cut = trc_data[(trc_data['Frame#'] >= flicker_start_frame_2) & (trc_data['Frame#'] <= flicker_end_frame_2)]
    print(trc_data_cut)

    from EMG_class import EMG_DATA
    emg_data= EMG_DATA(folder_path)
    emg_data.read_emg_file(0)
    first_event_index = emg_data.get_event_id_indices()[0]
    last_event_index = emg_data.get_event_id_indices()[-1]
    emg_data.our_data = emg_data.our_data.iloc[first_event_index:last_event_index]
    emg_data_cut = emg_data.process_data(normalize=True)

    trc_start_time = trc_data_cut['Time'].min()
    trc_end_time = trc_data_cut['Time'].max()
    emg_start_time = emg_data_cut.iloc[0]["Timestamp"]
    emg_end_time = emg_data_cut.iloc[-1]["Timestamp"]
    print(f"TRC Data Time Range: {trc_start_time} to {trc_end_time} = {trc_end_time - trc_start_time}")
    print(f"EMG Data Time Range: {emg_start_time} to {emg_end_time} = {emg_end_time - emg_start_time}")

    emg_data.save_processed_data("cutted_EMG_data")
    trc_data_cut.to_csv(trc_output_file_path, sep='\t', index=False)

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
    cv2.destroyAllWindows()  # 這個在每次處理完視頻後都應該被調用
