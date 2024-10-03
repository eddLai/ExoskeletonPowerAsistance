import cv2
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import iirnotch, butter, filtfilt

class BandFilter:
    def __init__(self, order=4, fc_bp=[20, 480], freq=1000):
        nyq = 0.5 * freq
        low = fc_bp[0] / nyq
        high = fc_bp[1] / nyq
        self.b, self.a = butter(order, (low, high), btype = 'bandpass', output='ba')
    
    def filtfilt(self, x):
        filtered_x = filtfilt(self.b, self.a, x)
        return filtered_x
    
class LowPassFilter:
    def __init__(self, order=2, fc_bp=4, freq=1000):
        nyq = 0.5 * freq
        low = fc_bp / nyq
        self.b, self.a = butter(order, low, btype = 'lowpass', output='ba')
    
    def filtfilt(self, x):
        filtered_x = filtfilt(self.b, self.a, x)
        return filtered_x

class HighPassFilter:
    def __init__(self, order=2, fc_bp=30, freq=1000):
        nyq = 0.5 * freq
        high = fc_bp / nyq
        self.b, self.a = butter(order, high, btype = 'highpass', output='ba')
    
    def filtfilt(self, x):
        filtered_x = filtfilt(self.b, self.a, x)
        return filtered_x

class NotchFilter:
    def __init__(self, f0=60, freq=1000):
        f0 = f0  # Frequency to be removed from signal (Hz)
        Q = 30.0  # Quality factor
        self.b, self.a = iirnotch(f0, Q, freq)
    
    def filtfilt(self, x):
        filtered_x = filtfilt(self.b, self.a, x)
        return filtered_x

def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return data
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

sampling_rate = 1000
h_filter = HighPassFilter(order=4, fc_bp=30)
l_filter = LowPassFilter(order=4, fc_bp=4)
n_filter = NotchFilter(f0=60)

def EMGprocessor(signal):
    filted_emg = h_filter.filtfilt(signal)
    filted_emg = n_filter.filtfilt(filted_emg)
    rect_emg = np.abs(filted_emg)
    envelope = l_filter.filtfilt(rect_emg)
    envelope = normalize_data(envelope)
    return envelope

video_path = 'data/1 (1).mp4'
trc_file_path = 'data/Empty_project_filt_0-30 (5).trc'
emg_file_path = 'data/emg_path1_04_162956-S.csv'
trc_output_file_path = 'output/filtered_trc_file.trc'
emg_output_file_path = 'output/post_event_data.csv'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("無法打開影片")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
lamp_x, lamp_y = 745, 469
frame_number = 0
brightness_values = []
frame_indexes = []
brightness_threshold = 150

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray_frame[lamp_y, lamp_x]
    if brightness > brightness_threshold:
        frame_indexes.append(frame_number)
    frame_number += 1

cap.release()
cv2.destroyAllWindows()

if len(frame_indexes) > 1:
    intervals = np.diff(frame_indexes) / fps
    frequencies = 1 / intervals

    for i, freq in enumerate(frequencies):
        if np.isclose(freq, 16, atol=1):
            first_frame_16hz = frame_indexes[i]
            print(f"開始以 16 Hz 閃爍的幀號: {first_frame_16hz}")
            break
    else:
        print("未檢測到 16 Hz 的閃爍頻率")
else:
    print("沒有足夠的亮度超過門檻的幀來檢測頻率")

column_headers = ['Frame#', 'Time'] + [
    'Hip_X', 'Hip_Y', 'Hip_Z', 
    'RHip_X', 'RHip_Y', 'RHip_Z', 
    'RKnee_X', 'RKnee_Y', 'RKnee_Z',
    'RAnkle_X', 'RAnkle_Y', 'RAnkle_Z',
    'RBigToe_X', 'RBigToe_Y', 'RBigToe_Z',
    'RSmallToe_X', 'RSmallToe_Y', 'RSmallToe_Z',
    'RHeel_X', 'RHeel_Y', 'RHeel_Z',
    'LHip_X', 'LHip_Y', 'LHip_Z',
    'LKnee_X', 'LKnee_Y', 'LKnee_Z',
    'LAnkle_X', 'LAnkle_Y', 'LAnkle_Z',
    'LBigToe_X', 'LBigToe_Y', 'LBigToe_Z',
    'LSmallToe_X', 'LSmallToe_Y', 'LSmallToe_Z',
    'LHeel_X', 'LHeel_Y', 'LHeel_Z',
    'Neck_X', 'Neck_Y', 'Neck_Z',
    'Head_X', 'Head_Y', 'Head_Z',
    'Nose_X', 'Nose_Y', 'Nose_Z',
    'RShoulder_X', 'RShoulder_Y', 'RShoulder_Z',
    'RElbow_X', 'RElbow_Y', 'RElbow_Z',
    'RWrist_X', 'RWrist_Y', 'RWrist_Z',
    'LShoulder_X', 'LShoulder_Y', 'LShoulder_Z',
    'LElbow_X', 'LElbow_Y', 'LElbow_Z',
    'LWrist_X', 'LWrist_Y', 'LWrist_Z'
]

data = pd.read_csv(trc_file_path, delim_whitespace=True, skiprows=6, names=column_headers)
extracted_data = data.iloc[first_frame_16hz:].copy()
extracted_data['Time'] = extracted_data['Time'] - extracted_data['Time'].iloc[0]
extracted_data['Frame#'] = range(1, len(extracted_data) + 1)

with open(trc_file_path, 'r') as file:
    header_lines = [file.readline() for _ in range(6)]

with open(trc_output_file_path, 'w') as file:
    for line in header_lines:
        file.write(line)
    extracted_data.to_csv(file, sep='\t', index=False, header=False, line_terminator='\n')
print(f"提取的資料已寫入 {trc_output_file_path}，時間和 Frame# 已重設。")

df = pd.read_csv(emg_file_path, skiprows=10)
event_col = 'Event Id'
first_event_index = df[df[event_col].notna()].index[4]
post_event_data = df.iloc[first_event_index:]
post_event_data.to_csv(emg_output_file_path, index=False)
print(f"處理結果已保存到: {emg_output_file_path}")
################################################################
emg_output = pd.read_csv('output/post_event_data.csv')
emg_output = emg_output.iloc[1:, 2:10]
emg_output = emg_output.transpose()
emg_output = emg_output.to_numpy()
emg_data = np.zeros((8,emg_output.shape[1]) ,dtype=float)

for i in range(8):
    emg_data[i] = EMGprocessor(emg_output[i, ...])
    
emg_data[emg_data < 0] = 0
plt.rcParams["figure.figsize"] = (30,8)
t1 = np.arange(emg_data.shape[1]) / sampling_rate
file_path = 'output/filtered_trc_file.trc'
trc_data = pd.read_csv(file_path, delim_whitespace=True, skiprows=6, names=column_headers)
plt.plot(t1, emg_data[1])
t2 = np.arange(trc_data['RHeel_Y'].shape[0]) / 30
plt.plot(t2, trc_data['RHeel_Y'])
plt.show()