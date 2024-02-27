import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd

# 假設我們有一個名為 raw_emg 的原始 EMG 信號數據
# 指定文件路径
file_path = r"C:\Users\pp\Desktop\EMG_rawdata\EMG_path1_1.csv"
# 使用 pandas 读取 CSV 文件
df = pd.read_csv(file_path)

column_values = df.iloc[1:, 9]

# 将这些值转换为列表
raw_emg = column_values.tolist()

# 定義一個帶通濾波器
def bandpass_filter(signal, low_freq, high_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# 設定你的採樣率
sample_rate = 1000  # Hz

# 設定你的濾波器頻率範圍，這將取決於你的應用
low_freq = 20  # Hz
high_freq = 450  # Hz

# 應用帶通濾波器
filtered_emg = bandpass_filter(raw_emg, low_freq, high_freq, sample_rate)

# 定義一個帶拒濾波器，消除電源線干擾
def bandstop_filter(signal, low_cut, high_cut, fs, order=4):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = filtfilt(b, a, signal)
    return y

# 带拒滤波器参数
bs_low_freq = 55  # Hz
bs_high_freq = 65  # Hz

# 应用带拒滤波器
notch_filtered_emg = bandstop_filter(filtered_emg, bs_low_freq, bs_high_freq, sample_rate)

# 可以將信號整流，取絕對值
rectified_emg = np.abs(notch_filtered_emg)

# 定義一個平滑濾波器(低通)，繪製
def lowpass_filter(signal, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, signal)
    return y

# 低通滤波器参数
lp_cutoff = 5  # Hz

# 应用低通滤波器
smoothed_emg = lowpass_filter(rectified_emg, lp_cutoff, sample_rate)

# 以下為肌力回饋
def calculate_rms(signal, window_size):
    """计算信号的RMS值。"""
    return np.sqrt(np.mean(signal**2))

def map_to_levels(value, relaxation_threshold, initial_max_rms):
    """将值映射到0到10级，基于放松阈值和初始最大RMS值。"""
    if value <= relaxation_threshold:
        return 0
    elif value >= initial_max_rms:
        return 10
    else:
        # 线性映射到0到10
        normalized_value = (value - relaxation_threshold) / (initial_max_rms - relaxation_threshold)
        return int(round(normalized_value * 10))  # 取整操作


# 假设这是你的预处理后的EMG信号
signal = smoothed_emg

window_duration = 0.05  # 窗口持续时间0.01秒
window_size = int(sample_rate * window_duration)  # 窗口大小

# 使用前10000个点的数据来确定放松状态的基准阈值
relaxation_segment = signal[:10000]
relaxation_rms_values = [calculate_rms(relaxation_segment[i:i+window_size], window_size) for i in range(0, len(relaxation_segment), window_size)]
relaxation_threshold = np.mean(relaxation_rms_values)

# 使用第10000到第20000个点的数据来确定初始的最大RMS值
initial_max_segment = signal[10000:20000]
initial_max_rms_values = [calculate_rms(initial_max_segment[i:i+window_size], window_size) for i in range(0, len(initial_max_segment), window_size)]
initial_max_rms = max(initial_max_rms_values)

# 存储每个窗口的等级以便绘图
levels = []

# 实时处理后续信号并映射到0到10级
for i in range(20000, len(signal), window_size):
    window = signal[i:i+window_size]
    rms = calculate_rms(window, window_size)
    level = map_to_levels(rms, relaxation_threshold, initial_max_rms)
    levels.append(level)
    # 可能需要更新初始最大RMS值以适应更高的活动
    if rms > initial_max_rms:
        initial_max_rms = rms
    # print(f"RMS: {rms}, Level: {level}")

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(levels, linestyle='-', color='b')
plt.title('Muscle Activity Levels Over Time')
plt.xlabel('Time (0.05s per point)')
plt.ylabel('Activity Level (0-10)')
plt.grid(True)
plt.xticks(range(0, len(levels), len(levels)//10), rotation=45)
plt.yticks(range(0, 11))
plt.tight_layout()
plt.show()