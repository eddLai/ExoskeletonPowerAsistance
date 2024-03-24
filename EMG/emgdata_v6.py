import numpy as np
from scipy.signal import butter, lfilter, iirnotch, lfilter_zi
import asyncio
import aiohttp
import websockets
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import keyboard 
import pandas as pd



stop_event = asyncio.Event()  # 控制无限循环的停止

# 数据存储
times = []  # 存储每个数据点的时间（以第一笔数据为0秒）
all_eeg_values = []  # 存储eeg数据值
filtered_emg_values = []  # 存储濾波後emg数据值
emg_level_values=[]

#時間參數
start_time = None

# 初始化全局变量，用于存储滤波器的状态

bp_filter_state = None
notch_filter_state = None
lp_filter_state = None

#肌力參數
initial_max_rms_values = None
initial_min_rms_values = None

async def read_specific_data_from_websocket(uri):
    try:
        async with websockets.connect(uri) as websocket:
            while not stop_event.is_set():  # 检查停止事件是否被设置
                data = await websocket.recv()
                await process_data_from_websocket(data)
    except Exception as e:
         print(f"WebSocket error: {e}")

async def process_data_from_websocket(data):
    global start_time
    eeg_values = []
    try:
        data_dict = json.loads(data)  # 解析JSON数据
        if "contents" in data_dict:
            # 提取 serial_number 和 eeg 的值
            serial_numbers_eegs = [(item['serial_number'][0], item['eeg']) for item in data_dict['contents']]
            # 輸出結果
            for serial_number, eeg in serial_numbers_eegs:
                print(f"Serial Number: {serial_number}, EEG: {eeg}")
                if start_time is None:
                    start_time = serial_number
                time = (serial_number - start_time)*0.001
                times.append(time)
                all_eeg_values.append(eeg)  # 全部的emg資料
                eeg_values.append(eeg[0])      # 最新的50筆emg資料
            eeg_array = np.array(eeg_values) # 将列表转换为NumPy数组
            try:
                await process_emg_signal(eeg_array)
            except Exception as e:
                print(f"处理信号时发生错误: {e}")
    except json.JSONDecodeError:
        print("Failed to decode JSON from WebSocket")
    except Exception as e:
        print(f"Error processing data from WebSocket: {e}")

# 带通滤波器设计
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    global bp_filter_state
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    if bp_filter_state is None:
        bp_filter_state = lfilter_zi(b, a)
    y, bp_filter_state = lfilter(b, a, data, zi=bp_filter_state)
    return y

# 陷波滤波器设计
def notch_filter(data, notch_freq, fs, quality_factor=30):
    global notch_filter_state
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = iirnotch(freq, quality_factor)
    if notch_filter_state is None:
        notch_filter_state = lfilter_zi(b, a)
    y, notch_filter_state = lfilter(b, a, data, zi=notch_filter_state)
    return y

# 全波整流
def full_wave_rectification(data):
    return np.abs(data)

# 低通滤波器设计（提取包络）
def lowpass_filter(data, cutoff, fs, order=4):
    global lp_filter_state
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if lp_filter_state is None:
        lp_filter_state = lfilter_zi(b, a)
    y, lp_filter_state = lfilter(b, a, data, zi=lp_filter_state)
    return y

# 实时信号处理函数
async def process_emg_signal(data, fs=1000):
    
    # 带通滤波
    bandpassed = bandpass_filter(data, 20, 450, fs)
    
    # 50Hz陷波滤波
    notch_filtered = notch_filter(bandpassed, 50, fs)
    
    # 全波整流
    rectified = full_wave_rectification(notch_filtered)
    
    # 低通滤波提取包络
    enveloped = lowpass_filter(rectified, 10, fs)

    filtered_emg_values.extend(enveloped)

    emg_level= calculate_emg_level(enveloped)

    emg_level_values.append(emg_level)

# 以下為肌力回饋
def calculate_emg_level(data):
    global initial_max_rms_values
    global initial_min_rms_values
    #前1秒為暖機
    if len(times) <= 1000:
        return 0
    # 使用第1秒到第10秒的数据来确定初始的最小、最大RMS值
    elif 1000 < len(times) <= 10000:
        rms_values = calculate_rms(data)
        if initial_max_rms_values is None or rms_values > initial_max_rms_values:
            initial_max_rms_values = rms_values
        elif initial_min_rms_values is None or rms_values < initial_min_rms_values:
            initial_min_rms_values = rms_values
        return 0
    #每0.05秒傳出reward值
    else:
        rms_values = calculate_rms(data)
        y = map_to_levels(rms_values, initial_min_rms_values, initial_max_rms_values)
        print(rms_values, initial_min_rms_values, initial_max_rms_values)
        return y

def calculate_rms(signal):
    """计算信号的RMS值。"""
    return np.sqrt(np.mean(signal**2))

def map_to_levels(value, min_rms_values, max_rms_values):
    """将值映射到超出5到-5级的线性值上，基于放松阈值和初始最大RMS值，
    但在上下限内分为5到-5十个等级区间。"""
    # 计算每个等级的值范围大小
    level_range = (max_rms_values - min_rms_values) / 10
    
    if value <= min_rms_values:
        # 计算低于min_rms_values的值应映射到哪个级别
        level_diff = (min_rms_values-value) / level_range
        return 5 + level_diff
    elif value >= max_rms_values:
        # 计算高于max_rms_values的值应映射到哪个级别
        level_diff = (value - max_rms_values) / level_range
        return -5 + level_diff
    else:
        # 线性映射到5到-5
        normalized_value = (value - min_rms_values) / (max_rms_values - min_rms_values)
        return int(round(normalized_value * (-10))) + 5

def plot_data():
    # 假设all_eeg_values现在包含的是包含两个元素的列表，即每个时间点有两个eeg值
    # 分别提取两组EEG数据
    eeg_values_0 = [eeg[0] for eeg in all_eeg_values]  # 提取每个时间点的第一个eeg值
    eeg_values_1 = [eeg[1] for eeg in all_eeg_values]  # 提取每个时间点的第二个eeg值
    new_times = times[::50]  # 从times中每50个元素取一个
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # 创建两个子图
    
    # 绘制第一个EEG值
    axs[0].plot(times, eeg_values_0, 'r-')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Value')
    axs[0].set_title('Raw EMG')
    
    # 绘制第二个EEG值
    axs[1].plot(times, filtered_emg_values, 'b-')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Value')
    axs[1].set_title('filtered EMG')

    # 绘制emg level
    axs[2].plot(new_times, emg_level_values, 'g-')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Value')
    axs[2].set_title('EMG Level')

    print(len(new_times),len(emg_level_values))
    
    plt.tight_layout()  # 调整子图布局，避免标题和轴标签重叠
    plt.show()


def on_press_key(e):
    if e.name == 'esc':  # 如果按下的是'p'键p
        stop_event.set()  # 设置停止事件
        print("Stop event set, exiting...")

async def main():
    websocket_uri = "ws://localhost:31278/ws"
    
    # 在新的线程中监听键盘事件
    keyboard.on_press(on_press_key)

    await read_specific_data_from_websocket(websocket_uri)

    # 在程序结束时绘制所有收集到的数据
    plot_data()

if __name__ == "__main__":
    asyncio.run(main())

