import numpy as np
from scipy.signal import butter, lfilter, iirnotch, lfilter_zi
import asyncio
import aiohttp
import websocket
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import keyboard 
import pandas as pd

def read_specific_data_from_websocket(uri, bp_parameter, nt_parameter, lp_parameter,initial_max_min_rms_values,init_time):
        try:
            ws = websocket.WebSocket()
            ws.connect(uri)
            print("Press 'q' to quit...")
            while True:
                # 检查是否按下了 'q' 键
                if keyboard.is_pressed('q'):
                    print("Quitting...")
                    break  # 跳出循环
                data = ws.recv()
                emg_array, bp_parameter, nt_parameter, lp_parameter = process_data_from_websocket(data, bp_parameter, nt_parameter, lp_parameter)
                if emg_array.shape[0] != 0:
                    #print(emg_array)
                    #print(bp_parameter, nt_parameter, lp_parameter)                       
                    init_time = init_time + 50  #len(new_emg_observation)
                    emg_observation = np.sqrt(np.mean(emg_array**2, axis=1))
                    reward, initial_max_min_rms_values = calculate_emg_level(emg_observation, initial_max_min_rms_values, init_time)
                    print(f"reward:{reward}, EMG:{emg_observation}")
        except Exception as e:
            print(f"WebSocket error: {e}")
            pass
        finally:
            ws.close()

def process_data_from_websocket(data, bp_parameter, nt_parameter, lp_parameter):
    emg_values = np.zeros((8,50))
    j = 0
    try:
        data_dict = json.loads(data)
        if "contents" in data_dict:
            # 提取 serial_number 和 eeg 的值
            serial_numbers_eegs = [(item['serial_number'][0], item['eeg']) for item in data_dict['contents']]
            # 輸出結果
            for serial_number, eeg in serial_numbers_eegs:
                # print(f"Serial Number: {serial_number}, EEG: {eeg}")
                for i in range(8):
                    emg_values[i,j] = eeg[i]      # 最新的50筆emg資料
                j+=1
            try:
                emg_array = np.empty((8, 50))
                for k in range(8):
                    #print("check2",emg_values[k],bp_parameter[k], nt_parameter[k], lp_parameter[k])
                    emg_array[k], bp_parameter[k], nt_parameter[k], lp_parameter[k] = process_emg_signal(emg_values[k],bp_parameter[k], nt_parameter[k], lp_parameter[k])
                    #print("check5",emg_values[k],bp_parameter[k], nt_parameter[k], lp_parameter[k])
                return emg_array, bp_parameter, nt_parameter, lp_parameter
            except Exception as e:
                print(f"处理信号时发生错误: {e}")
                return np.array([]), bp_parameter, nt_parameter, lp_parameter
    except json.JSONDecodeError:
        print("Failed to decode JSON from WebSocket")
    except Exception as e:
        # print(f"Error processing data from WebSocket: {e}")
        return np.array([]), bp_parameter, nt_parameter, lp_parameter

# 带通滤波器设计
def bandpass_filter(data, lowcut, highcut, fs, bp_filter_state, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    if bp_filter_state.all() == 0:
        bp_filter_state = lfilter_zi(b, a)
        #print("check4", bp_filter_state)
    y, bp_filter_state = lfilter(b, a, data, zi=bp_filter_state)
    return y, bp_filter_state

# 陷波滤波器设计 
def notch_filter(data, notch_freq, fs, notch_filter_state, quality_factor=30):
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = iirnotch(freq, quality_factor)
    if notch_filter_state.all() == 0:
        notch_filter_state = lfilter_zi(b, a)
        #print("check5", notch_filter_state)
    y, notch_filter_state = lfilter(b, a, data, zi=notch_filter_state)
    return y, notch_filter_state

# 全波整流
def full_wave_rectification(data):
    return np.abs(data)

# 低通滤波器设计（提取包络）
def lowpass_filter(data, cutoff, fs, lp_filter_state, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if lp_filter_state.all() == 0:
        lp_filter_state = lfilter_zi(b, a)
        #print("check8", lp_filter_state)
    y, lp_filter_state = lfilter(b, a, data, zi=lp_filter_state)
    return y, lp_filter_state

# 实时信号处理函数
def process_emg_signal(data, bp_parameter, nt_parameter, lp_parameter, fs=1000):
    #print("check3",data, bp_parameter, nt_parameter, lp_parameter)
    # 带通滤波
    bandpassed, bp_parameter = bandpass_filter(data, 20, 450, fs, bp_parameter)
    
    # 50Hz陷波滤波
    notch_filtered, nt_parameter = notch_filter(bandpassed, 50, fs, nt_parameter)
    # 全波整流
    rectified = full_wave_rectification(notch_filtered)
    # 低通滤波提取包络
    enveloped, lp_parameter = lowpass_filter(rectified, 10, fs, lp_parameter)
    #print(enveloped)
    #print(bp_parameter, nt_parameter, lp_parameter)
    return enveloped, bp_parameter, nt_parameter, lp_parameter
# 以下為肌力回饋
def calculate_emg_level(data, initial_max_min_rms_values, times):

    #前1秒為暖機
    if times <= 1000:
        return 0, initial_max_min_rms_values
    # 使用第1秒到第10秒的数据来确定初始的最小、最大RMS值
    elif 1000 < times <= 10000:
        for i in range(8):
            rms_values = data[i]
            if initial_max_min_rms_values[i][0] == 0 or rms_values > initial_max_min_rms_values[i][0]:
                initial_max_min_rms_values[i][0] = rms_values
            elif initial_max_min_rms_values[i][1] == 0 or rms_values < initial_max_min_rms_values[i][1]:
                initial_max_min_rms_values[i][1] = rms_values
        return 0, initial_max_min_rms_values
    #每0.05秒傳出reward值
    else:
        reward = np.zeros(8)
        y = 0
        for i in range(8):
            rms_values = data[i]
            reward[i] = map_to_levels(rms_values, initial_max_min_rms_values[i])
            y = y + reward[i]
        return y, initial_max_min_rms_values

def calculate_rms(signal):
    """计算信号的RMS值。"""
    return np.sqrt(np.mean(signal**2))

def map_to_levels(value, max_min_rms_values):
    """将值映射到超出5到-5级的线性值上，基于放松阈值和初始最大RMS值，
    但在上下限内分为5到-5十个等级区间。"""
    # 计算每个等级的值范围大小
    level_range = (max_min_rms_values[0] - max_min_rms_values[1]) / 10
    
    if value <= max_min_rms_values[1]:
        # 计算低于min_rms_values的值应映射到哪个级别
        level_diff = (value - max_min_rms_values[1]) / level_range
        return int(round(5 + level_diff))
    elif value >= max_min_rms_values[0]:
        # 计算高于max_rms_values的值应映射到哪个级别
        level_diff = (value - max_min_rms_values[0]) / level_range
        return int(round(-5 + level_diff))
    else:
        # 线性映射到5到-5
        normalized_value = (value - max_min_rms_values[1]) / (max_min_rms_values[0] - max_min_rms_values[1])
        return int(round(normalized_value * (-10))) + 5

def main():
    websocket_uri = "ws://localhost:31278/ws"

    bp_parameter = np.zeros((8,8))
    nt_parameter = np.zeros((8,2))
    lp_parameter = np.zeros((8,4))
    initial_max_min_rms_values = np.zeros((8,2))
    init_time = 0

    read_specific_data_from_websocket(websocket_uri,bp_parameter, nt_parameter, lp_parameter,initial_max_min_rms_values,init_time)

if __name__ == "__main__":
    main()

