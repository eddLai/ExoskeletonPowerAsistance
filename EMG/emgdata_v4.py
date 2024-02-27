import asyncio
import aiohttp
import websockets
import json
import matplotlib.pyplot as plt
import time
import keyboard 
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd

stop_event = asyncio.Event()

times = []
all_emg_values = []
filtered_emg_values = []

global start_time
start_time = None

async def read_specific_data_from_websocket(uri):
    try:
        async with websockets.connect(uri) as websocket:
            while not stop_event.is_set():
                data = await websocket.recv()
                await process_data_from_websocket(data)
    except Exception as e:
         print(f"WebSocket error: {e}")

async def process_data_from_websocket(data):
    global start_time
    emg_values = []
    try:
        data_dict = json.loads(data)
        if "algorithm" in data_dict or "response" in data_dict:
            return
        # 提取 serial_number 和 emg 的值
        serial_numbers_emgs = [(item['serial_number'][0], item['emg']) for item in data_dict['contents']]

        # 輸出結果
        for serial_number, emg in serial_numbers_emgs:
            print(f"Serial Number: {serial_number}, emg: {emg}")
            if start_time is None:
                start_time = serial_number
            time = (serial_number - start_time)*0.001
            times.append(time)
            all_emg_values.append(emg)  # 全部的emg資料
            emg_values.append(emg[0])      # 最新的50筆emg資料
        await process_data_from_emg(emg_values)
    except json.JSONDecodeError:
        print("Failed to decode JSON from WebSocket")
    except Exception as e:
        print(f"Error processing data from WebSocket: {e}")

# 定義一個帶通濾波器
def bandpass_filter(signal, low_freq=20, high_freq=450, fs=1000, order=4):
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# 定義一個帶拒濾波器，消除電源線干擾
def bandstop_filter(signal, low_cut=55, high_cut=65, fs=1000, order=4):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = filtfilt(b, a, signal)
    return y

# 定義一個平滑濾波器(低通)，繪製
def lowpass_filter(signal, cutoff=5, fs=1000, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, signal)
    return y

# emg訊號前處理
async def process_data_from_emg(raw_emg):
    filtered_emg = bandpass_filter(raw_emg)
    notch_filtered_emg = bandstop_filter(filtered_emg)
    rectified_emg = np.abs(notch_filtered_emg)
    smoothed_emg = lowpass_filter(rectified_emg)
    filtered_emg_values.extend(smoothed_emg)

def plot_data():
    # 假设all_emg_values现在包含的是包含两个元素的列表，即每个时间点有两个emg值
    # 分别提取两组emg数据
    emg_values_0 = [emg[0] for emg in all_emg_values]
    emg_values_1 = [emg[1] for emg in all_emg_values]  # 提取每个时间点的第二个emg值
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 12)) 
    
    axs[0].plot(times, emg_values_0, 'r-')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('emg Value')
    axs[0].set_title('RSA')
    
    axs[1].plot(times, filtered_emg_values, 'b-')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('emg Value')
    axs[1].set_title('filtered EMG')
    
    plt.tight_layout()
    plt.show()

def on_press_key(e):
    if e.name == 'esc':
        stop_event.set()
        print("Stop event set, exiting...")

async def main():
    websocket_uri = "ws://localhost:31278/ws"
    keyboard.on_press(on_press_key)
    await read_specific_data_from_websocket(websocket_uri)
    plot_data()

if __name__ == "__main__":
    asyncio.run(main())
