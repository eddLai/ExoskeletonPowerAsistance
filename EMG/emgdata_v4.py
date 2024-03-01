import asyncio
import aiohttp
import websockets
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from matplotlib.animation import FuncAnimation
import keyboard

stop_event = asyncio.Event()

times = []
all_emg_values = []
filtered_emg_values = []

global start_time
start_time = None

plt.ion()

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
        serial_numbers_emgs = [(item['serial_number'][0], item['emg']) for item in data_dict['contents']]
        for serial_number, emg in serial_numbers_emgs:
            print(f"Serial Number: {serial_number}, emg: {emg}")
            if start_time is None:
                start_time = serial_number
            time = (serial_number - start_time)*0.001
            times.append(time)
            all_emg_values.append(emg)
            emg_values.append(emg[0])
        await process_data_from_emg(emg_values)
    except json.JSONDecodeError:
        print("Failed to decode JSON from WebSocket")
    except Exception as e:
        print(f"Error processing data from WebSocket: {e}")

def bandpass_filter(signal, low_freq=20, high_freq=450, fs=1000, order=4):
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def bandstop_filter(signal, low_cut=55, high_cut=65, fs=1000, order=4):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = filtfilt(b, a, signal)
    return y

def lowpass_filter(signal, cutoff=5, fs=1000, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, signal)
    return y

async def process_data_from_emg(raw_emg):
    filtered_emg = bandpass_filter(raw_emg)
    notch_filtered_emg = bandstop_filter(filtered_emg)
    rectified_emg = np.abs(notch_filtered_emg)
    smoothed_emg = lowpass_filter(rectified_emg)
    filtered_emg_values.extend(smoothed_emg)

fig, axs = plt.subplots(2, 1, figsize=(10, 12))

def init():
    axs[0].set_xlim(0, 5)  
    axs[0].set_ylim(-1, 1) 
    axs[1].set_xlim(0, 5)  
    axs[1].set_ylim(-1, 1) 

def update(frame):
    if times:
        axs[0].plot(times, [emg[0] for emg in all_emg_values], 'r-')
        axs[1].plot(times, filtered_emg_values, 'b-')
    return axs

ani = FuncAnimation(fig, update, init_func=init, blit=False, interval=100)

def on_press_key(e):
    if e.name == 'esc':
        stop_event.set()
        print("Stop event set, exiting...")

async def main():
    websocket_uri = "ws://localhost:31278/ws"
    keyboard.on_press(on_press_key)
    await read_specific_data_from_websocket(websocket_uri)

if __name__ == "__main__":
    asyncio.run(main())
