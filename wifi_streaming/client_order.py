import asyncio
import numpy as np
from EMG import emgdata
import numpy as np
from scipy.signal import butter, lfilter, iirnotch, lfilter_zi
import asyncio
import aiohttp
import websockets
import json
import time

def analysis(data):
    result = []
    if data.startswith("X"):
        parts = data[1:].strip().split()
        count = 0
        for part in parts:
            if count == 9:
                break
            clean_part = ''.join(filter(lambda x: x in '0123456789.-', part))
            if clean_part and clean_part != '-' and not clean_part.endswith('.'):
                try:
                    result.append(float(clean_part))
                    count += 1
                except ValueError as e:
                    print(f"Error converting '{clean_part}' to float: {e}")
                    continue
        
        if len(result) == 9:
            return np.array(result)
    return np.array([])


async def FREEX_CMD(writer, mode1="A", value1="-5000", mode2="A", value2="-5000"):
    cmd_str = f"X {mode1} {value1} {mode2} {value2}\r\n\0"
    # print(f"Sending command: {cmd_str}")
    writer.write(cmd_str.encode('ascii'))
    await writer.drain()

async def connect_FREEX(host='192.168.4.1', port=8080):
    reader, writer = await asyncio.open_connection(host, port)
    print(f"Successfully connected to {host}:{port}")
    return reader, writer

async def get_INFO(reader, uri, bp_parameter, nt_parameter, lp_parameter):
    try:
        data = await reader.readuntil(separator=b'\n')
        data_str = data.decode('utf-8').strip()
        # print("raw data: ", data_str)
        analyzed_data = analysis(data_str)
        # print("analyzed: ", analyzed_data)
        # analyzed_data = np.random.rand(9)
        # emg
        emg_observation, bp_parameter, nt_parameter, lp_parameter = await emgdata.read_specific_data_from_websocket(uri ,bp_parameter, nt_parameter, lp_parameter)
        return analyzed_data, emg_observation, bp_parameter, nt_parameter, lp_parameter
    
    except asyncio.IncompleteReadError as ex:
        print(f"An error occurred: {ex}")
        return np.array([]), np.array([]), bp_parameter, nt_parameter, lp_parameter

def check_if_safe(limit:int, angle, speed):
    print(angle)
    angle = int(angle)
    if angle is None:
        print("ur safe now (angle is None)")
        return 0
    elif (angle >= limit and speed > 0) or (angle <= -limit and speed < 0):
        print("ur safe now")
        return 0
    else:
        return speed
    
async def disable_exoskeleton(writer):
    # await FREEX_CMD(writer, "E", "0", "E", "0")
    pass

async def if_not_safe(limit, angle, speed):
    if (angle >= limit and speed > 0) or (angle <= -limit and speed < 0):
        return True
    else:
        return False

async def send_action_to_exoskeleton_speed(writer, action, state):
    action[0] *= 1000
    action[1] *= 1000
    LIMIT = 10
    R_angle = state[0]
    L_angle = state[3]
    check_R = await if_not_safe(LIMIT, action[0], R_angle)
    check_L = await if_not_safe(LIMIT, action[1], L_angle)
    if check_R and check_L:
        print("both aborted")
        await FREEX_CMD(writer, "E", "0", "E", "0")
    elif check_R:
        print("motor R: ", action[0], "\tangle: ", R_angle, ", aborted")
        await FREEX_CMD(writer, "E", "0", 'C', f"{action[1]}")
    elif check_L:
        await FREEX_CMD(writer, 'C', f"{action[0]}", "E", "0")
        print("motor L: ", action[1], "\tangle: ", L_angle, ", aborted")
    else:
        print("OK")
        await FREEX_CMD(writer, 'C', f"{action[0]}", 'C', f"{action[1]}")
    print("-----------------------------")

async def send_action_to_exoskeleton(writer, action, state, control_type='speed'):
    if control_type == 'speed':
        return await send_action_to_exoskeleton_speed(writer, action, state)
    elif control_type == 'disable':
        return await disable_exoskeleton(writer)
    else:
        raise ValueError("Unknown control_type specified.")
