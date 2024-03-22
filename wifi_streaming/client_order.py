import asyncio
import numpy as np
from EMG import emgdata
import numpy as np
import asyncio

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
    print(f"Failed to analyze data: {data}")
    return np.zeros([9,])


async def FREEX_CMD(writer, mode1="A", value1="-5000", mode2="A", value2="-5000"):
    cmd_str = f"X {mode1} {value1} {mode2} {value2}\r\n\0"
    # print(f"Sending command: {cmd_str}")
    writer.write(cmd_str.encode('ascii'))
    await writer.drain()

async def connect_FREEX(host='192.168.4.1', port=8080):
    reader, writer = await asyncio.open_connection(host, port)
    print(f"Successfully connected to {host}:{port}")
    return reader, writer

import asyncio

async def read_line(reader):
    try:
        data = await reader.readuntil(separator=b'\n')
        line = data.decode('ascii').rstrip('\n').rstrip('\r')
        return line
    except Exception as e:
        print(f"Error reading line: {e}")
        return None

async def get_INFO(reader, uri, bp_parameter, nt_parameter, lp_parameter):
    try:
        info = await read_line(reader)
        print("raw_data: ", info)
        analyzed_data = analysis(info)
        print("analyzed: ", analyzed_data)
        # analyzed_data = np.random.rand(9)
        # emg
        emg_observation, bp_parameter, nt_parameter, lp_parameter = await emgdata.read_specific_data_from_websocket(uri ,bp_parameter, nt_parameter, lp_parameter)
        return analyzed_data, emg_observation, bp_parameter, nt_parameter, lp_parameter
    
    except asyncio.IncompleteReadError as ex:
        print(f"An error occurred: {ex}")
        return np.zeros([9,]), np.zeros([6,]), bp_parameter, nt_parameter, lp_parameter

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

async def if_not_safe(limit, angle, speed):
    if (angle >= limit and speed > 0) or (angle <= -limit and speed < 0):
        return True
    else:
        return False

last_action_was_zero = False

async def send_action_to_exoskeleton_speed(writer, action, state):
    global last_action_was_zero


    action[0] *= 10000
    action[1] *= 10000
    LIMIT = 75
    R_angle = state[0]
    L_angle = state[3]
    R_current = state[2]
    L_current = state[5]
    # print("action: ", action)
    # print("R: ",R_angle, "L: ", L_angle)

    current_action_is_zero = action[0] == 0 and action[1] == 0
    if (current_action_is_zero and last_action_was_zero):
        return

    check_R = await if_not_safe(LIMIT, action[0], R_angle)
    check_L = await if_not_safe(LIMIT, action[1], L_angle)
    if (check_R and check_L) or current_action_is_zero:
        # print("both aborted")
        await FREEX_CMD(writer, "E", "0", "E", "0")
    elif check_R or (action[0] == 0):
        print("motor R: ", action[0], "\tangle: ", R_angle, "\tcurrent: ", R_current, "aborted")
        await FREEX_CMD(writer, "E", "0", 'C', f"{action[1]}")
    elif check_L or (action[1] == 0):
        print("motor L: ", action[1], "\tangle: ", L_angle, "\tcurrent: ", L_current, "aborted")
        await FREEX_CMD(writer, 'C', f"{action[0]}", "E", "0")
    else:
        # print("OK")
        await FREEX_CMD(writer, 'C', f"{action[0]}", 'C', f"{action[1]}")

    last_action_was_zero = current_action_is_zero
    print("-----------------------------")

async def send_action_to_exoskeleton(writer, action, state, control_type='speed'):
    if control_type == 'speed':
        return await send_action_to_exoskeleton_speed(writer, action, state)
    elif control_type == 'disable':
        pass
    else:
        raise ValueError("Unknown control_type specified.")
