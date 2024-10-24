# import socket
import serial
import numpy as np
from data_streaming.EMG import emg_localhost

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
            return np.array(result), True
    # print(f"Failed to analyze data: {data}")
    return np.zeros(9), False

def FREEX_CMD(serial_conn, mode1="E", value1="0", mode2="E", value2="0"):
    cmd_str = f"X {mode1} {value1} {mode2} {value2}\r\n\0"
    try:
        serial_conn = serial.write(cmd_str.encode('ascii'))
        # sock.send(cmd_bytes)
    except Exception as e:
        FREEX_CMD(serial_conn, "E", "0", "E", "0")
        print(f"Error when sending: {e}")

def connect_FREEX(port='/dev/ttyUSB0', baudrate=115200):
    try:
        serial_conn = serial.Serial(port, baudrate, timeout=1)
        print(f"Successfully connected to {port} with baudrate {baudrate}")
        return serial_conn
    except Exception as e:
        print(f"Failed to connect to {port}: {e}")
        return None

def read_line(serial_conn):
    try:
        data = serial_conn.readline().decode('ascii').rstrip('\r\n')
        if not data:
            return None
        return data
    except Exception as e:
        FREEX_CMD(serial_conn, "E", "0", "E", "0")
        print(f"Error when reading line: {e}")
        return None

def get_INFO(serial_conn, uri, bp_parameter, nt_parameter, lp_parameter):
    while True:
        info = read_line(serial_conn)
        if info is None or info == "":
            FREEX_CMD(serial_conn, "E", "0", "E", "0")
            print("stucking in EXO data failed")
            continue
        # print("raw_data: ", info)  
        analyzed_data, is_analyzed = analysis(info)
        if is_analyzed:
            break
        else:
            FREEX_CMD(serial_conn, "E", "0", "E", "0")
    # print("analyzed: ", analyzed_data)
    # analyzed_data = np.random.rand(9)
    # emg
    emg_observation, bp_parameter, nt_parameter, lp_parameter = emg_localhost.read_specific_data_from_websocket(uri ,bp_parameter, nt_parameter, lp_parameter)

    return analyzed_data, emg_observation, bp_parameter, nt_parameter, lp_parameter

def if_not_safe(limit, angle, speed):
    # if (angle >= limit and speed > 0) or (angle <= -limit and speed < 0):
    if (angle >= limit) or (angle <= -limit):    
        return True
    else:
        return False

last_action_was_zero = False
left_disabled = False
right_disabled = False

def send_action_to_exoskeleton_speed(writer, action, state):
    global last_action_was_zero
    action[0] *= 10000  # Scale the action for the right side
    action[1] *= 10000  # Scale the action for the left side
    LIMIT = 10
    CURRENT_LIMIT = 50000
    R_angle, L_angle = state[0], state[3]
    R_current, L_current = state[2], state[5]
    current_action_is_zero = all(a == 0 for a in action)

    if current_action_is_zero and last_action_was_zero:
        return

    # print(f"action: {action}, angle: {R_angle}, {L_angle}, current: {R_current}, {L_current}")

    check_R = if_not_safe(LIMIT, R_angle, action[0])
    check_L = if_not_safe(LIMIT, L_angle, action[1])

    if check_R and check_L:
        # print("both actions aborted due to safety")
        FREEX_CMD(writer, "E", "0", "E", "0")
    elif check_R:
        # print("Right action aborted due to safety")
        FREEX_CMD(writer, "E", "0", 'C', f"{action[1]}" if not check_L else "0")
    elif check_L:
        # print("Left action aborted due to safety")
        FREEX_CMD(writer, 'C', f"{action[0]}" if not check_R else "0", "E", "0")
    else:
        FREEX_CMD(writer, 'C', f"{action[0]}", 'C', f"{action[1]}")

    last_action_was_zero = current_action_is_zero
    print("-----------------------------")




def send_action_to_exoskeleton(writer, action, state, control_type='speed'):
    if control_type == 'speed':
        return send_action_to_exoskeleton_speed(writer, action, state)
    elif control_type == 'disable':
        pass
    else:
        raise ValueError("Unknown control_type specified.")
