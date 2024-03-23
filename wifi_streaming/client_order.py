import socket
import numpy as np
from EMG import emg_nonasync

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

def FREEX_CMD(sock, mode1="E", value1="0", mode2="E", value2="0"):
    cmd_str = f"X {mode1} {value1} {mode2} {value2}\r\n\0"
    cmd_bytes = cmd_str.encode('ascii')
    sock.send(cmd_bytes)

def connect_FREEX(host='192.168.4.1', port=8080):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print(f"Successfully connected to {host}:{port}")
    return sock

def read_line(sock):
    data = sock.recv(1024)
    if not data:
        return None
    try:
        data = data.decode('ascii').rstrip('\r\n\0')
        return data
    except UnicodeDecodeError as e:
        print(f"Error decoding data: {e}")
        return None

def get_INFO(sock, uri, bp_parameter, nt_parameter, lp_parameter):
    while True:
        info = read_line(sock)
        if info is None or info == "":
            continue
        # print("raw_data: ", info)  
        analyzed_data, is_analyzed = analysis(info)
        if is_analyzed:
            break
    # print("analyzed: ", analyzed_data)
    # analyzed_data = np.random.rand(9)
    # emg
    emg_observation, bp_parameter, nt_parameter, lp_parameter = emg_nonasync.read_specific_data_from_websocket(uri ,bp_parameter, nt_parameter, lp_parameter)

    return analyzed_data, emg_observation, bp_parameter, nt_parameter, lp_parameter

def if_not_safe(limit, angle, speed):
    if (angle >= limit and speed > 0) or (angle <= -limit and speed < 0):
        return True
    else:
        return False

last_action_was_zero = False
left_disabled = False
right_disabled = False

def send_action_to_exoskeleton_speed(writer, action, state):
    global last_action_was_zero
    action[0] *= 10000
    action[1] *= 10000
    LIMIT = 35
    CURRENT_LIMIT = 50000
    R_angle = state[0]
    L_angle = state[3]
    R_current = state[2]
    L_current = state[5]
    print(state)
    current_action_is_zero = action[0] == 0 and action[1] == 0
    if (current_action_is_zero and last_action_was_zero):
        return

    check_R = if_not_safe(LIMIT, action[0], R_angle) or R_current > CURRENT_LIMIT
    check_L = if_not_safe(LIMIT, action[1], L_angle) or L_current > CURRENT_LIMIT
    if (check_R and check_L) or current_action_is_zero:
        # print("both aborted")
        FREEX_CMD(writer, "E", "0", "E", "0")
    elif check_R or (action[0] == 0):
        print("motor R: ", action[0], "\tangle: ", R_angle, "\tcurrent: ", R_current, "aborted")
        FREEX_CMD(writer, "E", "0", 'C', f"{action[1]}")
    elif check_L or (action[1] == 0):
        print("motor L: ", action[1], "\tangle: ", L_angle, "\tcurrent: ", L_current, "aborted")
        FREEX_CMD(writer, 'C', f"{action[0]}", "E", "0")
    else:
        # print("OK")
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
