import socket
import torch
import numpy as np

def FREEX_CMD(client_socket, mode1="E", value1="0", mode2="E", value2="0"):
    # Write, 可能不可以用這種format的方式
    cmd_str = f"X {mode1} {value1} {mode2} {value2}\r\n"
    cmd_bytes = cmd_str.encode('ascii')
    print(f"Sending command: {cmd_str}")
    client_socket.sendall(cmd_bytes)

def connect_FREEX(host='192.168.4.1', port=8080):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host, port))
        print(f"Successfully connected to {host}:{port}")
        return client_socket
    except socket.error as err:
        print(f"Failed to connect: {err}")
        return None

def get_INFO(client_socket):
    buffer = ''
    try:
        while True:
            data = client_socket.recv(1024).decode('utf-8')
            if not data:
                break
            buffer += data
            if '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                return line
    except Exception as ex:
        print(f"An error occurred: {ex}")
        return None

def analysis(data):
    result = np.array([])
    if data.startswith("X"):
        parts = data[1:].strip().split()
        if len(parts) == 9:
            result = np.array([float(part) for part in parts])
    return result

def send_action_to_exoskeleton_torque(client_socket, action):
    # 假设动作范围是[-1, 1]，并且映射到电机速度的范围是[-60000, 60000]
    motor_speed = int(action * 60000)
    FREEX_CMD(client_socket, 'C', motor_speed[0], 'C', motor_speed[1])
    return True

def send_action_to_exoskeleton_angle(client_socket, action):
    # 将动作值映射到[-45, 45]度的角度上
    # 动作值应该是一个包含两个元素的数组，分别对应两个电机
    motor_angle_1 = int(action[0] * 4500)  # 将动作值映射到角度值，考虑到角度单位是0.01度
    motor_angle_2 = int(action[1] * 4500)
    FREEX_CMD(client_socket, 'A', motor_angle_1, 'A', motor_angle_2)
    return True

def send_action_to_exoskeleton(client_socket, action, control_type='torque'):
    if action == "reset":
        FREEX_CMD(client_socket, "A", 0, "A", 0)
        return True
    elif control_type == 'torque':
        return send_action_to_exoskeleton_torque(client_socket, action)
    elif control_type == 'angle':
        return send_action_to_exoskeleton_angle(client_socket, action)
    else:
        raise ValueError("Unknown control_type specified.")


