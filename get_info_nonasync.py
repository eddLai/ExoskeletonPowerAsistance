import socket
import random
import numpy as np
import time
import keyboard

running = True

def FREEX_CMD(sock, mode1="E", value1="0", mode2="E", value2="0"):
    cmd_str = f"X {mode1} {value1} {mode2} {value2}\r\n\0"
    cmd_bytes = cmd_str.encode('ascii')
    sock.send(cmd_bytes)

def connect_FREEX(host='192.168.4.1', port=8080):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print(f"Successfully connected to {host}:{port}")
    return sock

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

def old_analysis(data, last_valid_data):
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
    print(f"Failed to analyze data: {data}")
    return last_valid_data, False

def new_read_line(sock):
    data = sock.recv(1024)
    if not data:
        return None
    return data.decode('ascii').rstrip('\r\n\0')

def get_INFO(sock):
    while True:
        info = new_read_line(sock)
        if info is None or info == "":
            continue
        # print("raw_data: ", info)  
        analyzed_data, is_analyzed = analysis(info)
        if is_analyzed:
            break
    return analyzed_data

def main():
    i = 0
    last_valid_data = np.zeros([9,])  # Initialize with zeros
    sock = connect_FREEX()
    while not keyboard.is_pressed('q'):
        # info = new_read_line(sock)
        # if info is None or info == "":  # 检查是否收到空数据
        #     print("Received empty data. Skipping command.")
        #     continue  # 跳过当前循环迭代，不发送命令
        # # print("raw_data: ", info)
        # data, valid = analysis(info, last_valid_data)
        # if valid:
        #     last_valid_data = data  # Update last valid data if current data is valid
        #     print("R_angle", data[0], "L_angle", data[3])
        #     value = random.randint(-5, 5) * 1000
        #     FREEX_CMD(sock, "E", "0", "E", "0")
        #     # FREEX_CMD(sock, "E", "0", "C", str(value))
        #     print(f"done {i}")
        # else:
        #     pass
        #     # print("Invalid data received. Skipping command.")
        data = get_INFO(sock)
        print("R_angle", data[0], "L_angle", data[3])
        i += 1

    FREEX_CMD(sock)
    sock.close()


def old_main():
    i = 0
    last_valid_data = np.zeros([9,])  # Initialize with zeros
    sock = connect_FREEX()
    while not keyboard.is_pressed('q'):
        info = new_read_line(sock)
        if info is None:
            print("No more data. Exiting.")
            break
        print("raw_data: ", info)
        data, valid = analysis(info, last_valid_data)
        if valid:
            last_valid_data = data  # Update last valid data if current data is valid
        print("R_angle", data[0], "L_angle", data[3])
        value = random.randint(-5, 5) * 1000
        FREEX_CMD(sock, "E", "0", "E", "0")
        # FREEX_CMD(sock, "E", "0", "C", f"{value}")
        print(f"done {i}")
        i += 1

    FREEX_CMD(sock)
    sock.close()

if __name__ == "__main__":
    main()
    print("Program terminated.")
