import socket
import random
import numpy as np
import time

def FREEX_CMD(sock, mode1="A", value1="-5000", mode2="A", value2="-5000"):
    cmd_str = f"X {mode1} {value1} {mode2} {value2}\r\n\0"
    cmd_bytes = cmd_str.encode('ascii')
    sock.send(cmd_bytes)

def connect_FREEX(host='192.168.4.1', port=8080):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print(f"Successfully connected to {host}:{port}")
    return sock

def analysis(data, last_valid_data):
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
    return data.decode('ascii').rstrip('\r\n')

def main():
    i = 0
    last_valid_data = np.zeros([9,])  # Initialize with zeros
    sock = connect_FREEX()
    while True:
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
        FREEX_CMD(sock, "E", "0", "C", f"{value}")
        print(f"done {i}")
        i += 1

    sock.close()

if __name__ == "__main__":
    main()
    print("Program terminated.")
