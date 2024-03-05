import socket
import time
import random

# def FREEX_CMD(client_socket):
#     # Write
#     cmd_str = f"X A -5000 A -5000\r\n\0"
#     cmd_bytes = cmd_str.encode('ascii')
#     print(f"Sending command: {cmd_str}")
#     client_socket.sendall(cmd_bytes)

def FREEX_CMD(client_socket, mode1="A", value1="-5000", mode2="A", value2="-5000"):
    cmd_str = f"X {mode1} {value1} {mode2} {value2}\r\n\0"
    print(cmd_str)
    cmd_bytes = cmd_str.encode('ascii')
    # print(f"Sending command: {cmd_str}")
    client_socket.sendall(cmd_bytes)

def connect_FREEX(host='192.168.4.1', port=8080):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host, port))
        print(f"Successfully connected to {host}:{port}")
    except socket.error as err:
        print(f"Failed to connect: {err}")
    return client_socket

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
        return ex

def analysis(data):
    # print("raw data", data)
    result = []
    if data.startswith("X"):
        parts = data[1:].strip().split()
        if len(parts) == 9:
            result = [float(part) for part in parts]
    return result

def display_data(data):
    if data:
        pass
        # print(f"Motor 1 (Right) - Angle: {data[0]} deg, Speed: {data[1]} deg/s, Current: {data[2] * 0.01} A")
        # print(f"Motor 2 (Left) - Angle: {data[3]} deg, Speed: {data[4]} deg/s, Current: {data[5] * 0.01} A")
        # print(f"Roll: {data[6]} deg, Pitch: {data[7]} deg, Yaw: {data[8]} deg")
    else:
        print("Invalid data received")

def main():
    client_socket = connect_FREEX()
    while True: 
        print(get_INFO(client_socket))
        time.sleep(0.5)
        value = random.randint(-5, 5) * 1000
        FREEX_CMD(client_socket, "A", "0", "C", f"{value}")
    # str = input("cmd_str:example X A 20 E 1\r\n\0")
    # for value in range(0, 8000, 1000):
    #     FREEX_CMD(client_socket, "A", "0", "C", f"{value}")
    #     time.sleep(0.2)
    # FREEX_CMD(client_socket, "C", "0", "C", "0")
    # client_socket.close()

if __name__ == "__main__":
    main()
    print("Program terminated.")
