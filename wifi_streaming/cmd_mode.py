import socket
import time
import random
import keyboard

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

def calculate_rotation_time(angle, angular_velocity):
    """
    根據角度和角速度計算完成旋轉所需的時間
    :param angle: 需要旋轉的角度（度）
    :param angular_velocity: 角速度（度/秒）
    :return: 完成旋轉所需的時間（秒）
    """
    if angular_velocity == 0:
        return 0
    return abs(angle / angular_velocity)


def main():
    client_socket = connect_FREEX()
    FREEX_CMD(client_socket, "A", "0", "A", "0")

    # 假設左右腳需要旋轉的角度為 50 度
    angle_to_rotate = 50  # 單位：度
    right_foot_speed = 5000  # 右腳初始角速度 (25 deg/s)
    left_foot_speed = -5000  # 左腳初始角速度 (-25 deg/s)

    print("按下Enter進行一次角速度變化，按下q結束")

    while True:
        if keyboard.is_pressed('enter'):
            # 計算所需等待的時間，因為速度是0.01度/秒，先將速度轉換為度/秒
            time_to_complete = calculate_rotation_time(angle_to_rotate, right_foot_speed / 100)

            # 發送角速度命令，右腳向前，左腳向後
            FREEX_CMD(client_socket, "C", f"{right_foot_speed}", "C", f"{left_foot_speed}")
            print(f"右腳速度: {right_foot_speed / 100} 度/秒, 左腳速度: {left_foot_speed / 100} 度/秒")

            # 等待轉動完成
            print(f"等待 {time_to_complete} 秒讓動作完成...")
            time.sleep(time_to_complete)
            FREEX_CMD(client_socket, "E", 0, "E", 0)

            # 翻轉角速度，右腳向後，左腳向前
            right_foot_speed = -right_foot_speed
            left_foot_speed = -left_foot_speed
            FREEX_CMD(client_socket, "C", f"{right_foot_speed}", "C", f"{left_foot_speed}")
            time.sleep(time_to_complete)
            FREEX_CMD(client_socket, "E", 0, "E", 0)

            # 避免重複觸發按鍵
            while keyboard.is_pressed('enter'):
                pass

        if keyboard.is_pressed('q'):  # 按下 q 鍵退出程式
            print("退出")
            break

    # str = input("cmd_str:example X A 20 E 1\r\n\0")
    # for value in range(0, 8000, 1000):
    #     FREEX_CMD(client_socket, "A", "0", "C", f"{value}")
    #     time.sleep(0.2)
    # FREEX_CMD(client_socket, "C", "0", "C", "0")
    # client_socket.close()

if __name__ == "__main__":
    main()
    print("Program terminated.")
