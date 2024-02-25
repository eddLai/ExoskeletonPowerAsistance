import socket

def FREEX_CMD(client_socket, mode1="E", value1="0", mode2="E", value2="0"):
    # Write
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
    result = []
    if data.startswith("X"):
        parts = data[1:].strip().split()
        if len(parts) == 9:
            result = [float(part) for part in parts]
    return result