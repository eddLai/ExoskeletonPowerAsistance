import asyncio
import random
import numpy as np

async def FREEX_CMD(writer, mode1="A", value1="-5000", mode2="A", value2="-5000"):
    cmd_str = f"X {mode1} {value1} {mode2} {value2}\r\n\0"
    # print(cmd_str)
    cmd_bytes = cmd_str.encode('ascii')
    writer.write(cmd_bytes)
    await writer.drain()

async def connect_FREEX(host='192.168.4.1', port=8080):
    reader, writer = await asyncio.open_connection(host, port)
    print(f"Successfully connected to {host}:{port}")
    return reader, writer

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

async def new_read_line(reader):
    """
    读取直到遇到换行符的数据，并返回处理后的行。
    :param reader: asyncio.StreamReader的实例，用于异步读取数据。
    :return: 处理后的行数据（不包含换行符）。
    """
    try:
        # 读取数据直到遇到换行符
        data = await reader.readuntil(separator=b'\n')
        # 将字节数据解码为字符串，假设使用UTF-8编码，根据实际情况调整
        line = data.decode('utf-8').rstrip('\n').rstrip('\r')
        return line
    except asyncio.IncompleteReadError as e:
        # 如果在数据流结束前没有遇到换行符，处理剩余的数据
        line = e.partial.decode('utf-8').rstrip('\n').rstrip('\r')
        if line:
            return line
        else:
            # 如果没有剩余的数据，返回None表示读取结束
            return None
    except Exception as e:
        # 处理其他可能的异常
        print(f"Error reading line: {e}")
        return None

async def old_read_line(reader):
    buffer = ''
    try:
        while True:
            data = await reader.read(1024)
            data = data.decode('utf-8')
            if not data:
                break
            buffer += data
            if '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                return line
    except Exception as ex:
        print(f"An error occurred: {ex}")
        return ex

def display_data(data):
    if data:
        pass
    else:
        print("Invalid data received")

async def main():
    i = 0
    reader, writer = await connect_FREEX()
    while True:
        info = await new_read_line(reader)
        # print("raw_data: ", info)
        data = analysis(info)
        print(data[0], data[3])
        value = random.randint(-5, 5) * 1000
        await FREEX_CMD(writer, "E", "0", "C", f"{value}")
        print(f"done {i}")
        i += 1

if __name__ == "__main__":
    asyncio.run(main())
    print("Program terminated.")
