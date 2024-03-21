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

async def read_line(reader):
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
        info = await read_line(reader)
        # print("raw_data: ", info)
        data = analysis(info)
        print(data[0], data[3])
        await asyncio.sleep(0.01)
        value = random.randint(-5, 5) * 1000
        await FREEX_CMD(writer, "E", "0", "C", f"{value}")
        print(f"done {i}")
        i += 1

if __name__ == "__main__":
    asyncio.run(main())
    print("Program terminated.")
