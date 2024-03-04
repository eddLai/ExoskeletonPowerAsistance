import asyncio
import random
import numpy as np
import keyboard

def analysis(data):
    # print("raw data", data)
    result = []
    if data.startswith("X"):
        parts = data[1:].strip().split()
        for part in parts:
            clean_part = ''.join(filter(lambda x: x in '0123456789.-', part))
            if clean_part and clean_part != '-' and not clean_part.endswith('.'):
                try:
                    result.append(float(clean_part))
                except ValueError as e:
                    print(f"Error converting '{clean_part}' to float: {e}")
                    continue  # 转换失败时跳过该部分
    return np.array(result)

async def FREEX_CMD(writer, mode1="A", value1="-5000", mode2="A", value2="-5000"):
    cmd_str = f"X {mode1} {value1} {mode2} {value2}\r\n\0"
    # print(f"Sending command: {cmd_str}")
    writer.write(cmd_str.encode('ascii'))
    await writer.drain()

async def connect_FREEX(host='192.168.4.1', port=8080):
    reader, writer = await asyncio.open_connection(host, port)
    print(f"Successfully connected to {host}:{port}")
    return reader, writer

async def get_INFO(reader):
    try:
        data = await reader.readuntil(separator=b'\n')
        data_str = data.decode('utf-8').strip()
        analyzed_data = analysis(data_str)
        # print(analyzed_data)
        return analyzed_data
    except asyncio.IncompleteReadError as ex:
        print(f"An error occurred: {ex}")
        return None

def check_if_safe(limit:int, angle, speed):
    if angle >= limit and speed > 0 or angle <= -limit and speed < 0 or angle == None:
        print("ur safe now")
        return "0"
    else:
        return str(speed)

async def main():
    host = '192.168.4.1'
    port = 8080
    reader, writer = await connect_FREEX(host, port)
    
    try:
        while True:
            if keyboard.is_pressed('q'):
                print("Exiting...")
                break
            observation = await get_INFO(reader)
            value = str(random.randint(-5, 5) * 1000)
            if len(observation) > 0:
                value = check_if_safe(10, observation[0], value)
            await FREEX_CMD(writer, "A", "0", "C", value)
            await asyncio.sleep(0.05)
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        writer.close()
        await writer.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
