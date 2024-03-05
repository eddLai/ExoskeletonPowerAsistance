import asyncio
import numpy as np
# import keyboard
# import random

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
    return np.array([])


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
        # print("raw data: ", data_str)
        analyzed_data = analysis(data_str)
        # print("analyzed: ", analyzed_data)
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
    
async def send_action_to_exoskeleton_speed(writer, action):
    motor_speed = action * 1000
    await FREEX_CMD(writer, 'C', f"{motor_speed[0]}", 'C', f"{motor_speed[1]}")

def send_action_to_exoskeleton_angle(client_socket, action):
    # 将动作值映射到[-45, 45]度的角度上
    # 动作值应该是一个包含两个元素的数组，分别对应两个电机
    motor_angle_1 = int(action[0] * 4500)  # 将动作值映射到角度值，考虑到角度单位是0.01度
    motor_angle_2 = int(action[1] * 4500)
    FREEX_CMD(client_socket, 'A', motor_angle_1, 'A', motor_angle_2)
    return True

async def send_action_to_exoskeleton(writer, action, control_type='torque'):
    if control_type == 'speed':
        return await send_action_to_exoskeleton_speed(writer, action)
    elif control_type == 'angle':
        return await send_action_to_exoskeleton_angle(writer, action)
    else:
        raise ValueError("Unknown control_type specified.")


# async def main():
#     host = '192.168.4.1'
#     port = 8080
#     reader, writer = await connect_FREEX(host, port)
    
#     try:
#         while True:
#             if keyboard.is_pressed('q'):
#                 print("Exiting...")
#                 break
#             observation = await get_INFO(reader)
#             value = str(random.randint(-5, 5) * 1000)
#             if len(observation) > 0:
#                 value = check_if_safe(10, observation[0], value)
#             await FREEX_CMD(writer, "A", "0", "C", value)
#             await asyncio.sleep(0.05)
#     except KeyboardInterrupt:
#         print("Program terminated by user.")
#     finally:
#         writer.close()
#         await writer.wait_closed()

# if __name__ == "__main__":
#     asyncio.run(main())
