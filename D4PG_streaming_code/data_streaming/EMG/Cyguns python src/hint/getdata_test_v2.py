from enum import Enum
from struct import unpack, pack
from threading import Thread
import time
import ftd2xx
from cobs import cobs
import queue
import threading

# 初始化队列和锁
command_queue = queue.Queue()
data_queue = queue.Queue()
message_lock = threading.Lock()

# List Devices flags
LIST_NUMBER_ONLY = 0x80000000
LIST_BY_INDEX = 0x40000000
LIST_ALL = 0x20000000

# OpenEx flags
OPEN_BY_SERIAL_NUMBER = 1
OPEN_BY_DESCRIPTION = 2
OPEN_BY_LOCATION = 4

FT_WordLength_8 = 8
FT_StopBits_1 = 0
FT_Parity_None = 0

# 定义 Command 类
class Command:
    class Type(Enum):
        ON = 1
        OFF = 2
        IMPEDANCE_AC = 3
        IMPEDANCE_DC = 4
        IMPEDANCE_AUTO = 5
        IMPEDANCE_OFF = 6
        GET_MACHINE_INFO = 7
        INPUT_AP_SSID_KEY = 8
        READ_SYNC_TICK = 9
        GET_CONN_STATUS = 10

    def __init__(self, command_type, ssid=None, key=None, ap_mode=None):
        self.type = command_type
        self.ssid = ssid
        self.key = key
        self.ap_mode = ap_mode

# generator for command
class CommandGenerator(object):
    @classmethod
    def generate(cls, command):
        return {
            Command.Type.ON: lambda: cls.on(),
            Command.Type.OFF: lambda: cls.off(),
            Command.Type.IMPEDANCE_AC: lambda: cls.impedance_ac(),
            Command.Type.IMPEDANCE_DC: lambda: cls.impedance_dc(),
            Command.Type.IMPEDANCE_AUTO: lambda: cls.impedance_auto(),
            Command.Type.IMPEDANCE_OFF: lambda: cls.impedance_off(),
            Command.Type.GET_MACHINE_INFO: lambda: cls.get_machine_info(),
            Command.Type.INPUT_AP_SSID_KEY: lambda: cls.input_ap_ssid_key(command),
            Command.Type.READ_SYNC_TICK: lambda: cls.read_sync_tick(),
            Command.Type.GET_CONN_STATUS: lambda: cls.get_conn_status()
        }.get(command.type)()
        
    @classmethod
    def cmd_name(cls, command):
        return {
            Command.Type.ON: 'ON',
            Command.Type.OFF: 'OFF',
            Command.Type.IMPEDANCE_AC: 'IMPA',
            Command.Type.IMPEDANCE_DC: 'IMPD',
            Command.Type.IMPEDANCE_AUTO: 'IMPAUTO',
            Command.Type.IMPEDANCE_OFF: 'IMPO',
            Command.Type.GET_MACHINE_INFO: 'MINFO',
            Command.Type.INPUT_AP_SSID_KEY: 'SSID',
            Command.Type.READ_SYNC_TICK: 'SYNCTICK',
            Command.Type.GET_CONN_STATUS: 'CONNSTATUS'
        }.get(command.type)

    @classmethod
    def _command_builder(cls, code):
        signal = chr(5)
        signal += chr(1)
        signal += chr(8)
        signal += chr(2)
        signal += chr(code)
        signal += chr(1)
        signal += chr(0)
        return signal

    @classmethod
    def on(cls):
        return cls._command_builder(1)

    @classmethod
    def off(cls):
        return cls._command_builder(2)

    @classmethod
    def impedance_ac(cls):
        return cls._command_builder(6)

    @classmethod
    def impedance_dc(cls):
        return cls._command_builder(5)

    @classmethod
    def impedance_auto(cls):
        return cls._command_builder(19)

    @classmethod
    def impedance_off(cls):
        return cls._command_builder(7)

    @classmethod
    def get_machine_info(cls):
        return cls._command_builder(46)
    
    @classmethod
    def read_sync_tick(cls):
        return cls._command_builder(17)    # 0x11
    
    @classmethod
    def get_conn_status(cls):
        return cls._command_builder(18)    # 0x12

    @classmethod
    def input_ap_ssid_key(cls, command):
        signal = cls._command_builder(81)
        signal += chr(len(command.ssid))
        for c in command.ssid:
            signal += chr(ord(c))
        signal += chr(len(command.key))
        for c in command.key:
            signal += chr(ord(c))
        signal += command.ap_mode
        return signal

# sensor connection profile
class SensorConnection(object):
    connections = {}
    @classmethod
    def get_instance(cls, device_id, baudrate=1000000):
        if not cls.connections.get(device_id):
            cls.connections[device_id] = SensorConnection(device_id, baudrate)
        return cls.connections[device_id]

    def __init__(self, device_id, baudrate):
        self.device_id = device_id
        self.dongle_status = None
        self.is_running = False
        self.stop_reading = True
        self.ftdi_list = None
        self.conn = None
        self.baudrate = baudrate
        self.dongle_list = list()

    def find_dongle(self):
        message_lock.acquire()
        ftdi_list = ftd2xx.listDevices(flags=LIST_ALL | OPEN_BY_DESCRIPTION)
        self.dongle_list = list()
        if ftdi_list is not None:
            for b_name in ftdi_list:
                ftdi_descript = b_name.decode()
                self.dongle_list.append(ftdi_descript)
        message_lock.release()
        return self.dongle_list

    def connect_sensor(self, descrip_id):
        self.find_dongle()
        print(self.dongle_list)
        if not descrip_id in self.dongle_list:
            return self.dongle_status
        if self.conn is None:
            FT_WordLength_8 = 8
            FT_StopBits_1 = 0
            FT_Parity_None = 0
            message_lock.acquire()
            print(descrip_id)
            self.conn = ftd2xx.openEx(descrip_id.encode(), flags=OPEN_BY_DESCRIPTION)
            self.conn.setBaudRate(self.baudrate)
            self.conn.setDataCharacteristics(FT_WordLength_8, FT_StopBits_1, FT_Parity_None)
            self.conn.setTimeouts(10000, 10000)
            self.conn.resetDevice()
            self.dongle_status = True
            print('Driver opened!')
            message_lock.release()
            
            if self.dongle_status:
                print('Open IO thread......')
                self.is_running = True
                i_thread = Thread(target=self.command)
                i_thread.start()
                o_thread = Thread(target=self.read_data)                
                o_thread.start()

        return self.dongle_status

    def command(self):
        while self.is_running:
            cmd = command_queue.get()
            print('Cmd thread got Cmd.')
            command = CommandGenerator.generate(cmd)
            cmd_name = CommandGenerator.cmd_name(cmd)
            if command:                
                self.send_cmd(command, cmd_name)
            command_queue.task_done()
        print('End of Command thread.')

    def send_cmd(self, command, cmd_name):
        message_lock.acquire()
        write_len = self.conn.write(command.encode())
        message_lock.release()

    def read_data(self):
        print('reading...')
        pointer = 0
        read_buffer = None
        dataset = None
        result = bytearray()
        self.stop_reading = False

        while self.is_running:
            buffer_size = self.conn.getQueueStatus()
            if buffer_size > 0:
                message_lock.acquire()
                read_buffer = self.conn.read(buffer_size)
                message_lock.release()

                while pointer < buffer_size:
                    i = read_buffer[pointer]
                    if i == 0:
                        translated = cobs.decode(bytes(result))
                        dataset = eeg_data_factory(translated)
                        result = bytearray()
                        if dataset:
                            data_queue.put(dataset)
                    else:
                        result += bytes([i])
                    pointer += 1
                pointer = 0
        self.stop_reading = True

    def close(self):
        command_queue.put(CommandGenerator.off())
        time.sleep(0.8)
        self.is_running = False
        print('Closing the sensor......')

        while not self.stop_reading:
            time.sleep(0.05)
            continue

        message_lock.acquire()
        self.conn.close()
        message_lock.release()
        self.dongle_status = None
        self.conn = None
        return self.dongle_status

# generator for eeg data
def eeg_data_factory(raw):

    def generate_int_list(data):
        result = []
        for i in range(0, len(data), 3):
            packed_bytes = data[i:i+3]
            value = unpack('>i', (b'\0' if packed_bytes[0] < 128 else b'\xff') + packed_bytes)
            amp_ratio1 = 4.5
            amp_ratio2 = 12
            result.append(value[0]*(amp_ratio1*1000000)/(8388607*amp_ratio2))
        return result

    def generate_location(data):        
        gx = float(unpack('<h', data[0:2])[0]) / 262.4
        gy = float(unpack('<h', data[2:4])[0]) / 262.4
        gz = float(unpack('<h', data[4:6])[0]) / 262.4
        ax = float(unpack('<h', data[6:8])[0]) / 16384.0
        zy = float(unpack('<h', data[8:10])[0]) / 16384.0
        az = float(unpack('<h', data[10:12])[0]) / 16384.0
        return (gx, gy, gz, ax, zy, az)
    
    def generate_EulerAngles(data):      
        roll = float(unpack('>f', data[0:4])[0])
        pitch = float(unpack('>f', data[4:8])[0])
        yaw = float(unpack('>f', data[8:12])[0])
        return (roll, pitch, yaw)

    tag_counts = raw[0]
    args = {}
    idx = 1
    properties = []
    
    while idx < len(raw):
        tag_id = raw[idx]
        idx += 1
        tag_length = raw[idx]
        idx += 1

        data = raw[idx:idx + tag_length]
        idx += tag_length

        if tag_id == 1:
            properties.append('serial_number')
            args['serial_number'] = unpack('<I', data[0:4])
        elif tag_id == 2:
            properties.append('auxiliary')
            args['auxiliary'] = generate_int_list(data)
        elif tag_id == 3:            
            properties.append('eeg')
            args['eeg'] = generate_int_list(data)
        elif tag_id == 4:         
            properties.append('g_sensor')
            args['g_sensor'] = generate_location(data)
        elif tag_id == 6:
            properties.append('battery_power')
            args['battery_power'] = data[0]
        elif tag_id == 7:
            properties.append('event')
            event_code = data[0]
            print('Got event: ' + str(event_code))
            args['event'] = {
                'event_id': [event_code],
                'event_duration': [0]
            }
        elif tag_id == 8:
            properties.append('machine_info')
            args['machine_info'] = 'Hello!'
        elif tag_id == 9:
            return data[0]
        elif tag_id == 10:
            properties.append('sync_tick')
            args['sync_tick'] = unpack('<I', data[0:4])
        elif tag_id == 11:       
            properties.append('euler_angles')
            args['euler_angles'] = generate_EulerAngles(data)

    return properties, args

def process_received_data(received_data, start_time):
    current_time = time.time()
    elapsed_time = current_time - start_time  # 計算相對於基準點的經過時間
    elapsed_milliseconds = int(elapsed_time * 1000)  # 將經過時間轉換為毫秒並取整數

    values = received_data[1]
    serial_number = values['serial_number']
    emg = values['eeg']
    print(f'Elapsed Time: {elapsed_milliseconds} ms')
    print('Serial Number:', serial_number)
    print('EMG Data:', emg)
    print('-------------------')
    return serial_number, emg

def main():
    time_flag = 1
    # 创建 SensorConnection 实例
    sensor = SensorConnection.get_instance(device_id=b'A56QZ86T', baudrate=1000000)

    # 查找可用的 FTDI 设备
    available_devices = sensor.find_dongle()
    print("Available devices:", available_devices)

    # 连接到特定的设备（假设描述 ID 为 'STEEG_DG819202'）
    connected = sensor.connect_sensor(descrip_id='STEEG_DG819202')
    if connected:
        print("Connected to the sensor.")
        
        # 发送打开命令
        command_queue.put(Command(Command.Type.ON))

        # 設置基準時間
        start_time = time.time()
        
        # 读取数据
        while True:
            try:
                received_data = data_queue.get()  # 接收數據 
                if not time_flag:
                    # 設置基準時間
                    start_time = time.time()
                    time_flag = 1
                serial_number, emg = process_received_data(received_data,start_time)
                #print(received_data)
            except queue.Empty:
                print("No data received within timeout period.")
                break
        
        # 关闭连接
        sensor.close()
    else:
        print("Failed to connect to the sensor.")

if __name__ == "__main__":
    main()
