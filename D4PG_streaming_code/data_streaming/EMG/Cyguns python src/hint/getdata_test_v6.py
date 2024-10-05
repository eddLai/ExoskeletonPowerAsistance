import queue
import threading
import time
from enum import Enum
from struct import unpack, pack
from threading import Thread
import csv
import scipy.signal as signal

import ftd2xx
from cobs import cobs

# Initialize queues, lock, and stop event
command_queue = queue.Queue()
data_queue = queue.Queue()
message_lock = threading.Lock()
stop_event = threading.Event()

# Define flags and constants
LIST_NUMBER_ONLY = 0x80000000
LIST_BY_INDEX = 0x40000000
LIST_ALL = 0x20000000

OPEN_BY_SERIAL_NUMBER = 1
OPEN_BY_DESCRIPTION = 2
OPEN_BY_LOCATION = 4

FT_WordLength_8 = 8
FT_StopBits_1 = 0
FT_Parity_None = 0

# Define Command class
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

# Command generator class
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

# Sensor connection profile
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
        try:
            ftdi_list = ftd2xx.listDevices(flags=LIST_ALL | OPEN_BY_DESCRIPTION)
            self.dongle_list = []
            if ftdi_list is not None:
                for b_name in ftdi_list:
                    ftdi_descript = b_name.decode()
                    self.dongle_list.append(ftdi_descript)
        finally:
            message_lock.release()
        return self.dongle_list

    def connect_sensor(self, descrip_id):
        self.find_dongle()
        print(self.dongle_list)
        if descrip_id not in self.dongle_list:
            return self.dongle_status
        if self.conn is None:
            FT_WordLength_8 = 8
            FT_StopBits_1 = 0
            FT_Parity_None = 0
            message_lock.acquire()
            try:
                print(descrip_id)
                self.conn = ftd2xx.openEx(descrip_id.encode(), flags=OPEN_BY_DESCRIPTION)
                self.conn.setBaudRate(self.baudrate)
                self.conn.setDataCharacteristics(FT_WordLength_8, FT_StopBits_1, FT_Parity_None)
                self.conn.setTimeouts(10000, 10000)
                self.conn.resetDevice()
                self.dongle_status = True
                print('Driver opened!')
            finally:
                message_lock.release()
            
            if self.dongle_status:
                print('Open IO thread......')
                self.is_running = True
                self.i_thread = Thread(target=self.command)
                self.i_thread.start()
                self.o_thread = Thread(target=self.read_data)                
                self.o_thread.start()

        return self.dongle_status

    def command(self):
        while self.is_running:
            cmd = command_queue.get()
            print('Cmd thread got Cmd.')
            command = CommandGenerator.generate(cmd)
            cmd_name = CommandGenerator.cmd_name(cmd)
            if command:                
                self.send_cmd(command, cmd_name)
                print(f'send cmd: {cmd_name}')
            command_queue.task_done()
        print('End of Command thread.')

    def send_cmd(self, command, cmd_name):
        if self.conn is not None:
            message_lock.acquire()
            try:
                write_len = self.conn.write(command.encode())
            finally:
                message_lock.release()
        else:
            print(f"Warning: Attempted to send command '{cmd_name}' but connection is closed.")

    def read_data(self):
        print('reading...')
        pointer = 0
        read_buffer = None
        dataset = None
        result = bytearray()
        self.stop_reading = False

        while self.is_running and not stop_event.is_set():
            buffer_size = self.conn.getQueueStatus()
            if buffer_size > 0:
                message_lock.acquire()
                try:
                    read_buffer = self.conn.read(buffer_size)
                finally:
                    message_lock.release()

                while pointer < buffer_size:
                    i = read_buffer[pointer]
                    if i == 0:
                        try:
                            translated = cobs.decode(bytes(result))
                            dataset = eeg_data_factory(translated)
                            result = bytearray()
                            if dataset:
                                data_queue.put(dataset)
                        except cobs.DecodeError as e:
                            print(f"COBS decode error: {e}")
                            result = bytearray()  # Reset result in case of decode error
                    else:
                        result += bytes([i])
                    pointer += 1
                pointer = 0
        self.stop_reading = True

    def close(self):
        stop_event.set()  # Signal all threads to stop
        command_queue.put(Command(Command.Type.OFF))
        time.sleep(0.8)
        self.is_running = False

        # Ensure all commands are processed
        command_queue.join()
        
        print('test')
        self.o_thread.join()

        message_lock.acquire()
        try:
            if self.conn is not None:
                self.conn.close()
                self.conn = None
        finally:
            message_lock.release()
        self.dongle_status = None
        print('Sensor connection closed.')
        return self.dongle_status

# EEG data generator
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

# Define a bandpass filter with cutoff frequencies between 20 Hz and 400 Hz
def apply_filter(emg_data):
    fs = 1000  # Sampling frequency
    lowcut = 20  # Low cutoff frequency
    highcut = 400  # High cutoff frequency
    order = 4  # Filter order
    
    # Design the bandpass filter
    b, a = signal.butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')

    # Apply the filter to the data if length is sufficient
    filtered_data = signal.filtfilt(b, a, emg_data)
    return filtered_data

# Modify the process_received_data function to use a buffer
def process_received_data(received_data, start_time, time_flag, buffer, buffer_size):
    if not time_flag[0]:
        # Set baseline time
        start_time[0] = time.time()
        time_flag[0] = True

    current_time = time.time()
    elapsed_time = current_time - start_time[0]  # Calculate elapsed time relative to baseline
    elapsed_milliseconds = int(elapsed_time * 1000)  # Convert elapsed time to milliseconds

    values = received_data[1]
    serial_number = values['serial_number'][0]
    emg = values['eeg']

    # Add new data to the buffer
    buffer.extend(emg)

    # Ensure the buffer does not exceed the specified size
    while len(buffer) > buffer_size:
        buffer.popleft()

    # Apply the bandpass filter to the data only if the buffer has enough data
    if len(buffer) >= buffer_size:
        filtered_emg = apply_filter(list(buffer))  # Convert deque to list for filtering
        print(f'Elapsed Time: {elapsed_milliseconds} ms')
        print('Serial Number:', serial_number)
        print('Filtered EMG Data:', filtered_emg)
        print('-------------------')
        return elapsed_milliseconds, serial_number, filtered_emg
    else:
        print("Not enough data in buffer yet.")
        return elapsed_milliseconds, serial_number, emg  # Return unfiltered data until buffer is filled

def clear_and_initialize_sensor_data_file():
    filename = 'sensor_data.txt'
    with open(filename, 'w') as file:
        #file.write('Elapsed Time (ms)\tSerial Number\tEMG1\tEMG2\tEMG3\tEMG4\tEMG5\tEMG6\tEMG7\tEMG8\n')
        pass
    print(f'{filename} has been cleared and initialized with headers.')

# Save data to txt function
def save_data_to_txt(data_to_save):
    filename = 'sensor_data.txt'
    with open(filename, 'a') as file:  # Append to file
        for row in data_to_save:
            file.write(' '.join(map(str, row)) + '\n')
    print(f'Data appended to {filename}')

# User input handling thread
def user_input_thread():
    while not stop_event.is_set():
        user_input = input("Press 'q' to stop: ")
        if user_input.lower() == 'q':
            stop_event.set()

# Main function
def main():
    time_flag = [False]
    start_time = [None]
    data_to_save = []
    save_interval = 10 * 1  # 60 minutes in seconds
    last_save_time = time.time()

    # Create SensorConnection instance
    sensor = SensorConnection.get_instance(device_id=b'A56QZ86T', baudrate=1000000)

    try:
        # Find available FTDI devices
        available_devices = sensor.find_dongle()
        print("Available devices:", available_devices)

        # Connect to specific device (assuming description ID is 'STEEG_DG819202')
        connected = sensor.connect_sensor(descrip_id='STEEG_DG819202')
        if connected:
            print("Connected to the sensor.")
            
            # Send ON command
            command_queue.put(Command(Command.Type.ON))

            # Start user input thread
            input_thread = threading.Thread(target=user_input_thread)
            input_thread.start()
            # 调用函数清空文件并写入表头
            clear_and_initialize_sensor_data_file()

            # Read data
            while not stop_event.is_set():
                try:
                    received_data = data_queue.get(timeout=5)  # Receive data with timeout

                    elapsed_milliseconds, serial_number, emg = process_received_data(received_data, start_time, time_flag)
                    data_to_save.append([elapsed_milliseconds, serial_number] + emg)

                    # Check if it's time to save data
                    current_time = time.time()
                    if current_time - last_save_time >= save_interval:
                        save_data_to_txt(data_to_save)
                        data_to_save = []  # Clear saved data
                        last_save_time = current_time

                except queue.Empty:
                    print("No data received within timeout period.")
                    break
            
            # Close connection
            #sensor.close()
            print('Closing the sensor......')
            
            # Save remaining data to txt file
            if data_to_save:
                save_data_to_txt(data_to_save)
            
            input_thread.join()
            
        else:
            print("Failed to connect to the sensor.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping...")
        stop_event.set()

    finally:
        sensor.close()  # Ensure connection is closed

if __name__ == "__main__":
    main()

