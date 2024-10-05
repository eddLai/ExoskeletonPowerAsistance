# coding=utf8

from _struct import unpack, pack
from threading import Thread
import time
import ftd2xx
from cobs import cobs
import command_queue, data_queue, message_lock

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


#
# generator for command
#
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
            Command.Type.ON: lambda: 'ON',
            Command.Type.OFF: lambda: 'OFF',
            Command.Type.IMPEDANCE_AC: lambda: 'IMPA',
            Command.Type.IMPEDANCE_DC: lambda: 'IMPD',
            Command.Type.IMPEDANCE_AUTO: lambda: 'IMPAUTO',
            Command.Type.IMPEDANCE_OFF: lambda: 'IMPO',
            Command.Type.GET_MACHINE_INFO: lambda: 'MINFO',
            Command.Type.INPUT_AP_SSID_KEY: lambda: 'SSID',
            Command.Type.READ_SYNC_TICK: lambda: 'SYNCTICK',
            Command.Type.GET_CONN_STATUS: lambda: 'CONNSTATUS'
        }.get(command.type)()

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
        signal += chr(command.ssid.length())
        for c in command.ssid.split():
            signal += chr(c)
        signal += chr(command.key.length())
        for c in command.key.split():
            signal += chr(c)
        signal += command.ap_mode
        return signal


#
# sensor connection profile
#
class SensorConnection(object):
    connections = {}
    @classmethod
    def get_instance(cls, port, baudrate=1000000):  # TODO: (1)remove port (2)substituted by FTDI IDs
        if not cls.connections.get(port):
            cls.connections[port] = SensorConnection(port, baudrate)
        return cls.connections[port]

    def __init__(self, port, baudrate):
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
        if not descrip_id in self.dongle_list:
            print('Target ID ('+descrip_id+') is not exist.')
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

        # main reading thread
        while self.is_running:
            buffer_size = self.conn.getQueueStatus()
            # if buffer has new data
            if buffer_size > 0:                
                # get data from FTDI buffer
                message_lock.acquire()
                read_buffer = self.conn.read(buffer_size)
                message_lock.release()

                # analyze data and find the end byte of COBS packet
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
        command_queue.put(Command.OFF())
        time.sleep(0.8)
        self.is_running = False
        print('Closing the sensor......')

        # wait for finishing reading
        while not self.stop_reading:
            time.sleep(0.05)
            continue

        message_lock.acquire()
        self.conn.close()
        message_lock.release()
        self.dongle_status = None
        self.conn = None
        return self.dongle_status


#
# generator for eeg data
#
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
    
