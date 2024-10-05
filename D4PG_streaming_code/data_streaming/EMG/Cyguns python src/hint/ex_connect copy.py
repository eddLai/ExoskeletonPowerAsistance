import queue

import ftd2xx

# Initialize queues, lock, and stop event
command_queue = queue.Queue()

class Command(object):

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
    def impedance_off(cls):
        return cls._command_builder(7)


# sensor connection profile
class SensorConnection(object):

    connections = {}

    @classmethod
    def get_instance(cls, port, baudrate=1000000):
        if not cls.connections.get(port):
            cls.connections[port] = SensorConnection(port, baudrate)
        return cls.connections[port]

    def __init__(self, port, baudrate):
        self.dongle_status = None
        self.ftdi_list = None
        self.conn = None
        self.baudrate = baudrate

    def connect_sensor(self, descrip_id):
        self.conn = ftd2xx.openEx(str(descrip_id), flags=OPEN_BY_DESCRIPTION)  # open the FTDI device
        self.conn.setBaudRate(self.baudrate)
        self.conn.setDataCharacteristics(FT_WordLength_8, FT_StopBits_1, FT_Parity_None)
        self.conn.setTimeouts(10000, 10000)  # set (read, write) timeout in milliseconds
        self.conn.resetDevice()

        # 發送命令以啟動
        command_queue.put(Command.on())

    def disconnect_sensor(self):
        if self.conn:
            # 發送命令以關閉
            command_queue.put(Command.off())
            self.conn.close()
            self.conn = None
            print("Sensor disconnected")

    def send_command(self, command):
        if self.conn:
            command_str = command.encode()
            self.conn.write(command_str)
            print(f"Sent command: {command_str}")
