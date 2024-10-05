# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:35:04 2022

@author: Jeff Chang, ArtiseBiomedical Co., Ltd.
"""

import logging
import os
import time
from subprocess import Popen
import threading
import json
import websocket
from modules.api_cmd import ApiCmd


log = logging.getLogger(__name__)



class StreamController(ApiCmd):
    def __init__(self, parent, url="ws://localhost:31278/ws"):
        super().__init__(self)
        self.parent = parent
        self.kernel = Kernel_exe.run() 
        self.data_running = False
        self.raw_sample_rate = None
        self.ch_num = None
        self.ch_label = list()
        # self.API_FFT_isRun = False
        self.connect_status = False
        self.dongle_list = list()  
        self.ws_thread_run = True
        self.first_tick = None
        self.impedance_data = list()
        self.deci_data = list()
        self.spectrum_data = list()
        
        
        self.ws = websocket.WebSocketApp(url,
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        self.ws_thread = threading.Thread(target=self.on_connect)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        log.debug('Websockt thread start. ID = '+str(self.ws_thread.ident))
        
        
    def on_connect(self):    
        while self.ws_thread_run:
            self.ws.run_forever()
            log.warning('WS connection closed reconnect in 1 sec')
            time.sleep(1)        
        log.debug('End of WS client thread.')        
    
    
    def on_message(self, ws, message):
        """
        Handle data from socket and store it into different data
        structure
        """
        # print('Got WS Msg.', message)
        raw = json.loads(message)
        if raw["type"]["type"] == "response":
            if raw["type"]["source_name"] == "dongle":
                self.dongle_list = raw["contents"]["dongle_list"]
            elif raw["type"]["source_name"] == "system":
                pass
                # self.main_ui.API_message = raw["contents"]["message"]   #system message from API                
            elif raw["type"]["source_name"] == "device":                
                # self.main_ui.set_device_info('A8EEG', raw["contents"]["sampling_rate"], raw["contents"]["resolution"], raw["contents"]["battery"])
                self.raw_sample_rate = raw["contents"]["sampling_rate"]
                self.ch_num = raw["contents"]["ch_num"]
                self.ch_label = raw["contents"]["ch_label"]
                self.batt = raw["contents"]["battery"]
                self.data_running = raw["contents"]["data_running"]  
                
                # if raw["contents"]["ch_label"] is None or len(raw["contents"]["ch_label"]) == 0:
                #     for i in range(1, self.ch_num + 1):
                #         self.ch_label.append("Channel_{}".format(i))
       
                
        if raw["type"]["type"] == "ack":
            if raw["type"]["source_name"] == "dongle":
                self.connect_status = raw["contents"]["status"]           
        
        if raw["type"]["type"] == "data":
            if raw["type"]["source_name"] == "decimation":
                for data_in_chunk in raw["contents"]:
                    if self.first_tick is None:
                        self.first_tick = data_in_chunk["sync_tick"]
                    timestamp = (data_in_chunk["sync_tick"]-self.first_tick)/self.raw_sample_rate
                    self.deci_data.append((timestamp, data_in_chunk["data"]))
                    while len(self.deci_data) > 1000:
                        self.deci_data.pop(0)                            
            elif raw["type"]["source_name"] == "impedance":
                self.impedance_data.append(raw["contents"]["impedance"])
                while len(self.impedance_data) > 3:
                    self.impedance_data.pop(0)                    
            elif raw["type"]["source_name"] == "FFT":
                self.spectrum_data.append(raw["contents"]["psd"])
                while len(self.spectrum_data) > 10:
                    self.spectrum_data.pop(0)                  
                # if raw["type"]["type"] != "data":
                #     self.FFT_response.append(message)
                # else:                    
                #     self.FFT_data_msg.append(message)   
                
        
        
        
    def on_open(self, ws):
        """
        Handle websocket open
        """        
        self.send_find_dongle()
        
    def on_error(self, ws, error):
        """
        Handle websocket error
        """
        # log.error('WS error: ' + str(error))
        pass
        

    def on_close(self, ws, close_status_code, close_reason):
        """
        Handle websocket close event
        """
        # log.debug('WS close: ' + str(close_reason))
        self.impedance_data = list()
        self.deci_data = list()
        self.first_tick = None
        
        
    def close(self):
        self.send_connect_device(False)
        self.send_sys_shutdown()
        self.ws_thread_run = False
        # self.kernel.close()
    
    
    def is_connected(self):
        if self.ws is not None:
            if self.ws.sock is not None:
                try:
                    sock_conn = self.ws.sock.connected
                    return sock_conn
                except:
                    log.debug('websocket lost')                                        
        return False
        
        
        
        
    



class Kernel_exe():
    kernel_exe = None    
    @classmethod
    def run(cls):
        try:
            result = os.system("TASKKILL /F /IM CygnusKernel.exe")
            log.info('Watting for kernel.')
            cls.kernel_exe = Popen(['./core/CygnusKernel.exe'], cwd=r'./core/')
            log.info("Kernel's pid:" + str(cls.kernel_exe.pid))
        except Exception as e:
            log.error(str(e))
        return cls
    
    @classmethod
    def close(cls):
        if cls.kernel_exe:
            try:
                log.info("Kernel's pid:" + str(cls.kernel_exe.pid))
                cls.kernel_exe.kill()                
                log.info('Kernel is closed.')
            except Exception as e:
                log.error(str(e))
                
                
                
