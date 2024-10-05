# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:54:57 2022

@author: Jeff Chang, ArtiseBiomedical Co., Ltd.
"""

import json
import logging
log = logging.getLogger(__name__)

class ApiCmd(object):
    def __init__(self, stream_ctrl):
        self.stream_ctrl = stream_ctrl       
        
        
    def send_find_dongle(self):
            device_request_msg = json.dumps({
                "type": {
                    "type": "request",
                    "target_type": "device",
                    "target_name": "dongle"
                },
                "contents": "find_dongle"
            })        
            self.write_ws(device_request_msg)
            
            
    def send_connect_device(self, operation, dongle_id='null'):
        device_request_msg = json.dumps({
            "type": {
                "type": "setting",
                "target_type": "device",
                "target_name": "dongle"
            },
            "contents": {
                "on_off": operation,
                "target_id": dongle_id,
                "ch_config": 0
            }
        })    
        self.write_ws(device_request_msg)
    
    
    def send_sys_shutdown(self):
        device_request_msg = json.dumps({
            "type": {
                "type": "setting",
                "target_type": "system",
                "target_name": "system"
            },
            "contents": {
                "shut_down": True
                }
        })    
        self.write_ws(device_request_msg)
        
        
    def send_info_req(self):
        device_request_msg = json.dumps({
            "type": {
                "type": "request",
                "target_type": "device",
                "target_name": "device"
            },
            "contents": "device_info"
        })    
        self.write_ws(device_request_msg)
       
        
    def send_setting_raw(self, operation):
        raw_setting_msg = json.dumps({
            "type": {
                "type": "setting",
                "target_type": "raw",
                "target_name": "raw"
            },
            "contents": {
                "enable": operation,
                "chunk_size": 50     # should not less than 50.
            }
        })    
        self.write_ws(raw_setting_msg)            
    
    
    def send_setting_dec(self, operation, deci_num):   
        dec_setting_msg = json.dumps({
            "type": {
                "type": "setting",
                "target_type": "algorithm",
                "target_name": "decimation"
            },
            "contents": {
                "enable": operation,
                "use_clean_data": True,
                "decimate_num": deci_num,
                "notch_filter":  60 # 0, 50Hz, 60Hz                
            }
        })    
        self.write_ws(dec_setting_msg) 
    
    
    def send_setting_imp(self, operation):
        imp_setting_msg = json.dumps({
            "type": {
                "type": "setting",
                "target_type": "device",
                "target_name": "impedance"
            },
            "contents": {
                "enable": operation
            }
        })
        self.write_ws(imp_setting_msg)
        
    
    def send_setting_FFT(self, operation):
        FFT_setting_msg = json.dumps({
            "type": {
                "type": "setting",
                "target_type": "algorithm",
                "target_name": "FFT"
            },
            "contents": {
                "enable": operation,
                "window_size": 2,
                "window_interval": 1,
                "freq_range": [0, 30],
                "notch_filter": 60 # 0, 50Hz, 60Hz
            }
        })    
        self.write_ws(FFT_setting_msg) 
        self.stream_ctrl.API_FFT_isRun = operation
        
    
    def send_request_raw(self):
        raw_request_msg = json.dumps({
            "type": {
                "type": "request",
                "target_type": "raw",
                "target_name": "raw"
            },
            "contents": {
                "requirement": [
                    "enable",
                    "sps_origin",
                    "ch_num",
                    "chunk_size",
                    "ch_label"
                ]
            }
        })    
        self.write_ws(raw_request_msg)
        
    
    def send_request_dec(self):
        dec_request_msg = json.dumps({
            "type": {
                "type": "request",
                "target_type": "algorithm",
                "target_name": "decimation"
            },
            "contents": {
                "requirement": [
                    "enable",
                    "sps_origin",
                    "sps_decimated",
                    "decimate_num",
                    "ch_num",
                    "ch_label"
                ]
            }
        })    
        self.write_ws(dec_request_msg)
        
            
    def send_request_imp(self):
        imp_request_msg = json.dumps({
            "type": {
                "type": "request",
                "target_type": "device",
                "target_name": "impedance"
            },
            "contents": {               
                "requirement": [
                        "enable",
                        "sps_origin",
                        "ch_num",
                        "ch_label"
                ]
            }
        })    
        self.write_ws(imp_request_msg)        
        
    
    def send_request_FFT(self):
        FFT_request_msg = json.dumps({
            "type": {
                "type": "request",
                "target_type": "algorithm",
                "target_name": "FFT"
            },
            "contents": {
                "requirement": [
                    "enable",
                    "sps_origin",
                    "window_size",
                    "window_interval",
                    "freq_range",
                    "freq_label",
                    "data_size",
                    "ch_label"
                ]
            }
        })    
        self.write_ws(FFT_request_msg)
        

    def write_ws(self, cmd):
        if self.stream_ctrl.is_connected():
            self.stream_ctrl.ws.send(cmd)
        else:
            log.error('API dis-connected.')
        
        
        