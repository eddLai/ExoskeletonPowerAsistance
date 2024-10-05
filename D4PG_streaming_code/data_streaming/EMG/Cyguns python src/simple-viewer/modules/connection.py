# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:13:13 2022

@author: Jeff Chang, ArtiseBiomedical Co., Ltd.
"""

import logging
import tkinter as tk
from tkinter import ttk

log = logging.getLogger(__name__)


class Tab_Connection(tk.Frame):
    def __init__(self, root, parent, stream_controller):
        super().__init__(parent)
        self.root = root
        self.parent = parent
        self.stream = stream_controller
        self.status = "disconn" # "conn" or "disconn"
        self.device_lsit = list()
        
        # setup UI object
        self.label_device = ttk.Label(self, text="Device : ")      
        self.label_info = ttk.Label(self, text="Device's Info. : ")
        self.label_info_text = ttk.Label(self, text="")   
        self.box_device_list = ttk.Combobox(self, value = self.device_lsit, state='readonly')        
        self.btn_scan = ttk.Button(self, text="Scan", command=self.scan)        
        self.btn_connect = ttk.Button(self, text="Connect", command=self.connect)        
        self.btn_disconn = ttk.Button(self, text="Disconnect", command=self.disconnect)
        self.btn_disconn.state(["disabled"])
        
        # setup UI layout
        self.label_device.grid(row=0, column=0, sticky='e')
        self.label_info.grid(row=2, column=0, sticky='e')
        self.label_info_text.grid(row=2, column=1, columnspan=3, sticky='w')
        self.box_device_list.grid(row=0, column=1, columnspan=3, sticky='ew' )
        self.btn_scan.grid(row=1, column=1, sticky='ew')
        self.btn_connect.grid(row=1, column=2, sticky='ew')
        self.btn_disconn.grid(row=1, column=3, sticky='ew')        
        
        
    def scan(self):
        self.stream.send_find_dongle()
        # self.box_device_list.set(self.stream.dongle_list)
        self.box_device_list['values'] = self.stream.dongle_list
        
    
    def connect(self):        
        dongle_name = self.stream.dongle_list[self.box_device_list.current()]        
        self.stream.send_connect_device(True, dongle_name)  
        self.process_connect()
        self.status = "conn"        
        log.info('Connet to '+ str(dongle_name))
            

    def disconnect(self):
        self.stream.send_connect_device(False)
        self.btn_scan.state(["!disabled"])
        self.btn_connect.state(["!disabled"])
        self.btn_disconn.state(["disabled"])
        log.info('Disconnct pressed.')
        self.status = "disconn"
        
    
    def process_connect(self):
        if not self.stream.is_connected():
            log.warning('Kernel-API is not ready.')
            return
        if self.stream.connect_status:
            self.btn_scan.state(["disabled"])
            self.btn_connect.state(["disabled"])
            self.btn_disconn.state(["!disabled"])
            self.stream.send_info_req()
            self.root.after(250, self.get_info)
        else:
            self.root.after(100, self.process_connect)       
        
    
    def get_info(self):        
        if self.stream.raw_sample_rate:
            info_text = str(self.stream.raw_sample_rate) + " Hz, Battery: "+ str(self.stream.batt)
            self.label_info_text.config(text = info_text)
        else:
            self.label_info_text.config(text = '')
        
        if self.status == "conn":
            self.stream.send_info_req()
            self.root.after(1500, self.get_info)
        elif self.status == "disconn":
            self.label_info_text.config(text = '')
        
        
    
    