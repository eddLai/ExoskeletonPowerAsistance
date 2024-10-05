# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:23:59 2022

@author: Jeff Chang, ArtiseBiomedical Co., Ltd.
"""
import logging
import tkinter as tk
from tkinter import ttk

log = logging.getLogger(__name__)


class Tab_Spectrum(tk.Frame):
    def __init__(self, root, parent, stream_controller):
        super().__init__(parent)
        self.root = root
        self.parent = parent
        self.stream = stream_controller
        self.status = "off" # "on" or "off"        
        
        # setup UI object                
        self.btn_on = ttk.Button(self, text="Spectrum ON", command=self.spectrum_on)        
        self.btn_off = ttk.Button(self, text="Spectrum OFF", command=self.spectrum_off)
        self.btn_off.state(["disabled"])
        self.table_fram = tk.Frame(self)
        self.table = Table(self.table_fram, (32,62), ['Channel']+[str(hz/2) for hz in range(61)])
        # scrollbar_v = tk.Scrollbar(self, orient="vertical")
        # scrollbar_h = tk.Scrollbar(self, orient="horizontal")

         
        
        # setup UI layout       
        self.btn_on.grid(row=0, column=0, sticky='ew')
        self.btn_off.grid(row=0, column=1, sticky='ew')
        self.table_fram.grid(row=1, column=0, columnspan=20, sticky='ew')     
        # scrollbar_v.grid(row=1, column=1, sticky='w')
        # scrollbar_h.grid(row=2, column=0, columnspan=20, sticky='s')
        
    
    def spectrum_on(self):        
        if self.stream.connect_status:           
            self.stream.send_setting_FFT(True)
            self.status = "on"
            self.btn_on.state(["disabled"])
            self.btn_off.state(["!disabled"])
            self.update_spectrum()
        else:
            log.warning('Device is not connected.')
    
    def spectrum_off(self):
        self.stream.send_setting_FFT(False)
        self.status = "off"
        self.btn_on.state(["!disabled"])
        self.btn_off.state(["disabled"])
        
    
    def update_spectrum(self):
        if self.status == "on":
            if len(self.stream.spectrum_data) > 0:
                data_spectrum = self.stream.spectrum_data.pop(0)
                spectrum_table = [[i[0]]+i[1] for i in zip(self.stream.ch_label, data_spectrum)]          
                print('table len =', len(spectrum_table), len(spectrum_table[0]))
                self.table.set_data(spectrum_table)
            self.root.after(250, self.update_spectrum)
        
    
    
class Table:
    def __init__(self, parent, dim, header):
        # dim: table dimension (row, col)
        self.dim = dim
        n_row, n_col = dim
        null_table = [header]+[['']*n_col]*n_row
        self.entry_structure = list()
        
        for i in range(n_row+1):
            row_entry_list = list()
            for j in range(n_col):     
                if i ==0:                    
                    entry = tk.Entry(parent, bg='LightSteelBlue', fg='Black', font=('Calibri 8 bold'), width=5)                    
                else:
                    entry = tk.Entry(parent, fg='blue', font=('Calibri 8'), width=5)                
                entry.grid(row=i, column=j, sticky='ew')
                entry.insert(tk.END, null_table[i][j])
                row_entry_list.append(entry)
            if i > 0:
                self.entry_structure.append(row_entry_list)
    
    def set_data(self, table_list):
        n_row, n_col = self.dim
        n_row = len(table_list)
        for i in range(n_row):
            for j in range(n_col):
                self.entry_structure[i][j].delete(0,tk.END)
                self.entry_structure[i][j].insert(0,str(table_list[i][j]))
                
        
                
                
                
                