# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:23:59 2022

@author: Jeff Chang, ArtiseBiomedical Co., Ltd.
"""
import logging
import tkinter as tk
from tkinter import ttk

log = logging.getLogger(__name__)


class Tab_Impedance(tk.Frame):
    def __init__(self, root, parent, stream_controller):
        super().__init__(parent)
        self.root = root
        self.parent = parent
        self.stream = stream_controller
        self.status = "off" # "on" or "off"        
        
        # setup UI object                
        self.btn_on = ttk.Button(self, text="Impedance ON", command=self.imp_on)        
        self.btn_off = ttk.Button(self, text="Impedance OFF", command=self.imp_off)
        self.btn_off.state(["disabled"])
        self.table_fram = tk.Frame(self)
        self.table = Table(self.table_fram, (32,2), ['Channel', 'Impedance'])
         
        
        # setup UI layout       
        self.btn_on.grid(row=0, column=0, sticky='ew')
        self.btn_off.grid(row=0, column=1, sticky='ew')
        self.table_fram.grid(row=1, column=0, columnspan=2, sticky='ew')        
        
    
    def imp_on(self):        
        if self.stream.connect_status:           
            self.stream.send_setting_imp(True)
            self.status = "on"
            self.btn_on.state(["disabled"])
            self.btn_off.state(["!disabled"])
            self.update_imp()
        else:
            log.warning('Device is not connected.')
    
    def imp_off(self):
        self.stream.send_setting_imp(False)
        self.status = "off"
        self.btn_on.state(["!disabled"])
        self.btn_off.state(["disabled"])
        
    
    def update_imp(self):
        if self.status == "on":
            if len(self.stream.impedance_data) > 0:
                data_imp = self.stream.impedance_data.pop(0)[0]
                imp_table = [list(i) for i in zip(self.stream.ch_label, data_imp)]            
                self.table.set_data(imp_table)
            self.root.after(500, self.update_imp)
        
    
    
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
                    entry = tk.Entry(parent, bg='LightSteelBlue', fg='Black', font=('bold'))                    
                else:
                    entry = tk.Entry(parent, fg='blue')                
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
                
        
                
                
                
                