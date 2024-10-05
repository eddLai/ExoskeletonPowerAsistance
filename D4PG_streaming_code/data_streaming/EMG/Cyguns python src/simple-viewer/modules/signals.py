# -*- coding: utf-8 -*-
"""
Created on Mon Sep 1 15:35:36 2022

@author: Jeff Chang, ArtiseBiomedical Co., Ltd.
"""
import logging
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
style.use('fivethirtyeight')


import random
from itertools import count


log = logging.getLogger(__name__)


class Tab_Signals(tk.Frame):
    def __init__(self, root, parent, stream_controller):
        super().__init__(parent)
        self.root = root
        self.parent = parent
        self.stream = stream_controller
        self.status = "off" # "on" or "off"
        self.x_vals = []
        self.y_vals = []
        self.y_vals2 = []        
        self.index = count()
        
        # setup UI object                
        self.btn_on = ttk.Button(self, text="Signals ON", command=self.signals_on)        
        self.btn_off = ttk.Button(self, text="Signals OFF", command=self.signals_off)
        self.btn_off.state(["disabled"])
        self.plot_fram = tk.Frame(self)         
        
        # setup UI layout       
        self.btn_on.grid(row=0, column=0, sticky='ew')
        self.btn_off.grid(row=0, column=1, sticky='ew')
        self.plot_fram.grid(row=1, column=0, columnspan=10, sticky='ew')        
        
        # setup matplotlib object
        self.fig = Figure(figsize=(20,8), dpi=50)
        self.ax = self.fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(self.fig, master=self.plot_fram)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=200)
        self.ani.pause()
        
        # plot parameters
        self.scale = 300
        self.time_range = 10 # in sec
        self.deci_num = 4
        self.x0 = None
        

    def signals_on(self):
        if self.stream.connect_status: 
            self.stream.deci_data.clear()
            self.stream.send_setting_dec(True, self.deci_num)
            self.status = "on"            
            self.btn_on.state(["disabled"])
            self.btn_off.state(["!disabled"])
            self.ani.resume()            
        else:
            log.warning('Device is not connected.')
            
    
    def signals_off(self):
        self.ani.pause()
        self.stream.send_setting_dec(False, self.deci_num)
        self.status = "off"        
        self.btn_on.state(["!disabled"])
        self.btn_off.state(["disabled"])
        self.clear_buffer()
        
    
    def animate(self, i):     
        ax = self.fig.get_axes()
        if self.status == "on":
            if self.stream.ch_num:      
                yticks = [-1*self.scale*i for i in range(self.stream.ch_num)]
                update_plot = False
                while len(self.stream.deci_data) > 0:
                    timestamp, data = self.stream.deci_data.pop(0)
                    self.x_vals.append(timestamp)
                    self.y_vals.append(np.array(data)+np.array(yticks))
                    update_plot = True
                    if self.x0 is None:
                        self.x0 = timestamp                           
                    if timestamp - self.x0 >= self.time_range:
                        break
                if self.x0 is not None and update_plot:
                    self.ax.cla()
                    self.ax.plot(self.x_vals, np.array(self.y_vals))                     
                    self.ax.set_yticks(yticks)
                    self.ax.set_yticklabels(self.stream.ch_label)
                    self.ax.set_ylim([yticks[-1]-self.scale, yticks[0]+self.scale])
                    self.ax.set_xlim([np.floor(self.x0), self.x0+self.time_range])  
                    if timestamp - self.x0 >= self.time_range:
                        self.clear_buffer()
        return ax
    
    
    def clear_buffer(self):
        self.x0 = None
        self.x_vals.clear()
        self.y_vals.clear()
    
    
    
    