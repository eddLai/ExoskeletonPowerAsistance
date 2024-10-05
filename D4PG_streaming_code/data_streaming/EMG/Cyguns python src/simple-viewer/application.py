# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:18:20 2022

@author: Jeff Chang, ArtiseBiomedical Co., Ltd.
"""

import sys, time
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from modules import stream, connection, impedance, signals, spectrum


_app_ver = 'v2022.0'


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger(__name__)

log.info('App init......'+_app_ver)

root = tk.Tk()
root.withdraw()
root.wm_title("EEG Viewer")
root.geometry("1000x500")

notebook = ttk.Notebook(root)

eeg_stream = stream.StreamController(root)


tab1 = connection.Tab_Connection(root, notebook, eeg_stream)
tab2 = impedance.Tab_Impedance(root, notebook, eeg_stream)
tab3 = signals.Tab_Signals(root, notebook, eeg_stream)
tab4 = spectrum.Tab_Spectrum(root, notebook, eeg_stream)

notebook.add(tab1, text='Connection')
notebook.add(tab2, text='Impedance')
notebook.add(tab3, text='Signals')
notebook.add(tab4, text='Spectrum')
notebook.pack(expand=True, fill='both')


def on_closing():
    log.info('User close main win.')
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        eeg_stream.close()
        root.destroy()
        time.sleep(3)
        

def wait_kernel_ready():
    log.debug('Wait for kernel...')
    if not eeg_stream.is_connected():
        root.after(500, wait_kernel_ready)
    else:
        log.info('App init done.')
        root.deiconify() # show the app window
        

wait_kernel_ready()
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()

log.info('App closed.')

