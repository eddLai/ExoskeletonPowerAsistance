import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from itertools import count
import queue
from multiprocessing import Process, Queue, Event
from tkinter import Tk, Button

def create_exit_button():
    def on_exit():
        print("Setting exit event")
        exit_event.set()
    root = Tk()
    root.title("Stop the animation")
    btn = Button(root, text="Exiting", command=on_exit)
    btn.pack(pady=20)
    root.mainloop()

class PlotMotorAnimation:
    def __init__(self, data_source, window_width=10):
        self.data_source = data_source
        self.window_width = window_width
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 8))
        self.current_step = []
        self.data = [[] for _ in range(6)]
        self.lines = [
            self.axs[0].plot([], [], label='Motor 1 Angle')[0],
            self.axs[0].plot([], [], label='Motor 2 Angle')[0],
            self.axs[1].plot([], [], label='Motor 1 Angular Velocity')[0],
            self.axs[1].plot([], [], label='Motor 2 Angular Velocity')[0],
            self.axs[2].plot([], [], label='Motor 1 Angular Acceleration')[0],
            self.axs[2].plot([], [], label='Motor 2 Angular Acceleration')[0],
        ]
        self.y_lims = [(-360, 360), (-360, 360), (-360, 360)]
        for ax in self.axs:
            ax.legend()
        self.ani = FuncAnimation(self.fig, self.update, init_func=self.init, frames=count(), blit=False)
        
    def init(self):
        for ax in self.axs:
            ax.set_xlim(0, self.window_width)
        
        self.axs[0].set_ylim(self.y_lims[0])
        self.axs[1].set_ylim(self.y_lims[1])
        self.axs[2].set_ylim(self.y_lims[2])  
        return self.lines
    
    def update(self, frame):
        if exit_event.is_set():  # Check the flag status
            self.ani.event_source.stop()
            plt.close(self.fig)
            return self.lines
        try:
            new_data = self.data_source.get_nowait()  # Change to non-blocking get
            if new_data is None:  # Check for the exit signal
                self.running = False
                plt.close(self.fig)
                return self.lines
        except queue.Empty:
            return self.lines

        self.current_step.append(frame)
        if len(self.current_step) > self.window_width:
            self.current_step.pop(0)
            for data_list in self.data:
                data_list.pop(0)
        for i, line in enumerate(self.lines):
            self.data[i].append(new_data[i])
            line.set_data(self.current_step, self.data[i])
        for i in  range(3):
            self.axs[i].set_xlim(self.current_step[0], self.current_step[0] + self.window_width)
        return self.lines

    def show(self):
        plt.show()

class PlotIMUAnimation:
    def __init__(self, data_source, window_width=10):
        self.data_source = data_source
        self.window_width = window_width
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 8))
        self.current_step = []
        self.data = [[] for _ in range(3)]
        self.lines = [
            self.axs[0].plot([], [], label='X')[0],
            self.axs[1].plot([], [], label='Y')[0],
            self.axs[2].plot([], [], label='Z')[0],
        ]
        self.y_lims = [(-180, 180), (-180, 180), (-180, 180)]
        for ax in self.axs:
            ax.legend()
        self.ani = FuncAnimation(self.fig, self.update, init_func=self.init, frames=count(), blit=False)
        
    def init(self):
        for ax in self.axs:
            ax.set_xlim(0, self.window_width)
        
        self.axs[0].set_ylim(self.y_lims[0])
        self.axs[1].set_ylim(self.y_lims[1])
        self.axs[2].set_ylim(self.y_lims[2])
        return self.lines
    
    def update(self, frame):
        if exit_event.is_set():
            self.ani.event_source.stop()
            plt.close(self.fig)
            return self.lines
        try:
            new_data = self.data_source.get_nowait()
            if new_data is None:
                self.running = False
                plt.close(self.fig)
                return self.lines
        except queue.Empty:
            return self.lines
        
        self.current_step.append(frame)
        if len(self.current_step) > self.window_width:
            self.current_step.pop(0)
            for data_list in self.data:
                data_list.pop(0)
        for i, line in enumerate(self.lines):
            self.data[i].append(new_data[i])
            line.set_data(self.current_step, self.data[i])
        for i in  range(3):
            self.axs[i].set_xlim(self.current_step[0], self.current_step[0] + self.window_width)
        return self.lines
    
    def show(self):
        plt.show()

def get_data():
    return np.random.randint(0, 360, 9)

def run_motor_animation(data_queue):
    motor_animation = PlotMotorAnimation(data_queue, window_width=10)
    motor_animation.show()

def run_imu_animation(data_queue):
    imu_animation = PlotIMUAnimation(data_queue, window_width=10)
    imu_animation.show()

exit_event = Event()

if __name__ == '__main__':
    exit_button_process = Process(target=create_exit_button)
    exit_button_process.start()
    motor_data_queue = Queue()
    imu_data_queue = Queue()
    motor_plot_process = Process(target=run_motor_animation, args=(motor_data_queue,))
    imu_plot_process = Process(target=run_imu_animation, args=(imu_data_queue,))
    motor_plot_process.start()
    imu_plot_process.start()

    try:
        while not exit_event.is_set():
            motor_data = get_data()[:6]
            imu_data = get_data()[6:]
            motor_data_queue.put(motor_data)
            imu_data_queue.put(imu_data)

    finally:
        motor_plot_process.join()
        imu_plot_process.join()
        exit_button_process.terminate()
        print("All processes have been terminated.")


class PlotEMGAnimation:
    def __init__(self, data_source, window_width=10):
        self.data_source = data_source
        self.window_width = window_width
        self.fig, self.ax = plt.subplots()
        self.xdata, self.ydata = [], []
        self.ln, = self.ax.plot([], [], 'r-', animated=False)
        self.ani = FuncAnimation(self.fig, self.update, frames=count(), init_func=self.init, blit=False)
        
    def init(self):
        self.ax.set_xlim(0, self.window_width)
        self.ax.set_ylim(0, 10)
        return self.ln,
    
    def update(self, frame):
        self.xdata.append(frame)
        new_data = self.data_source
        self.ydata.append(new_data)
        
        if len(self.xdata) > self.window_width:
            self.xdata.pop(0)
            self.ydata.pop(0)
        self.ln.set_data(self.xdata, self.ydata)
        self.ax.set_xlim(self.xdata[0], self.xdata[0] + self.window_width)
        return self.ln,
    
    def show(self):
        plt.show()
