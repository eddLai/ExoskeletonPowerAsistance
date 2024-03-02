import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from itertools import count

def get_motor_angle():
    return np.random.randint(0, 360)

class PlotAnimation:
    def __init__(self, data_source, window_width=10):
        self.data_source = data_source
        self.window_width = window_width
        self.fig, self.ax = plt.subplots()
        self.xdata, self.ydata = [], []
        self.ln, = self.ax.plot([], [], 'r-', animated=False)
        self.ani = FuncAnimation(self.fig, self.update, frames=count(), init_func=self.init, blit=False)
        
    def init(self):
        self.ax.set_xlim(0, self.window_width)
        self.ax.set_ylim(0, 360)  # 根据实际数据调整y轴范围
        return self.ln,
    
    def update(self, frame):
        self.xdata.append(frame)
        new_angle = self.data_source()  # 从传入的函数获取新的数据点
        self.ydata.append(new_angle)
        
        if len(self.xdata) > self.window_width:
            self.xdata.pop(0)
            self.ydata.pop(0)
        self.ln.set_data(self.xdata, self.ydata)
        self.ax.set_xlim(self.xdata[0], self.xdata[0] + self.window_width)
        return self.ln,
    
    def show(self):
        plt.show()

animation = PlotAnimation(data_source=get_motor_angle, window_width=10)
animation.show()