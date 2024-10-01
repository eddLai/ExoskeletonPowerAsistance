import keyboard
from matplotlib.animation import FuncAnimation
from tensorboardX import SummaryWriter
from data_streaming import Env, client_order
import time
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

class DynamicPlotter(QMainWindow):
    def __init__(self, timewindow=10.0):
        super().__init__()
        self.timewindow = timewindow
        self.sample_rate = 1000  # 設定樣本率為1000Hz
        self.maxsamples = int(timewindow * self.sample_rate)
        
        # 初始化數據緩衝區和時間軸
        self.databuffer = np.array([])
        self.timebuffer = np.array([])
        
        # 創建並設置圖表
        self.plotWidget = pg.PlotWidget()
        self.setCentralWidget(self.plotWidget)
        self.curve = self.plotWidget.plot(self.timebuffer, self.databuffer)
        
    def update_plot(self, new_data):
        # 計算新數據的時間戳
        if self.timebuffer.size == 0:
            new_timestamps = np.linspace(0, 0.05 - 1/self.sample_rate, len(new_data))
        else:
            new_timestamps = np.linspace(self.timebuffer[-1] + 1/self.sample_rate, self.timebuffer[-1] + 0.05, len(new_data))
        
        # 更新數據緩衝區和時間軸
        self.databuffer = np.append(self.databuffer, new_data)[-self.maxsamples:]
        self.timebuffer = np.append(self.timebuffer, new_timestamps)[-self.maxsamples:]
        
        # 更新圖表以顯示新數據
        self.curve.setData(self.timebuffer, self.databuffer)
        
        # 更新X軸的範圍以動態顯示最新的10秒數據
        if self.timebuffer[-1] < self.timewindow:
            self.plotWidget.setXRange(0, self.timewindow)
        else:
            self.plotWidget.setXRange(self.timebuffer[-1] - self.timewindow, self.timebuffer[-1])
        QApplication.processEvents()  # 强制处理累积的事件 

def main():
    try:
        writer = SummaryWriter("runs/recording_EXO_newEMG_plot")
        env = Env.ExoskeletonEnv(writer, device='cuda')
        app = QApplication(sys.argv)
        window = DynamicPlotter()
        # window.show()
        state = env.reset(window)
        print("reset")
        done = False
        while not done:
            if keyboard.is_pressed('q'):
                print("Exiting...")
                break
            # action1 = np.random.uniform(-1, 1)
            # action2 = np.random.uniform(-1, 1)
            action1 = 0
            action2 = 0
            state, reward, done, info = env.step([action1,action2],window)
            print("R_angle: ", state[0], "L_angle: ", state[3],"reward: ",reward)
            app.processEvents()
            time.sleep(0.001)
    finally:
        client_order.FREEX_CMD(env.sock, "E", "0", "E", "0")
        print("disconnect")
        env.log_writer.close()

if __name__ == "__main__":
    main()