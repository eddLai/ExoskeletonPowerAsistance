from client_order import *
import curses
import time
import pandas as pd
from datetime import datetime

def display_data_curses(stdscr, data, nowtime, key, input=None):
    if data:
        stdscr.clear()
        stdscr.addstr(1, 0, f"{nowtime}, ESC to exit, s to save and exit, TAB to change mode to enter")
        stdscr.addstr(2, 0, f"Motor 1 (Right) - Angle: {data[0]} deg, Speed: {data[1]} deg/s, Current: {data[2] * 0.01} A")
        stdscr.addstr(3, 0, f"Motor 2 (Left) - Angle: {data[3]} deg, Speed: {data[4]} deg/s, Current: {data[5] * 0.01} A")
        stdscr.addstr(4, 0, f"Roll: {data[6]} deg, Pitch: {data[7]} deg, Yaw: {data[8]} deg")
        stdscr.refresh()


def main_curses(stdscr):
    client_socket = connect_FREEX()
    if client_socket is None:
        stdscr.addstr(6, 0, "Failed to connect to FREEX.")
        stdscr.refresh()
        stdscr.getch()
        return
    
    columns = ['Timestamp', 'Motor 1 Angle', 'Motor 1 Speed', 'Motor 1 Current', 
               'Motor 2 Angle', 'Motor 2 Speed', 'Motor 2 Current', 
               'Roll', 'Pitch', 'Yaw']
    data_df = pd.DataFrame(columns=columns)
    start_date = datetime.now().strftime('%Y-%m-%d')
    start_time = time.time()

    stdscr.nodelay(True)
    key = None

    while key != 27 and key != 115:  # ESC key in ASCII
        key = stdscr.getch()
        data = get_INFO(client_socket)
        elapsed_time = time.time() - start_time
        if data is None:
            stdscr.addstr(6, 0, "Failed to get data from FREEX.")
            stdscr.refresh()
            continue
        else:
            result = analysis(data)
            current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            display_data_curses(stdscr, result, current_time, key)
            if result:
                new_row = pd.Series([current_time] + result, index=columns)
                data_df = data_df.append(new_row, ignore_index=True)

    client_socket.close()
    if key == 115:
        data_df.to_csv('output_data.csv', index=False)
        stdscr.addstr(7, 0, "Data saved to output_data.csv")
        stdscr.addstr(8, 0, f"Total elapsed time: {elapsed_time:.2f} seconds")
        stdscr.refresh()
        stdscr.getch()
    else:
        pass


def main():
    client_socket = connect_FREEX()
    if client_socket == None:
        print("fail socket")
        return
    
    key = None

    while (1):  # ESC key in ASCII
        data = get_INFO(client_socket)
        if data is None:
            print("Failed to get data")
        else:
            result = analysis(data)
            print(result)
            
    client_socket.close()
    
if __name__ == "__main__":
    curses.wrapper(main_curses)
    # print("program finished")
    # main()
