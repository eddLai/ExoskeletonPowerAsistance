import cv2
import os
import glob
import numpy as np

# 設定基礎路徑和檔案來源
base_path = 'simulation/mocap_EMG_EEG_data/data_An_Yu/path1_09'

# 檢查視頻文件路徑
video_dir = os.path.join(base_path, 'videos')
print(f"Looking for video files in: {video_dir}")
video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
if len(video_files) == 0:
    raise FileNotFoundError(f"No video files found in {video_dir}")

video_path = video_files[0]
print(f"Found video file: {video_path}")

# 打開視頻檔案並獲取總幀數
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames: ", total_frames)

# 初始化參數
brightness_values = []
red_values = []
green_values = []
frame_number = 0
max_offset = 30
enable_movement = False

# 變數
lamp_x, lamp_y = None, None
half_size = 15

# 滑鼠事件處理函數，選擇燈的位置
def mouse_callback(event, x, y, flags, param):
    global lamp_x, lamp_y, frame_copy
    if event == cv2.EVENT_MOUSEMOVE:  # 當滑鼠移動時，更新動態矩形框
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (x - half_size, y - half_size), (x + half_size, y + half_size), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONDOWN:  # 當滑鼠左鍵按下時
        lamp_x, lamp_y = x, y
        print(f'Selected point: ({lamp_x}, {lamp_y})')

# 選擇幀進行手動燈標記
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()
frame_copy = frame.copy()  # 用於動態顯示滑鼠移動的矩形框

cv2.namedWindow('Select Point', cv2.WINDOW_NORMAL)  # 設置窗口可調整大小
cv2.resizeWindow('Select Point', 640*2, 480*2)  # 將窗口大小設置為 1280x960

cv2.setMouseCallback('Select Point', mouse_callback)

# 顯示圖像並等待滑鼠選擇
while True:
    cv2.imshow('Select Point', frame_copy)
    if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC鍵退出
        break
    if lamp_x is not None and lamp_y is not None:  # 如果有選擇點，則退出循環
        break

cv2.destroyAllWindows()

# 若未選擇點，則退出程式
if lamp_x is None or lamp_y is None:
    print("No point selected. Exiting.")
    cap.release()
    exit()

# 開始處理視頻，追蹤燈的位置和計算亮度、顏色
initial_lamp_x = lamp_x
initial_lamp_y = lamp_y
THRESHOLD = np.mean(cv2.cvtColor(frame[max(0, lamp_y - half_size):min(frame.shape[0], lamp_y + half_size),
                                        max(0, lamp_x - half_size):min(frame.shape[1], lamp_x + half_size)], 
                                  cv2.COLOR_BGR2GRAY))

# 重置視頻捕捉到開始位置
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 設置窗口來顯示處理過程中的幀
cv2.namedWindow("Original Frame with Cropped Area", cv2.WINDOW_NORMAL)  # 設置窗口大小可以調整
cv2.resizeWindow("Original Frame with Cropped Area", 1280, 960)  # 調整窗口大小為1280x960

# 循環處理每一幀
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    y_start = max(0, lamp_y - half_size)
    y_end = min(frame.shape[0], lamp_y + half_size)
    x_start = max(0, lamp_x - half_size)
    x_end = min(frame.shape[1], lamp_x + half_size)
    brightness_frame = gray_frame[y_start:y_end, x_start:x_end]

    # 確定燈位置和亮度
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(brightness_frame)
    if max_val > THRESHOLD and enable_movement:
        new_lamp_x = x_start + max_loc[0]
        new_lamp_y = y_start + max_loc[1]
        if abs(new_lamp_x - initial_lamp_x) > max_offset or abs(new_lamp_y - initial_lamp_y) > max_offset:
            print(f"Error: Lamp moved too far at frame {frame_number}")
            break
        else:
            lamp_x = new_lamp_x
            lamp_y = new_lamp_y
        
    avg_brightness = np.mean(brightness_frame)
    brightness_values.append(avg_brightness)

    red_channel = frame[y_start:y_end, x_start:x_end, 2]
    green_channel = frame[y_start:y_end, x_start:x_end, 1]
    avg_red = np.mean(red_channel)
    avg_green = np.mean(green_channel)
    red_values.append(avg_red)
    green_values.append(avg_green)

    # 在原始影像上繪製燈的裁剪框
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    
    # 顯示處理過程中的原始圖像和裁剪的區域
    cv2.imshow("Original Frame with Cropped Area", frame)

    # 檢查是否按下 'q' 鍵以退出
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    # 每隔 100 幀打印進度
    if frame_number % 100 == 0:
        print(f"Processed frame {frame_number}/{total_frames}")

    frame_number += 1

cap.release()
cv2.destroyAllWindows()
