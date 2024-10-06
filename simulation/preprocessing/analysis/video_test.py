import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

class Video:
    def __init__(self, base_path, file_type='*.mp4', sep='\t'):
        self.base_path = base_path
        self.file_type = file_type
        self.sep = sep
        self.video_file = self._load_video_file()
        self.lamp_x, self.lamp_y = 747, 465
        self.half_size = 15
        self.frame_rgb = None
        self.cropped_frame_rgb = None
        self.avg_brightness = None
        self.THRESHOLD = None
        self.brightness_values = None
        self.red_values = None
        self.green_values = None
        self.highest_plateau_value = None
        self.max_slope_frame = None

    def _load_video_file(self):
        """
        :return: all avaliable video files
        """
        video_files = glob.glob(os.path.join(self.base_path, 'videos', self.file_type))
        if not video_files:
            raise FileNotFoundError(f"at {os.path.join(self.base_path, 'videos')} found no video mp4 。")
        return video_files
    
    def get_total_frames(self):
        cap = cv2.VideoCapture(self.video_files[0])
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("total frames: ", total_frames)
        cap.release()

    def find_sync_marker(self):
        cap = cv2.VideoCapture(self.video_files[0])
        frame_number = 578
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x_start = max(0, self.lamp_x - self.half_size)
        x_end = min(frame.shape[1], self.lamp_x + self.half_size)
        y_start = max(0, self.lamp_y - self.half_size)
        y_end = min(frame.shape[0], self.lamp_y + self.half_size)
        cropped_frame = frame[y_start:y_end, x_start:x_end]
        self.cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        cropped_frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        self.avg_brightness = np.mean(cropped_frame_gray)
        self.THRESHOLD = self.avg_brightness
        cropped_lamp_x = self.lamp_x - x_start
        cropped_lamp_y = self.lamp_y - y_start
        cv2.circle(self.cropped_frame_rgb, (cropped_lamp_x, cropped_lamp_y), radius=5, color=(255, 0, 0), thickness=1)


    def plot_sync_marker(self):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})
        axs[0].imshow(self.frame_rgb)
        axs[0].set_title('Original Frame')
        axs[0].axis('off')
        axs[1].imshow(self.cropped_frame_rgb)
        axs[1].set_title(f'Crop around ({self.lamp_x}, {self.lamp_y})')
        axs[1].axis('off')

        axs[1].text(0.5, 0.1, f'Avg Brightness: {self.avg_brightness:.2f}', color='white',
                    fontsize=12, ha='center', va='center', transform=axs[1].transAxes)
        plt.subplots_adjust(wspace=0.02)
        plt.show()

    def find_the_light(self):
        self.brightness_values = []
        red_values = []
        green_values = []
        frame_number = 0
        max_offset = 30
        initial_lamp_x = self.lamp_x
        initial_lamp_y = self.lamp_y
        enable_movement = False
        enable_imageShow = True
        cap = cv2.VideoCapture(self.video_files[0])
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            exit()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if frame is None:
                    print(f"End of video at frame {frame_number}")
                else:
                    print(f"Error reading the frame at frame {frame_number}")
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            y_start = max(0, lamp_y - self.half_size)
            y_end = min(frame.shape[0], lamp_y + self.half_size)
            x_start = max(0, lamp_x - self.half_size)
            x_end = min(frame.shape[1], lamp_x + self.half_size)
            brightness_frame = gray_frame[y_start:y_end, x_start:x_end]

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(brightness_frame)
            if max_val > self.THRESHOLD and enable_movement:
                new_lamp_x = x_start + max_loc[0]
                new_lamp_y = y_start + max_loc[1]

                if abs(new_lamp_x - initial_lamp_x) > max_offset or abs(new_lamp_y - initial_lamp_y) > max_offset:
                    print(f"Error: Lamp moved too far at frame {frame_number}")
                    break
                else:
                    lamp_x = new_lamp_x
                    lamp_y = new_lamp_y
                    # print(f"Lamp moved to ({lamp_x}, {lamp_y}) at frame {frame_number}")
                
            avg_brightness = np.mean(brightness_frame)
            self.brightness_values.append(avg_brightness)

            red_channel = frame[y_start:y_end, x_start:x_end, 2]
            green_channel = frame[y_start:y_end, x_start:x_end, 1]
            avg_red = np.mean(red_channel)
            avg_green = np.mean(green_channel)
            red_values.append(avg_red)
            green_values.append(avg_green)

            cropped_frame = frame[y_start:y_end, x_start:x_end]
            cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            cropped_lamp_x = max_loc[0]
            cropped_lamp_y = max_loc[1]
            cv2.circle(cropped_frame_rgb, (cropped_lamp_x, cropped_lamp_y), 5, (255, 0, 0), 2)
            if enable_imageShow:
                cv2.imshow("Cropped Frame with Lamp Detected", cropped_frame_rgb)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        frame_number += 1

        cap.release()
        cv2.destroyAllWindows()

    def plot_brightness_over_time(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.brightness_values, label='Average Brightness')
        plt.plot(self.red_values, label='Red Channel Brightness', color='red')
        plt.plot(self.green_values, label='Green Channel Brightness', color='green')
        plt.axhline(y=self.THRESHOLD, color='r', linestyle='--', label=f'Brightness Threshold: {self.THRESHOLD}')
        plt.xlabel('Frame Number')
        plt.ylabel('Brightness')
        plt.title('Brightness over Time in Selected Region')
        plt.legend()
        plt.show()

    def find_start_time(self):
        plateau_threshold = 5
        slope_threshold = 5

        window_size = 10
        smoothed_green = np.convolve(self.green_values, np.ones(window_size)/window_size, mode='valid')

        plateau_mask = np.abs(np.diff(smoothed_green)) < plateau_threshold
        plateau_regions = np.where(plateau_mask)[0] + 1

        if len(plateau_regions) > 0:
            highest_plateau_idx = np.argmax(smoothed_green[plateau_regions])
            highest_plateau_start = plateau_regions[highest_plateau_idx]
            self.highest_plateau_value = smoothed_green[highest_plateau_start]
            print(f"Highest plateau starts at frame {highest_plateau_start} with value {self.highest_plateau_value}")
        else:
            print("No plateau found")

        green_slope = np.diff(self.green_values)

        slope_region_start = max(0, highest_plateau_start - window_size)
        slope_region_end = min(len(green_slope), highest_plateau_start + window_size)

        max_slope_idx = np.argmax(np.abs(green_slope[slope_region_start:slope_region_end]))
        self.max_slope_frame = slope_region_start + max_slope_idx

        print(f"Maximum slope change near highest plateau at frame {self.max_slope_frame}")

    def plot_start_time(self):
        plt.figure(figsize = (12, 6))
        plt.plot(self.green_values, color='green', label='Green Channel Brightness')
        plt.axhline(y = self.highest_plateau_value, color='orange', linestyle='--', label='Highest Plateau')
        plt.scatter(self.max_slope_frame, self.green_values[self.max_slope_frame], color='red', label='Max Slope Change', zorder=5)
        plt.xlabel('Frame Number')
        plt.ylabel('Brightness')
        plt.legend()
        plt.show()