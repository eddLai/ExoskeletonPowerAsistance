import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

class VideoAnalyzer:
    def __init__(self, base_path, file_type='*.mp4', sep='\t', lamp_position=(747, 465)):
        """
        Initialize the VideoAnalyzer class for loading video and analyzing brightness and light detection.
        :param base_path: Base path for video files
        :param file_type: Type of video files to load
        :param sep: Separator for text outputs
        :param lamp_position: Tuple containing the initial position of the lamp (x, y)
        """
        self.base_path = base_path
        self.file_type = file_type
        self.sep = sep
        self.video_files = self._load_video_files()
        self.lamp_x, self.lamp_y = lamp_position  # Position of the lamp
        self.half_size = 15
        self.brightness_values = []
        self.red_values = []
        self.green_values = []
        self.THRESHOLD = None
        self.avg_brightness = None
        self.frame_rgb = None
        self.cropped_frame_rgb = None
        self.highest_plateau_value = None
        self.max_slope_frame = None

    def _load_video_files(self):
        """
        Internal function: Load video files.
        :return: List of available video files
        """
        video_files = glob.glob(os.path.join(self.base_path, 'videos', self.file_type))
        if not video_files:
            raise FileNotFoundError(f"No video files found at {os.path.join(self.base_path, 'videos')}")
        return video_files

    def get_total_frames(self):
        """
        Get the total number of frames in the video.
        """
        cap = cv2.VideoCapture(self.video_files[0])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total frames in video: ", total_frames)
        cap.release()

    def find_sync_marker(self, frame_number=578):
        """
        Find the synchronization light in the specified frame.
        :param frame_number: Frame number to analyze
        """
        cap = cv2.VideoCapture(self.video_files[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Unable to read frame from video")
            return
        
        # Convert to RGB format
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Crop region around the fixed lamp position
        x_start = max(0, self.lamp_x - self.half_size)
        x_end = min(frame.shape[1], self.lamp_x + self.half_size)
        y_start = max(0, self.lamp_y - self.half_size)
        y_end = min(frame.shape[0], self.lamp_y + self.half_size)
        
        cropped_frame = frame[y_start:y_end, x_start:x_end]
        self.cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        cropped_frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        self.avg_brightness = np.mean(cropped_frame_gray)
        self.THRESHOLD = self.avg_brightness
        
        # Draw a small circle on the lamp position
        cropped_lamp_x = self.lamp_x - x_start
        cropped_lamp_y = self.lamp_y - y_start
        cv2.circle(self.cropped_frame_rgb, (cropped_lamp_x, cropped_lamp_y), radius=5, color=(255, 0, 0), thickness=1)

    def plot_sync_marker(self):
        """
        Plot the original frame and the cropped frame around the sync light.
        """
        if self.frame_rgb is None or self.cropped_frame_rgb is None:
            print("Sync marker not found. Please run find_sync_marker() first.")
            return
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})
        axs[0].imshow(self.frame_rgb)
        axs[0].set_title('Original Frame')
        axs[0].axis('off')
        axs[1].imshow(self.cropped_frame_rgb)
        axs[1].set_title(f'Cropped Region ({self.lamp_x}, {self.lamp_y})')
        axs[1].axis('off')

        axs[1].text(0.5, 0.1, f'Avg Brightness: {self.avg_brightness:.2f}', color='white',
                    fontsize=12, ha='center', va='center', transform=axs[1].transAxes)
        plt.subplots_adjust(wspace=0.02)
        plt.show()

    def find_the_light(self):
        """
        Track the brightness of the light throughout the video.
        """
        cap = cv2.VideoCapture(self.video_files[0])
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            y_start = max(0, self.lamp_y - self.half_size)
            y_end = min(frame.shape[0], self.lamp_y + self.half_size)
            x_start = max(0, self.lamp_x - self.half_size)
            x_end = min(frame.shape[1], self.lamp_x + self.half_size)
            brightness_frame = gray_frame[y_start:y_end, x_start:x_end]

            avg_brightness = np.mean(brightness_frame)
            self.brightness_values.append(avg_brightness)

            # Red and green channel brightness
            red_channel = frame[y_start:y_end, x_start:x_end, 2]
            green_channel = frame[y_start:y_end, x_start:x_end, 1]
            avg_red = np.mean(red_channel)
            avg_green = np.mean(green_channel)
            self.red_values.append(avg_red)
            self.green_values.append(avg_green)

            frame_number += 1

        cap.release()

    def plot_brightness_over_time(self):
        """
        Plot the brightness variation over time.
        """
        if not self.brightness_values:
            print("No brightness data found. Please run find_the_light() first.")
            return
        
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
        """
        Find the start time of the light based on brightness changes.
        """
        if not self.green_values:
            print("No brightness data found. Please run find_the_light() first.")
            return
        
        plateau_threshold = 5
        slope_threshold = 5

        window_size = 10
        smoothed_green = np.convolve(self.green_values, np.ones(window_size)/window_size, mode='valid')

        # Find plateau regions
        plateau_mask = np.abs(np.diff(smoothed_green)) < plateau_threshold
        plateau_regions = np.where(plateau_mask)[0] + 1

        if len(plateau_regions) > 0:
            highest_plateau_idx = np.argmax(smoothed_green[plateau_regions])
            highest_plateau_start = plateau_regions[highest_plateau_idx]
            self.highest_plateau_value = smoothed_green[highest_plateau_start]
            print(f"Highest plateau starts at frame {highest_plateau_start} with value {self.highest_plateau_value}")
        else:
            print("No plateau found")

        # Calculate slope of green channel
        green_slope = np.diff(self.green_values)

        slope_region_start = max(0, highest_plateau_start - window_size)
        slope_region_end = min(len(green_slope), highest_plateau_start + window_size)

        max_slope_idx = np.argmax(np.abs(green_slope[slope_region_start:slope_region_end]))
        self.max_slope_frame = slope_region_start + max_slope_idx

        print(f"Maximum slope change occurs at frame {self.max_slope_frame}")

    def plot_start_time(self):
        """
        Plot the start time of the light.
        """
        if self.highest_plateau_value is None or self.max_slope_frame is None:
            print("Start time data not found. Please run find_start_time() first.")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.green_values, color='green', label='Green Channel Brightness')
        plt.axhline(y=self.highest_plateau_value, color='orange', linestyle='--', label='Highest Plateau Value')
        plt.scatter(self.max_slope_frame, self.green_values[self.max_slope_frame], color='red', label='Max Slope Change', zorder=5)
        plt.xlabel('Frame Number')
        plt.ylabel('Brightness')
        plt.legend()
        plt.show()

    ####################################################
    #manual operation
    
    def find_green_start_time(self, green_threshold):
        """
        Find the first frame where the green channel exceeds the given threshold.
        :param green_threshold: The brightness threshold for the green channel to determine the start time
        """
        if not self.green_values:
            print("No brightness data found. Please run find_the_light() first.")
            return
        
        for frame_number, green_value in enumerate(self.green_values):
            if green_value > green_threshold:
                print(f"First frame exceeding green threshold ({green_threshold}): Frame {frame_number}")
                return frame_number

        print("No frame exceeds the given green threshold.")
        return None
    
    def plot_green_time(self,green_time):
        plt.figure(figsize=(12, 6))
        plt.plot(self.green_values, color='green', label='Green Channel Brightness')
        # plt.axhline(y=self.highest_plateau_value, color='orange', linestyle='--', label='Highest Plateau Value')
        plt.scatter(green_time, self.green_values[green_time], color='red', label='Max Slope Change', zorder=5)
        plt.xlabel('Frame Number')
        plt.ylabel('Brightness')
        plt.legend()
        plt.show()