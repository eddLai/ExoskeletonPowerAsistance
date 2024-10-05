import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, hilbert
import pandas as pd
import glob
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt

class BandFilter:
    def __init__(self, order=4, fc_bp=[20, 480], freq=1000):
        nyq = 0.5 * freq
        low = fc_bp[0] / nyq
        high = fc_bp[1] / nyq
        self.b, self.a = butter(order, (low, high), btype='bandpass', output='ba')
    
    def filtfilt(self, x):
        return filtfilt(self.b, self.a, x)

class LowPassFilter:
    def __init__(self, order=2, fc_bp=4, freq=1000):
        nyq = 0.5 * freq
        low = fc_bp / nyq
        self.b, self.a = butter(order, low, btype='lowpass', output='ba')
    
    def filtfilt(self, x):
        return filtfilt(self.b, self.a, x)

class HighPassFilter:
    def __init__(self, order=2, fc_bp=30, freq=1000):
        nyq = 0.5 * freq
        high = fc_bp / nyq
        self.b, self.a = butter(order, high, btype='highpass', output='ba')
    
    def filtfilt(self, x):
        return filtfilt(self.b, self.a, x)

class NotchFilter:
    def __init__(self, f0=60, freq=1000):
        Q = 30.0  # Quality factor
        self.b, self.a = iirnotch(f0, Q, freq)
    
    def filtfilt(self, x):
        return filtfilt(self.b, self.a, x)

class EMGProcessor:
    def __init__(self, sampling_rate=1000, highpass_order=4, highpass_fc=30,
                 lowpass_order=4, lowpass_fc=4, notch_f0=60):
        self.h_filter = HighPassFilter(order=highpass_order, fc_bp=highpass_fc, freq=sampling_rate)
        self.l_filter = LowPassFilter(order=lowpass_order, fc_bp=lowpass_fc, freq=sampling_rate)
        self.n_filter = NotchFilter(f0=notch_f0, freq=sampling_rate)

    def process(self, signal):
        # Apply high-pass filter
        filted_emg = self.h_filter.filtfilt(signal)
        # Apply notch filter
        filted_emg = self.n_filter.filtfilt(filted_emg)
        # Rectify the EMG signal
        rect_emg = np.abs(filted_emg)
        # Get the envelope using low-pass filter
        envelope = self.l_filter.filtfilt(rect_emg)
        return envelope
    
    def normalize_data(self, data):
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return data
        return (data - min_val) / (max_val - min_val)
    

class EMG_DATA:
    def __init__(self, base_path, file_type='*.csv', sep='\t', range_of_muscle=8):
        self.base_path = base_path
        self.file_type = file_type
        self.sep = sep
        self.range_of_muscle = range_of_muscle
        self.emg_files = self._load_emg_files()
        self.file_info = None
        self.our_data = None
        self.processed_our_data = None
        self.normed_our_data = None
        self.processed_df = None
        self.col2index = {}
        self.processor = EMGProcessor()

    def _load_emg_files(self):
        """
        :return: all avaliable EMG files
        """
        emg_files = glob.glob(os.path.join(self.base_path, 'EMG', self.file_type))
        if not emg_files:
            raise FileNotFoundError(f"at {os.path.join(self.base_path, 'EMG')} found no EMG csv。")
        return emg_files

    def read_emg_file(self, file_index=0):
        """
        :param file_index
        :return: raw csv DataFrame
        """
        if file_index >= len(self.emg_files):
            raise IndexError(f"file {file_index} out of index, only {len(self.emg_files)} files")
        
        emg_file_path = self.emg_files[file_index]
        emg_file = pd.read_csv(emg_file_path, sep=self.sep)
        self.file_info = emg_file.iloc[:10, :]
        emg_file_cut = emg_file.iloc[10:, 0].apply(lambda x: x.split(','))
        columns = self.file_info.iloc[9, 0].split(',')
        self.our_data = pd.DataFrame(emg_file_cut.tolist(), columns=columns)
        return self.our_data

    def process_data(self, normalize=True):
        """
        :param normalize:
        :return: DataFrame, but save in numpy array in class
        """
        if self.our_data is None:
            raise ValueError("no avaliable emg data, read_emg_file(index) first")
        
        RANGE_OF_MUSCLE = self.range_of_muscle
        self.processed_our_data = np.zeros((self.our_data.shape[0], 2 + RANGE_OF_MUSCLE), dtype=float)
        
        self.processed_our_data[:, :2] = self.our_data.iloc[:, :2].values
        
        for i in range(2, 2 + RANGE_OF_MUSCLE):
            column_data = pd.to_numeric(self.our_data.iloc[:, i], errors='coerce').fillna(0)
            processed_signal = self.processor.process(column_data)
            processed_signal[processed_signal < 0] = 0
            self.processed_our_data[:, i] = processed_signal
        
        if normalize:
            self.normed_our_data = np.copy(self.processed_our_data)
            for i in range(2, 2 + RANGE_OF_MUSCLE):
                column_data = self.processed_our_data[:, i]
                self.normed_our_data[:, i] = self.processor.normalize_data(column_data)
            self.processed_df = pd.DataFrame(self.normed_our_data, columns=self.our_data.columns[:2 + RANGE_OF_MUSCLE])
        else:
            self.processed_df = pd.DataFrame(self.processed_our_data, columns=self.our_data.columns[:2 + RANGE_OF_MUSCLE])

        self.columns = self.our_data.columns[:2 + RANGE_OF_MUSCLE]
        self.col2index = {col: idx for idx, col in enumerate(self.columns)}
        
        return self.processed_df

    def save_processed_data(self, output_file_name='post_event_data.csv'):
        """
        :param output_file_name: default as 'post_event_data.csv'
        """
        if self.processed_df is not None:
            output_file_path = os.path.join(self.base_path, 'preprocessing', 'output', output_file_name)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            self.processed_df.to_csv(output_file_path, index=False)
            print(f"saved at: {output_file_path}")
        else:
            raise ValueError("make sure data has been processed")

    def get_event_id_indices(self):
        """
        find out TTL event
        130 red on，131 red off，140 green on, 141 green off
        :return: indexs
        """
        if self.our_data is not None:
            event_id_indices = self.our_data[self.our_data['Event Id'].notna() & (self.our_data['Event Id'] != '')].index
            return event_id_indices
        else:
            raise ValueError("read_emg_file first")

    def display_event_id_data(self):
        event_id_indices = self.get_event_id_indices()
        if not event_id_indices.empty:
            event_id_data = self.our_data.loc[event_id_indices]
            return event_id_data
        else:
            print("Nothing in 'Event Id' column")
            return pd.DataFrame()

    def get_column_mapping(self):
        if self.col2index:
            print("columns name-> index:")
            for col, idx in self.col2index.items():
                print(f"{col}: {idx}")
        else:
            raise ValueError("process_data() first")

    def get_muscle_data(self, muscle_name):
        """
        :param muscle_name:
        :return: nparray
        """
        if self.col2index:
            if muscle_name in self.col2index:
                idx = self.col2index[muscle_name]
                return self.processed_our_data[:, idx]
            else:
                raise ValueError(f"no '{muscle_name}'")
        else:
            raise ValueError("'process_data' will map the index")
        

    def plot_emg(self, muscle_name, start_flag, end_flag, 
             show_raw=True, show_envelope_raw=True, 
             show_processed=True,  
             show_normalized=True,
             additional_data=None,  
             additional_start_flag=None,  
             additional_end_flag=None,  
             figsize=(10, 6)):
        """
        Generalized EMG plotting function with option to overlay additional data using dual x and y axes.
        
        :param muscle_name: Name of the muscle to plot.
        :param start_flag: Start timestamp (in seconds).
        :param end_flag: End timestamp (in seconds).
        :param show_raw: Whether to show raw EMG data.
        :param show_envelope_raw: Whether to show envelope of raw EMG data.
        :param show_processed: Whether to show processed EMG data.
        :param show_normalized: Whether to show normalized processed EMG data.
        :param additional_data: Optional dictionary containing additional data to plot.
                                Example:
                                {
                                    'data': rheel_y_sample,
                                    'label': 'RHeel_Y',
                                    'color': 'blue'
                                }
        :param additional_start_flag: Start timestamp for additional data (in seconds).
        :param additional_end_flag: End timestamp for additional data (in seconds).
        :param figsize: Size of the figure.
        """

        # Validate muscle name
        if muscle_name not in self.col2index:
            raise ValueError(f"Muscle name '{muscle_name}' does not exist. Check available muscle names.")
        
        # Get column indices
        timestamp_idx = self.col2index.get('Timestamp')
        if timestamp_idx is None:
            raise ValueError("Data does not contain 'Timestamp' column.")
        
        muscle_idx = self.col2index[muscle_name]
        
        # Extract raw data
        raw_time = pd.to_numeric(self.our_data['Timestamp'].values)
        raw_data = pd.to_numeric(self.our_data[muscle_name], errors='coerce').fillna(0).values
        mask_raw = (raw_time >= start_flag) & (raw_time <= end_flag)
        filtered_raw_time = raw_time[mask_raw]
        filtered_raw_data = raw_data[mask_raw]
        
        # Initialize plot
        fig, ax1 = plt.subplots(figsize=figsize)
        handles = []
        labels = []

        # Plot raw data only if show_raw is True
        if show_raw and filtered_raw_time is not None and filtered_raw_data is not None:
            line_raw, = ax1.plot(filtered_raw_time, filtered_raw_data, label='Raw EMG Data', color='blue')  # Light blue
            handles.append(line_raw)
            labels.append('Raw EMG Data')

        # Extract processed data (if available)
        processed_time = self.processed_our_data[:, timestamp_idx]  # Assuming processed data exists
        processed_data = self.processed_our_data[:, muscle_idx]
        mask_processed = (processed_time >= start_flag) & (processed_time <= end_flag)
        filtered_processed_time = processed_time[mask_processed]
        filtered_processed_data = processed_data[mask_processed]

        # Plot processed data with a light red color
        if show_processed and filtered_processed_time is not None and filtered_processed_data is not None:
            line_processed, = ax1.plot(filtered_processed_time, filtered_processed_data, 
                                        label='Processed EMG Data', color='blue')  # Light red
            handles.append(line_processed)
            labels.append('Processed EMG Data')
        
        ax1.set_xlabel('Timestamp (EMG Data) (s)')
        ax1.set_ylabel(f'{muscle_name} EMG Data', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Extract normalized data (if available)
        if self.normed_our_data is not None:
            normalized_time = self.normed_our_data[:, timestamp_idx]  # Extract normalized time
            normalized_data = self.normed_our_data[:, muscle_idx]     # Extract normalized muscle data
            mask_normalized = (normalized_time >= start_flag) & (normalized_time <= end_flag)
            filtered_normalized_time = normalized_time[mask_normalized]
            filtered_normalized_data = normalized_data[mask_normalized]

        # Plot normalized data
        ax2 = None
        if show_normalized and filtered_normalized_time is not None and filtered_normalized_data is not None:
            ax2 = ax1.twinx()
            line_normalized, = ax2.plot(filtered_normalized_time, filtered_normalized_data * 0.1,  # Adjusted size
                                        label='Normalized EMG Data', color='red')  # Changed color to avoid conflict
            
            ax2.set_ylabel('Normalized Amplitude (scaled)', color='red')  # 將顏色與線條匹配
            ax2.tick_params(axis='y', labelcolor='red')

        # Plot additional data even when show_raw=False
        if additional_data is not None:
            additional_series = additional_data.get('data')
            additional_label = additional_data.get('label', 'RHeel_Y')
            additional_color = additional_data.get('color', 'saddlebrown')  # 暖褐色作為外部數據顏色
            
            ax4 = ax1.twinx()  # 創建副 y 軸，用於顯示外部數據
            ax4.spines["right"].set_position(("axes", 1.1))  # 移動副 y 軸，避免與另一個 y 軸重疊

            ax3 = ax1.twiny()

            additional_time = pd.to_numeric(additional_series.index.values)
            mask_additional = (additional_time >= additional_start_flag) & (additional_time <= additional_end_flag)
            additional_time_filtered = additional_time[mask_additional]
            additional_values = additional_series.values[mask_additional]
            
            # 使用 raw 的時間範圍縮放
            additional_time_scaled = (additional_time_filtered - np.min(additional_time_filtered)) / (np.max(additional_time_filtered) - np.min(additional_time_filtered))
            additional_time_scaled *= (np.max(filtered_raw_time) - np.min(filtered_raw_time))
            additional_time_scaled += np.min(filtered_raw_time)
            
            # 顯示外部數據的幅值
            line_additional, = ax4.plot(additional_time_scaled, additional_values, label=additional_label, color=additional_color, linestyle='-.')
            
            ax4.set_ylabel(f'{additional_label}', color=additional_color)
            ax4.tick_params(axis='y', labelcolor=additional_color)  # 副 y 軸顏色與線條顏色一致

            # 設置副 x 軸標籤，保留 x 軸
            ax3.set_xlim(np.min(additional_time_filtered), np.max(additional_time_filtered))
            ax3.set_xlabel('Timestamp (External Data) (s)', color=additional_color)
            ax3.tick_params(axis='x', labelcolor=additional_color)
            
            handles.append(line_additional)
            labels.append(additional_label)
        
        # Set title, check if additional_label exists
        if additional_data is not None:
            plt.title(f'{muscle_name} EMG Data (Timestamp {start_flag} to {end_flag}) and {additional_label}')
        else:
            plt.title(f'{muscle_name} EMG Data (Timestamp {start_flag} to {end_flag})')

        
        # Combine legends from both axes
        if ax2:
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles += handles2
            labels += labels2
        ax1.legend(handles, labels, loc='upper right')
        
        # Optimize layout and add grid
        fig.tight_layout()
        plt.grid(True)
        plt.show()


    def __str__(self):
        lines = [
            f"EMG_DATA at: {self.base_path}",
            f"File type: {self.file_type}, Separator: '{self.sep}', Muscle range: {self.range_of_muscle}",
            f"Loaded EMG files ({len(self.emg_files)}): {', '.join(self.emg_files)}" if self.emg_files else "No EMG files loaded.",
            f"Raw data: {self.our_data.shape[0]} rows x {self.our_data.shape[1]} columns." if self.our_data is not None else "No raw data loaded.",
            f"Processed data: {self.processed_our_data.shape[0]} rows x {self.processed_our_data.shape[1]} columns." if self.processed_our_data is not None else "No data processed.",
            "Data normalized." if self.normed_our_data is not None else "Data not normalized.",
            f"Processed DataFrame: {self.processed_df.shape[0]} rows x {self.processed_df.shape[1]} columns." if self.processed_df is not None else "Processed DataFrame not created."
        ]
        return '\n'.join(lines)



    