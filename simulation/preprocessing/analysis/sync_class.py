import pandas as pd

class Synchronizer:
    def __init__(self, trc_file_path):
        self.trc_file_path = trc_file_path
        self.trc_file = self._load_trc_file()
        self.emg_file = self._load_emg_file()
        self.trc_header = None

    def _load_trc_file(self):    
        try:
            trc_data = pd.read_csv(self.trc_file_path)
        except pd.errors.ParserError:
            print("pandas_error")
            with open(self.trc_file_path, 'r') as file:
                trc_data = file.read()
        pass
        
    
    def _load_emg_file(self):
        pass

    def get_camera_info(self):
        with open(self.trc_file_path, 'r') as file:
            self.trc_header = [next(file) for _ in range(4)]

        camera_info_header = self.trc_header[1].strip().split('\t')
        camera_info_values = self.trc_header[2].strip().split('\t')
        camera_info = dict(zip(camera_info_header, camera_info_values))
        print(camera_info)

    def trc_remove_camera_info(self):
        header_line = self.trc_header[3].strip().split('\t')
        cleaned_header = [col for col in header_line if col != '']
        column_names = ['Frame#', 'Time'] + [f'{marker}_{axis}' for marker in cleaned_header[2:] for axis in ['X', 'Y', 'Z']]
        trc_data = pd.read_csv(self.trc_file_path, sep='\t', skiprows=5, header=None)
        trc_data.columns = column_names[:len(trc_data.columns)]

    def plot_trc_data(self):
        