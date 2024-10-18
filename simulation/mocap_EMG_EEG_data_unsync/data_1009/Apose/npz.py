import numpy as np

data = np.load('D:/data_1009/path1_01/opensim/sync_time_marker.npz')
print(data.files)

array_data = data['sync_timeline']
print(array_data)
print(len(array_data))

array_data = data['marker']
print(array_data)