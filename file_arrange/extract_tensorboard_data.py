import os
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

# 日誌文件所在的目錄
log_dir = r'runs/Mar30_23-30-36_tommy-d4pg_p7_train_1'

# 儲存 CSV 文件的目錄
output_csv_file = 'Mar30_23-30-36_Biceps_femoris_right.csv'

# 要提取的特定標籤
desired_tags = [
    'Filtered_EMG/Biceps_femoris_right'
]

def extract_tensors(log_dir, desired_tags):
    # 收集所有事件文件
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]

    # 用於保存數據的列表
    all_data = []

    for event_file in event_files:
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()

        for tag in desired_tags:
            if tag in event_acc.Tags()['scalars']:
                events = event_acc.Scalars(tag)
                for event in events:
                    all_data.append([event.wall_time, event.step, tag, event.value])

    return all_data

# 提取數據
data = extract_tensors(log_dir, desired_tags)

# 將數據轉換為 DataFrame 並保存為 CSV 文件
df = pd.DataFrame(data, columns=['wall_time', 'step', 'tag', 'value'])
df.to_csv(output_csv_file, index=False)

print(f"Data has been saved to {output_csv_file}")
